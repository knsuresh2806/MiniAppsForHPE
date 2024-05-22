#include "haloManager3D_gpu.h"
#include "haloManager3D_gpu_kernels.cuh"
#include <unistd.h>

haloManager3D_gpu::haloManager3D_gpu(decompositionManager3D* decomp, std::vector<cart_volume<realtype>**>& data,
                                     std::vector<std::vector<int>> wSz[3][2],
                                     std::vector<std::vector<bool>>&, // includeEdges, ignored (true)
                                     std::vector<std::vector<bool>>&) // includeCorners, ignored (true)
{
    MPI_Comm_dup(decomp->get_comm(), &comm);
    int rank = decomp->get_rank();
    int npes = decomp->get_npes();

    // Check if the topology is compatible with a sequential X->Y->Z exchange
    EMSL_VERIFY(consistent_halos(decomp, wSz));

    // *************** Local subdomains halo exchanges, X dimension ***************

    // X dimension
    create_graph_local_exchanges<0>(exchange_X_count, graph_X_exchange, decomp, data, wSz);

    // Y dimension
    create_graph_local_exchanges<1>(exchange_Y_count, graph_Y_exchange, decomp, data, wSz);

    // Z dimension
    create_graph_local_exchanges<2>(exchange_Z_count, graph_Z_exchange, decomp, data, wSz);

    // *************** MPI exchanges ***************

    // X dimension
    create_graph_mpi_exchanges<0>(mpi_msgs_X, mpi_ack_X, graph_X_packing, graph_X_unpacking, dest_ptr_xlo, dest_ptr_xhi,
                                  launch_x_unpacking, decomp, data, wSz);

    // Y dimension
    create_graph_mpi_exchanges<1>(mpi_msgs_Y, mpi_ack_Y, graph_Y_packing, graph_Y_unpacking, dest_ptr_ylo, dest_ptr_yhi,
                                  launch_y_unpacking, decomp, data, wSz);

    // Z dimension
    create_graph_mpi_exchanges<2>(mpi_msgs_Z, mpi_ack_Z, graph_Z_packing, graph_Z_unpacking, dest_ptr_zlo, dest_ptr_zhi,
                                  launch_z_unpacking, decomp, data, wSz);

    // **************** End of MPI exchanges discovery ****************

    // Create events to synchronize packing and MPI Sends
    CUDA_TRY(cudaEventCreate(&end_packing_X));
    CUDA_TRY(cudaEventCreate(&end_packing_Y));
    CUDA_TRY(cudaEventCreate(&end_packing_Z));

    // Resize the vector of requests for 2 requests (send recv) per message exchange
    requests_X.resize(2 * mpi_msgs_X.size());
    requests_Y.resize(2 * mpi_msgs_Y.size());
    requests_Z.resize(2 * mpi_msgs_Z.size());

    ack_requests_X.resize(2 * mpi_ack_X.size());
    ack_requests_Y.resize(2 * mpi_ack_Y.size());
    ack_requests_Z.resize(2 * mpi_ack_Z.size());

} //end constructor

haloManager3D_gpu::~haloManager3D_gpu()
{
    MPI_Comm_free(&comm);

    CUDA_TRY(cudaEventDestroy(end_packing_X));
    CUDA_TRY(cudaEventDestroy(end_packing_Y));

    // Destroy graphexecs
    if (exchange_X_count)
        CUDA_TRY(cudaGraphExecDestroy(graph_X_exchange));
    if (exchange_Y_count)
        CUDA_TRY(cudaGraphExecDestroy(graph_Y_exchange));
    if (exchange_Z_count)
        CUDA_TRY(cudaGraphExecDestroy(graph_Z_exchange));
    if (mpi_msgs_X.size()) {
        CUDA_TRY(cudaGraphExecDestroy(graph_X_packing));
        if (launch_x_unpacking)
            CUDA_TRY(cudaGraphExecDestroy(graph_X_unpacking));
    }
    if (mpi_msgs_Y.size()) {
        CUDA_TRY(cudaGraphExecDestroy(graph_Y_packing));
        if (launch_y_unpacking)
            CUDA_TRY(cudaGraphExecDestroy(graph_Y_unpacking));
    }
    if (mpi_msgs_Z.size()) {
        CUDA_TRY(cudaGraphExecDestroy(graph_Z_packing));
        if (launch_z_unpacking)
            CUDA_TRY(cudaGraphExecDestroy(graph_Z_unpacking));
    }
    // Destroy comm buffers (device memory, unless using IPC, then host memory ACKs)
    for (auto msg : mpi_msgs_X) {
        release_commbuf(msg.send_buf);
        release_commbuf(msg.recv_buf);
    }
    for (auto msg : mpi_msgs_Y) {
        release_commbuf(msg.send_buf);
        release_commbuf(msg.recv_buf);
    }
    for (auto msg : mpi_msgs_Z) {
        release_commbuf(msg.send_buf);
        release_commbuf(msg.recv_buf);
    }
    for (auto msg : mpi_ack_X) {
        free(msg.send_buf);
        free(msg.recv_buf);
    }
    for (auto msg : mpi_ack_Y) {
        free(msg.send_buf);
        free(msg.recv_buf);
    }
    for (auto msg : mpi_ack_Z) {
        free(msg.send_buf);
        free(msg.recv_buf);
    }
    // Close IPC handles, if any have been used
    for (auto ptr : dest_ptr_xlo)
        CUDA_TRY(cudaIpcCloseMemHandle(ptr));
    for (auto ptr : dest_ptr_xhi)
        CUDA_TRY(cudaIpcCloseMemHandle(ptr));
    for (auto ptr : dest_ptr_ylo)
        CUDA_TRY(cudaIpcCloseMemHandle(ptr));
    for (auto ptr : dest_ptr_yhi)
        CUDA_TRY(cudaIpcCloseMemHandle(ptr));
    for (auto ptr : dest_ptr_zlo)
        CUDA_TRY(cudaIpcCloseMemHandle(ptr));
    for (auto ptr : dest_ptr_zhi)
        CUDA_TRY(cudaIpcCloseMemHandle(ptr));
}

// Create a CUDA graph to launch all the local exchange kernels for a given dimension
// This will exchange halos between subvolumes that are on the same MPI rank
template <int dim>
void
haloManager3D_gpu::create_graph_local_exchanges(int& exchange_count,        // OUT: number of nodes in the graph
                                                cudaGraphExec_t& graphexec, // OUT: CUDA graph
                                                decompositionManager3D* decomp,
                                                std::vector<cart_volume<realtype>**>& data,
                                                std::vector<std::vector<int>> wSz[3][2])
{

    constexpr NH_ID_3D neg_sides[] = { _NH_ID_3D_NEGX_CENY_CENZ, _NH_ID_3D_CENX_NEGY_CENZ, _NH_ID_3D_CENX_CENY_NEGZ };
    constexpr void (*exchange_funcs[])(cart_volume<T>*, cart_volume<T>*, int, int,
                                       cudaStream_t) = { exchange_halos_X, exchange_halos_Y, exchange_halos_Z };
    constexpr auto neg_side = neg_sides[dim];
    constexpr auto exchange_func = exchange_funcs[dim];

    int nfields = data.size();
    int nsubdom = decomp->getTotalNumLocalSubDom();

    // Allocate streams and events to create the graph
    int nstreams = nsubdom * nfields;
    cudaStream_t* graph_streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
    cudaEvent_t* graph_events = (cudaEvent_t*)malloc(nstreams * sizeof(cudaEvent_t));
    for (int i = 0; i < nstreams; i++) {
        CUDA_TRY(cudaStreamCreate(&graph_streams[i]));
        CUDA_TRY(cudaEventCreate(&graph_events[i]));
    }

    // NOTE: CUDA doesn't seem to like to record empty graphs with no kernel calls
    exchange_count = 0;

    // Find all the local subdomains that need to communicate in this dimension and this side
    for (int j = 0; j < nsubdom; ++j) {
        int nhSubDomId[_NH_ID_3D_TOTAL];
        decomp->getLocalNeighborID(j, nhSubDomId);
        int nbr_id = nhSubDomId[neg_side];
        if (nbr_id >= 0) {
            // This subdomain has a local lower neighbor
            for (int i = 0; i < nfields; ++i) {
                // Start the graph capture if this is the first exchange
                if (exchange_count == 0) {
                    CUDA_TRY(cudaStreamBeginCapture(graph_streams[0], cudaStreamCaptureModeRelaxed));
                    cudaEventRecord(graph_events[0], graph_streams[0]);
                } else
                    // Express a dependency between this graph node and graph_streams[0]
                    CUDA_TRY(cudaStreamWaitEvent(graph_streams[exchange_count], graph_events[0]));
                // Launch the dedicated 2-way copy in a dedicated stream
                exchange_func(data[i][nbr_id],        // neighbor volume,
                              data[i][j],             // this volume
                              wSz[dim][0][i][j],      // Low halo = negative halo[dim] of this volume
                              wSz[dim][1][i][nbr_id], // Positive halo[dim] of the negative volume
                              graph_streams[exchange_count]);
                // Express the inverse dependency between this graph node and graph_streams[0]
                if (exchange_count > 0) {
                    CUDA_TRY(cudaEventRecord(graph_events[exchange_count], graph_streams[exchange_count]));
                    CUDA_TRY(cudaStreamWaitEvent(graph_streams[0], graph_events[exchange_count]));
                }
                exchange_count++;
            }
        }
    }
    if (exchange_count) {
        cudaGraph_t graph;
        CUDA_TRY(cudaStreamEndCapture(graph_streams[0], &graph));
        CUDA_TRY(cudaGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));
        CUDA_TRY(cudaGraphDestroy(graph));
    }

    // Clean up streams & events
    for (int i = 0; i < nstreams; i++) {
        CUDA_TRY(cudaStreamDestroy(graph_streams[i]));
        CUDA_TRY(cudaEventDestroy(graph_events[i]));
    }
    free(graph_streams);
    free(graph_events);
}

// Create CUDA graphs to launch all the MPI packing/unpacking kernels for a given dimension.
// This will exchange halos between subvolumes on different ranks.
// Pack and unpack graphs are built simultaneously, the MPI communication buffers (device memory)
// are allocated, and the parameters of the MPI exchanges are added in the vectors.
template <int dim>
void
haloManager3D_gpu::create_graph_mpi_exchanges(std::vector<mpi_msg>& mpi_msg_vec, // OUT: MPI messages vector
                                              std::vector<mpi_msg>& mpi_ack_vec, // OUT: MPI acknowledgement vector
                                              cudaGraphExec_t& graph_pack,       // OUT: CUDA packing graph
                                              cudaGraphExec_t& graph_unpack,     // OUT: CUDA unpacking graph
                                              std::vector<T*>& ipc_vec_lo,       // OUT : IPC ptrs vector lower neighbor
                                              std::vector<T*>& ipc_vec_hi, // OUT : IPC ptrs vector higher neighbor
                                              bool& launch_unpacking,      // OUT: boolean if unpacking graph exists
                                              decompositionManager3D* decomp,
                                              std::vector<cart_volume<realtype>**>& data,
                                              std::vector<std::vector<int>> wSz[3][2])
{
    // Dim-specific values and functions
    constexpr NH_ID_3D neg_sides[] = { _NH_ID_3D_NEGX_CENY_CENZ, _NH_ID_3D_CENX_NEGY_CENZ, _NH_ID_3D_CENX_CENY_NEGZ };
    constexpr NH_ID_3D pos_sides[] = { _NH_ID_3D_POSX_CENY_CENZ, _NH_ID_3D_CENX_POSY_CENZ, _NH_ID_3D_CENX_CENY_POSZ };
    typedef int (*pfunc_getsize)(cart_volume<T>*, int); // Pointer to a get_size_combuf function
    constexpr pfunc_getsize get_size_funcs[] = { get_size_combuf_X, get_size_combuf_Y, get_size_combuf_Z };
    typedef void (*pfunc_copy)(cart_volume<T>*, T*, int, cudaStream_t); // Pointer to a copy_halos function
    constexpr pfunc_copy lo_pack_funcs[] = { copy_halos_X<true, true>, copy_halos_Y<true, true>,
                                             copy_halos_Z<true, true> };
    constexpr pfunc_copy hi_pack_funcs[] = { copy_halos_X<false, true>, copy_halos_Y<false, true>,
                                             copy_halos_Z<false, true> };
    constexpr pfunc_copy lo_unpack_funcs[] = { copy_halos_X<true, false>, copy_halos_Y<true, false>,
                                               copy_halos_Z<true, false> };
    constexpr pfunc_copy hi_unpack_funcs[] = { copy_halos_X<false, false>, copy_halos_Y<false, false>,
                                               copy_halos_Z<false, false> };
    typedef void (*pfunc_ipc_copy)(cart_volume<T>*, T*, int, int, int,
                                   cudaStream_t); // Pointer to IPC copy_halos function
    constexpr pfunc_ipc_copy lo_ipc_copy_funcs[] = { copy_halos_X<true>, copy_halos_Y<true>, copy_halos_Z<true> };
    constexpr pfunc_ipc_copy hi_ipc_copy_funcs[] = { copy_halos_X<false>, copy_halos_Y<false>, copy_halos_Z<false> };

    constexpr auto neg_side = neg_sides[dim];
    constexpr auto pos_side = pos_sides[dim];
    constexpr auto get_size_func = get_size_funcs[dim];
    constexpr auto lo_pack_func = lo_pack_funcs[dim];
    constexpr auto hi_pack_func = hi_pack_funcs[dim];
    constexpr auto lo_unpack_func = lo_unpack_funcs[dim];
    constexpr auto hi_unpack_func = hi_unpack_funcs[dim];
    constexpr auto lo_ipc_copy_func = lo_ipc_copy_funcs[dim];
    constexpr auto hi_ipc_copy_func = hi_ipc_copy_funcs[dim];

    int nfields = data.size();
    int nsubdom = decomp->getTotalNumLocalSubDom();
    int ntot_fields = nfields * nsubdom;

    // MPI requests to echange hostnames or IPC info
    MPI_Request requests[4];
    int nr;

    // Get MPI neighbors on both sides
    int nhRank[_NH_ID_3D_TOTAL];
    decomp->getNeighbors(nhRank);
    int lo_rank = nhRank[neg_side];
    int hi_rank = nhRank[pos_side];
    if (lo_rank < 0 && hi_rank < 0)
        return;

    // Check if the neighbors are on the same node
    char hostname[HOST_NAME_MAX + 1], lo_hostname[HOST_NAME_MAX + 1], hi_hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);
    nr = 0;
    if (lo_rank >= 0) {
        MPI_Irecv(lo_hostname, HOST_NAME_MAX + 1, MPI_CHAR, lo_rank, 100, decomp->get_comm(), &requests[nr++]);
        MPI_Isend(hostname, HOST_NAME_MAX + 1, MPI_CHAR, lo_rank, 100, decomp->get_comm(), &requests[nr++]);
    }
    if (hi_rank >= 0) {
        MPI_Irecv(hi_hostname, HOST_NAME_MAX + 1, MPI_CHAR, hi_rank, 100, decomp->get_comm(), &requests[nr++]);
        MPI_Isend(hostname, HOST_NAME_MAX + 1, MPI_CHAR, hi_rank, 100, decomp->get_comm(), &requests[nr++]);
    }
    if (nr)
        MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
    bool lo_samehost = (lo_rank >= 0 && strcmp(hostname, lo_hostname) == 0);
    bool hi_samehost = (hi_rank >= 0 && strcmp(hostname, hi_hostname) == 0);

    // Allocate buffers to keep track of buffers exchanged with CUDA IPC instead of MPI
    cuda_ipc_info_t *ipc_infos_lo, *ipc_send_infos_lo;
    cuda_ipc_info_t *ipc_infos_hi, *ipc_send_infos_hi;
    if (lo_samehost || hi_samehost) {
        if (lo_samehost) {
            ipc_send_infos_lo = (cuda_ipc_info_t*)malloc(ntot_fields * sizeof(cuda_ipc_info_t));
            ipc_infos_lo = (cuda_ipc_info_t*)malloc(ntot_fields * sizeof(cuda_ipc_info_t));
        }
        if (hi_samehost) {
            ipc_send_infos_hi = (cuda_ipc_info_t*)malloc(ntot_fields * sizeof(cuda_ipc_info_t));
            ipc_infos_hi = (cuda_ipc_info_t*)malloc(ntot_fields * sizeof(cuda_ipc_info_t));
        }
    }
    // Neighbor nodes can have a different value of ntot_fields, but they will agree
    // on the number of buffers to echange with IPC.
    int nbuf_ipc_lo = 0;
    int nbuf_ipc_hi = 0;

    // First pass:
    // Compute the total size of the communication buffers,
    // or for same-node neighbors, gather IPC info to send to the neighbors
    size_t lo_send_buf_sz = 0;
    size_t lo_recv_buf_sz = 0;
    size_t hi_send_buf_sz = 0;
    size_t hi_recv_buf_sz = 0;
    for (int j = 0; j < nsubdom; ++j) {
        int nhSubDomId[_NH_ID_3D_TOTAL];
        decomp->getLocalNeighborID(j, nhSubDomId);
        if (lo_rank >= 0 && nhSubDomId[neg_side] < 0) {
            // This subdomain has an MPI neighbor on its lower side
            for (int i = 0; i < nfields; ++i) {
                int radius_recv = wSz[dim][0][i][j]; // negative halo
                int radius_send = wSz[dim][1][i][j]; // positive halo
                if (lo_samehost)
                    get_cuda_ipc_info<dim, true>(&ipc_send_infos_lo[nbuf_ipc_lo++], data[i][j], radius_recv);
                else {
                    lo_send_buf_sz += get_size_func(data[i][j], radius_send);
                    lo_recv_buf_sz += get_size_func(data[i][j], radius_recv);
                }
            }
        }
        if (hi_rank >= 0 && nhSubDomId[pos_side] < 0) {
            // This subdomain has an MPI neighbor on its higher side
            for (int i = 0; i < nfields; ++i) {
                int radius_recv = wSz[dim][1][i][j]; // positive halo
                int radius_send = wSz[dim][0][i][j]; // negative halo
                if (hi_samehost)
                    get_cuda_ipc_info<dim, false>(&ipc_send_infos_hi[nbuf_ipc_hi++], data[i][j], radius_recv);
                else {
                    hi_send_buf_sz += get_size_func(data[i][j], radius_send);
                    hi_recv_buf_sz += get_size_func(data[i][j], radius_recv);
                }
            }
        }
    }

    if (lo_send_buf_sz == 0 && lo_recv_buf_sz == 0 && hi_send_buf_sz == 0 && hi_recv_buf_sz == 0 && nbuf_ipc_lo == 0 &&
        nbuf_ipc_hi == 0)
        return;

    // Allocate contiguous send and recv buffers, add an entry in the MPI msgs vector for this send & recv
    T *lo_send_buf, *lo_recv_buf, *hi_send_buf, *hi_recv_buf;
    if (lo_send_buf_sz > 0 || lo_recv_buf_sz > 0) {
        lo_send_buf_sz *= sizeof(T);
        lo_recv_buf_sz *= sizeof(T);
        CUDA_TRY(cudaMalloc((void**)&lo_send_buf, lo_send_buf_sz));
        CUDA_TRY(cudaMalloc((void**)&lo_recv_buf, lo_recv_buf_sz));
        mpi_msg_vec.push_back(mpi_msg(lo_rank, lo_send_buf_sz, lo_recv_buf_sz, lo_send_buf, lo_recv_buf, 911));
    }
    if (hi_send_buf_sz > 0 || hi_recv_buf_sz > 0) {
        hi_send_buf_sz *= sizeof(T);
        hi_recv_buf_sz *= sizeof(T);
        CUDA_TRY(cudaMalloc((void**)&hi_send_buf, hi_send_buf_sz));
        CUDA_TRY(cudaMalloc((void**)&hi_recv_buf, hi_recv_buf_sz));
        mpi_msg_vec.push_back(mpi_msg(hi_rank, hi_send_buf_sz, hi_recv_buf_sz, hi_send_buf, hi_recv_buf, 911));
    }
    // For IPC exchanges, we only send/recv an ACK that the node is ready to receive and the data has been copied, size = 1
    if (nbuf_ipc_lo > 0) {
        lo_send_buf_sz = sizeof(T);
        lo_recv_buf_sz = sizeof(T);
        lo_send_buf = (T*)malloc(lo_send_buf_sz);
        lo_recv_buf = (T*)malloc(lo_recv_buf_sz);
        mpi_msg_vec.push_back(mpi_msg(lo_rank, lo_send_buf_sz, lo_recv_buf_sz, lo_send_buf, lo_recv_buf, 911));

        T* lo_send_ack = (T*)malloc(sizeof(T));
        T* lo_recv_ack = (T*)malloc(sizeof(T));
        mpi_ack_vec.push_back(mpi_msg(lo_rank, sizeof(T), sizeof(T), lo_send_ack, lo_recv_ack, 912));
    }
    if (nbuf_ipc_hi > 0) {
        hi_send_buf_sz = sizeof(T);
        hi_recv_buf_sz = sizeof(T);
        hi_send_buf = (T*)malloc(hi_send_buf_sz);
        hi_recv_buf = (T*)malloc(hi_recv_buf_sz);
        mpi_msg_vec.push_back(mpi_msg(hi_rank, hi_send_buf_sz, hi_recv_buf_sz, hi_send_buf, hi_recv_buf, 911));

        T* hi_send_ack = (T*)malloc(sizeof(T));
        T* hi_recv_ack = (T*)malloc(sizeof(T));
        mpi_ack_vec.push_back(mpi_msg(hi_rank, sizeof(T), sizeof(T), hi_send_ack, hi_recv_ack, 912));
    }

    // Echange IPC info with the neighbors
    nr = 0;
    if (nbuf_ipc_lo) {
        MPI_Irecv(ipc_infos_lo, nbuf_ipc_lo * sizeof(cuda_ipc_info_t), MPI_BYTE, lo_rank, 101, decomp->get_comm(),
                  &requests[nr++]);
        MPI_Isend(ipc_send_infos_lo, nbuf_ipc_lo * sizeof(cuda_ipc_info_t), MPI_BYTE, lo_rank, 101, decomp->get_comm(),
                  &requests[nr++]);
    }
    if (nbuf_ipc_hi) {
        MPI_Irecv(ipc_infos_hi, nbuf_ipc_hi * sizeof(cuda_ipc_info_t), MPI_BYTE, hi_rank, 101, decomp->get_comm(),
                  &requests[nr++]);
        MPI_Isend(ipc_send_infos_hi, nbuf_ipc_hi * sizeof(cuda_ipc_info_t), MPI_BYTE, hi_rank, 101, decomp->get_comm(),
                  &requests[nr++]);
    }
    if (nr)
        MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
    if (lo_samehost)
        free(ipc_send_infos_lo);
    if (hi_samehost)
        free(ipc_send_infos_hi);

    // Second pass:
    // Same loops again, now that we have a buffer, record the graphs
    size_t lo_send_off = 0;
    size_t lo_recv_off = 0;
    size_t hi_send_off = 0;
    size_t hi_recv_off = 0;
    int ibuf_ipc_lo = 0;
    int ibuf_ipc_hi = 0;

    // Each subvolume can launch 2 x packing or unpacking kernels -> 2 streams per subvolume
    int nstreams = 2 * nsubdom * nfields;
    cudaStream_t* packing_streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
    cudaEvent_t* packing_events = (cudaEvent_t*)malloc(nstreams * sizeof(cudaEvent_t));
    cudaStream_t* unpacking_streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
    cudaEvent_t* unpacking_events = (cudaEvent_t*)malloc(nstreams * sizeof(cudaEvent_t));
    for (int i = 0; i < nstreams; i++) {
        CUDA_TRY(cudaStreamCreate(&packing_streams[i]));
        CUDA_TRY(cudaEventCreate(&packing_events[i]));
        CUDA_TRY(cudaStreamCreate(&unpacking_streams[i]));
        CUDA_TRY(cudaEventCreate(&unpacking_events[i]));
    }

    // Start the recording of the packing / unpacking graphs on stream 0, mark this node with an event
    // Same-node IPC exchanges are treated as packing kernels, there is no unpacking.
    CUDA_TRY(cudaStreamBeginCapture(packing_streams[0], cudaStreamCaptureModeRelaxed));
    cudaEventRecord(packing_events[0], packing_streams[0]);
    if (!lo_samehost || !hi_samehost) { // No unpacking if neighbors are on the same node
        CUDA_TRY(cudaStreamBeginCapture(unpacking_streams[0], cudaStreamCaptureModeRelaxed));
        cudaEventRecord(unpacking_events[0], unpacking_streams[0]);
    }
    int istream = 0;

    for (int j = 0; j < nsubdom; ++j) {
        int nhSubDomId[_NH_ID_3D_TOTAL];
        decomp->getLocalNeighborID(j, nhSubDomId);
        if (lo_rank >= 0 && nhSubDomId[neg_side] < 0) {
            // This subdomain has a negative MPI neighbor
            for (int i = 0; i < nfields; ++i) {
                int radius_recv = wSz[dim][0][i][j]; // negative halo
                int radius_send = wSz[dim][1][i][j]; // positive halo

                // Express dependency from this stream to the root event of the graph
                if (istream) {
                    // Add kernels to the graphs, expressing dependencies with first node of the graphs
                    CUDA_TRY(cudaStreamWaitEvent(packing_streams[istream], packing_events[0]));
                    if (!lo_samehost)
                        CUDA_TRY(cudaStreamWaitEvent(unpacking_streams[istream], unpacking_events[0]));
                }
                // Add the kernels to the graphs
                if (lo_samehost) {
                    // Use IPC info, extract pointer from the handle
                    T* dest_ptr;
                    CUDA_TRY(cudaIpcOpenMemHandle((void**)&dest_ptr, ipc_infos_lo[ibuf_ipc_lo].handle,
                                                  cudaIpcMemLazyEnablePeerAccess));
                    int dest_ldimx = ipc_infos_lo[ibuf_ipc_lo].ldimx;
                    int dest_ldimy = ipc_infos_lo[ibuf_ipc_lo].ldimy;
                    // Keep original pointer to IPC-unmap in destructor
                    ipc_vec_lo.push_back(dest_ptr);
                    dest_ptr += ipc_infos_lo[ibuf_ipc_lo].offset;
                    ibuf_ipc_lo++;
                    // Push kernel treated as a packing kernel, before MPI message
                    lo_ipc_copy_func(data[i][j], dest_ptr, dest_ldimx, dest_ldimy, radius_send,
                                     packing_streams[istream]); // Low side push kernel
                } else {
                    lo_pack_func(data[i][j], lo_send_buf + lo_send_off, radius_send, packing_streams[istream]);
                    lo_unpack_func(data[i][j], lo_recv_buf + lo_recv_off, radius_recv, unpacking_streams[istream]);
                    lo_send_off += get_size_func(data[i][j], radius_send);
                    lo_recv_off += get_size_func(data[i][j], radius_recv);
                }

                // Express the inverse dependency with first nodes of the graphs
                if (istream) {
                    CUDA_TRY(cudaEventRecord(packing_events[istream], packing_streams[istream]));
                    CUDA_TRY(cudaStreamWaitEvent(packing_streams[0], packing_events[istream]));
                    if (!lo_samehost) {
                        CUDA_TRY(cudaEventRecord(unpacking_events[istream], unpacking_streams[istream]));
                        CUDA_TRY(cudaStreamWaitEvent(unpacking_streams[0], unpacking_events[istream]));
                    }
                }
                istream++;
            }
        }
        if (hi_rank >= 0 && nhSubDomId[pos_side] < 0) {
            // This subdomain has a positive MPI neighbor
            for (int i = 0; i < nfields; ++i) {
                int radius_recv = wSz[dim][1][i][j]; // positive halo
                int radius_send = wSz[dim][0][i][j]; // negative halo

                // Express dependency from this stream to the root event of the graph
                if (istream) {
                    // Add kernels to the graphs, expressing dependencies with first node of the graphs
                    CUDA_TRY(cudaStreamWaitEvent(packing_streams[istream], packing_events[0]));
                    if (!hi_samehost)
                        CUDA_TRY(cudaStreamWaitEvent(unpacking_streams[istream], unpacking_events[0]));
                }
                // Add the kernels to the graphs
                if (hi_samehost) {
                    // Use IPC info, extract pointer from the handle
                    T* dest_ptr;
                    CUDA_TRY(cudaIpcOpenMemHandle((void**)&dest_ptr, ipc_infos_hi[ibuf_ipc_hi].handle,
                                                  cudaIpcMemLazyEnablePeerAccess));
                    int dest_ldimx = ipc_infos_hi[ibuf_ipc_hi].ldimx;
                    int dest_ldimy = ipc_infos_hi[ibuf_ipc_hi].ldimy;
                    // Keep original pointer to IPC-unmap in destructor
                    ipc_vec_hi.push_back(dest_ptr);
                    dest_ptr += ipc_infos_hi[ibuf_ipc_hi].offset;
                    ibuf_ipc_hi++;
                    // Push kernel treated as a packing kernel, before MPI message
                    hi_ipc_copy_func(data[i][j], dest_ptr, dest_ldimx, dest_ldimy, radius_send,
                                     packing_streams[istream]); // High side push kernel
                } else {
                    hi_pack_func(data[i][j], hi_send_buf + hi_send_off, radius_send, packing_streams[istream]);
                    hi_unpack_func(data[i][j], hi_recv_buf + hi_recv_off, radius_recv, unpacking_streams[istream]);
                    hi_send_off += get_size_func(data[i][j], radius_send);
                    hi_recv_off += get_size_func(data[i][j], radius_recv);
                }

                // Express the inverse dependency with first nodes of the graphs
                if (istream) {
                    CUDA_TRY(cudaEventRecord(packing_events[istream], packing_streams[istream]));
                    CUDA_TRY(cudaStreamWaitEvent(packing_streams[0], packing_events[istream]));
                    if (!hi_samehost) {
                        CUDA_TRY(cudaEventRecord(unpacking_events[istream], unpacking_streams[istream]));
                        CUDA_TRY(cudaStreamWaitEvent(unpacking_streams[0], unpacking_events[istream]));
                    }
                }
                istream++;
            }
        }
    }

    // Finalize the graphs
    launch_unpacking = false;
    if (istream) {
        cudaGraph_t graph;
        CUDA_TRY(cudaStreamEndCapture(packing_streams[0], &graph));
        CUDA_TRY(cudaGraphInstantiate(&graph_pack, graph, nullptr, nullptr, 0));
        CUDA_TRY(cudaGraphDestroy(graph));
        if (!lo_samehost || !hi_samehost) {
            CUDA_TRY(cudaStreamEndCapture(unpacking_streams[0], &graph));
            CUDA_TRY(cudaGraphInstantiate(&graph_unpack, graph, nullptr, nullptr, 0));
            CUDA_TRY(cudaGraphDestroy(graph));
            launch_unpacking = true;
        }
    }

    // Clean up streams & events
    for (int i = 0; i < nstreams; i++) {
        CUDA_TRY(cudaStreamDestroy(packing_streams[i]));
        CUDA_TRY(cudaEventDestroy(packing_events[i]));
        CUDA_TRY(cudaStreamDestroy(unpacking_streams[i]));
        CUDA_TRY(cudaEventDestroy(unpacking_events[i]));
    }
    free(packing_streams);
    free(packing_events);
    free(unpacking_streams);
    free(unpacking_events);

    if (lo_samehost)
        free(ipc_infos_lo);
    if (hi_samehost)
        free(ipc_infos_hi);
}

// Rationale for the split between start_update and finish_update:
// On GPU, the halos are exchanged sequentially in X, then Y, then Z.
// Ideally, the MPI domain decomposition used is only 1D or 3D, not 3D, so in the timeline
// below the MPI exchanges in X are in parenthesis, assuming they don't exist.
// If they do exist, it's not going to be optimal.
// The timeline is :
//    (packing_kernels_X)
//    local_exchange_X
//    (MPI_Send+Recv_X)   (can start as soon as packing_kernels_X is done)
//    (unpack_kernels_X)
//    packing_kernels_Y
//    local_exchange_Y
// ------------------------- split here between start_update() and finish_update() with 2D decomposition.
//    MPI_Send+Recv_Y     (can start as soon as packing_kernels_Y is done)
//    unpack_kernels_Y
//    packing_kernels_Z
//    local_exchange_Z
// ------------------------- split here for 1D domain decomposition
//    MPI_Send+Recv_Z
//    unpack_kernels_Z
//
// If the MPI domain decomposition is 1D, then the split between start_update and finish_update
// should be right before the MPI Z exchanges.
//
// If the domain decomposition is 2D (or 3D), the split will happen right before the Y MPI exchanges.
// The Y unpacking + Z kernels will then compete to get on the GPU with whatever other GPU workload is
// trying to overlap with the communication, but this can be solved by running the halo exchanges
// in a dedicated high priority stream.

void
haloManager3D_gpu::start_update(cudaStream_t stream)
{
    int n_msgs_X = mpi_msgs_X.size();
    int n_msgs_Y = mpi_msgs_Y.size();
    int n_msgs_Z = mpi_msgs_Z.size();

    int n_ack_X = mpi_ack_X.size();
    int n_ack_Y = mpi_ack_Y.size();
    int n_ack_Z = mpi_ack_Z.size();

    // This will guarantee that we will not start receiving any MPI data
    // before all of the previous kernels in this stream are finished.
    // For X and Y, the cart_volume will be updated until
    // graph_X_unpacking and graph_Y_unpacking.
    // This fix should be temporary until Z is fixed or further optimization.
    CUDA_TRY(cudaStreamSynchronize(stream));
    // First, post all the async receives for all X Y Z exchanges.
    // It helps MPI quickly figure out where data goes during sends.
    for (int i = 0; i < n_msgs_X; i++)
        MPI_Irecv(mpi_msgs_X[i].recv_buf, mpi_msgs_X[i].recv_size, MPI_BYTE, mpi_msgs_X[i].rank, mpi_msgs_X[i].tag,
                  comm, &requests_X[i]);

    for (int i = 0; i < n_msgs_Y; i++)
        MPI_Irecv(mpi_msgs_Y[i].recv_buf, mpi_msgs_Y[i].recv_size, MPI_BYTE, mpi_msgs_Y[i].rank, mpi_msgs_Y[i].tag,
                  comm, &requests_Y[i]);

    for (int i = 0; i < n_msgs_Z; i++)
        MPI_Irecv(mpi_msgs_Z[i].recv_buf, mpi_msgs_Z[i].recv_size, MPI_BYTE, mpi_msgs_Z[i].rank, mpi_msgs_Z[i].tag,
                  comm, &requests_Z[i]);

    // Post receives to know the neighbor rank is ready to receive IPC data
    for (int i = 0; i < n_ack_X; i++)
        MPI_Irecv(mpi_ack_X[i].recv_buf, mpi_ack_X[i].recv_size, MPI_BYTE, mpi_ack_X[i].rank, mpi_ack_X[i].tag, comm,
                  &ack_requests_X[i]);

    for (int i = 0; i < n_ack_Y; i++)
        MPI_Irecv(mpi_ack_Y[i].recv_buf, mpi_ack_Y[i].recv_size, MPI_BYTE, mpi_ack_Y[i].rank, mpi_ack_Y[i].tag, comm,
                  &ack_requests_Y[i]);

    for (int i = 0; i < n_ack_Z; i++)
        MPI_Irecv(mpi_ack_Z[i].recv_buf, mpi_ack_Z[i].recv_size, MPI_BYTE, mpi_ack_Z[i].rank, mpi_ack_Z[i].tag, comm,
                  &ack_requests_Z[i]);

    // Tell neighbors this rank is ready to receive IPC data, if any
    for (int i = 0; i < n_ack_X; i++)
        MPI_Isend(mpi_ack_X[i].send_buf, mpi_ack_X[i].send_size, MPI_BYTE, mpi_ack_X[i].rank, mpi_ack_X[i].tag, comm,
                  &ack_requests_X[n_ack_X + i]);

    for (int i = 0; i < n_ack_Y; i++)
        MPI_Isend(mpi_ack_Y[i].send_buf, mpi_ack_Y[i].send_size, MPI_BYTE, mpi_ack_Y[i].rank, mpi_ack_Y[i].tag, comm,
                  &ack_requests_Y[n_ack_Y + i]);

    for (int i = 0; i < n_ack_Z; i++)
        MPI_Isend(mpi_ack_Z[i].send_buf, mpi_ack_Z[i].send_size, MPI_BYTE, mpi_ack_Z[i].rank, mpi_ack_Z[i].tag, comm,
                  &ack_requests_Z[n_ack_Z + i]);

    // Launch packing kernels in X
    if (n_msgs_X) {
        // Wait for X neighbors to be ready to receive IPC data before we launch packing/IPC
        if (n_ack_X)
            MPI_Waitall(2 * n_ack_X, ack_requests_X.data(), MPI_STATUSES_IGNORE);
        CUDA_TRY(cudaGraphLaunch(graph_X_packing, stream));
        // Record an event to mark the end of the packing
        CUDA_TRY(cudaEventRecord(end_packing_X, stream));
    }

    // Launch local halo exchanges in X
    if (exchange_X_count)
        CUDA_TRY(cudaGraphLaunch(graph_X_exchange, stream));

    // MPI exchanges in X, then unpack
    if (n_msgs_X) {
        // Only start after the end of the X packing
        CUDA_TRY(cudaEventSynchronize(end_packing_X));

        // MPI_Sends, put the requests after the Irecv requests
        for (int i = 0; i < n_msgs_X; i++)
            MPI_Isend(mpi_msgs_X[i].send_buf, mpi_msgs_X[i].send_size, MPI_BYTE, mpi_msgs_X[i].rank, mpi_msgs_X[i].tag,
                      comm, &requests_X[n_msgs_X + i]);

        // MPI_Waitall, for sends and recvs
        MPI_Waitall(2 * n_msgs_X, requests_X.data(), MPI_STATUSES_IGNORE);

        // Unpack the data in X
        if (launch_x_unpacking)
            CUDA_TRY(cudaGraphLaunch(graph_X_unpacking, stream));
    }
    // Launch packing kernels in Y
    if (n_msgs_Y) {
        // Wait for Y neighbors to be ready to receive IPC data before we launch packing/IPC
        if (n_ack_Y)
            MPI_Waitall(2 * n_ack_Y, ack_requests_Y.data(), MPI_STATUSES_IGNORE);
        CUDA_TRY(cudaGraphLaunch(graph_Y_packing, stream));
        // Record an event to mark the end of the packing
        CUDA_TRY(cudaEventRecord(end_packing_Y, stream));
    }

    // Launch local halo exchanges in Y
    if (exchange_Y_count)
        CUDA_TRY(cudaGraphLaunch(graph_Y_exchange, stream));

    // If the domain decomposition is 2D, stop here for start_update in the middle
    // of the Y exchanges. The rest of the Y MPI will be done in finish_update().
    // BUT...
    // If the MPI domain decomposition is 1D (Z only, no Y MPI exchanges),
    // the Z packing and local exchanges are included in start_update()

    if (n_msgs_Y == 0) {
        if (n_msgs_Z) {
            // Wait for Z neighbors to be ready to receive IPC data before we launch packing/IPC
            if (n_ack_Z)
                MPI_Waitall(2 * n_ack_Z, ack_requests_Z.data(), MPI_STATUSES_IGNORE);
            CUDA_TRY(cudaGraphLaunch(graph_Z_packing, stream));
            // Record an event to mark the end of the packing
            CUDA_TRY(cudaEventRecord(end_packing_Z, stream));
        }
        if (exchange_Z_count)
            CUDA_TRY(cudaGraphLaunch(graph_Z_exchange, stream));
    }
}

// Finish the work started in start_update.
// Must be called with the same stream!
void
haloManager3D_gpu::finish_update(cudaStream_t stream)
{
    int n_msgs_Y = mpi_msgs_Y.size();
    int n_msgs_Z = mpi_msgs_Z.size();

    int n_ack_Z = mpi_ack_Z.size();

    // MPI exchanges in Y, then unpack
    if (n_msgs_Y) {
        // Only start after the end of the Y packing
        CUDA_TRY(cudaEventSynchronize(end_packing_Y));

        // MPI_Sends, put the requests after the Irecv requests
        for (int i = 0; i < n_msgs_Y; i++)
            MPI_Isend(mpi_msgs_Y[i].send_buf, mpi_msgs_Y[i].send_size, MPI_BYTE, mpi_msgs_Y[i].rank, mpi_msgs_Y[i].tag,
                      comm, &requests_Y[n_msgs_Y + i]);

        // MPI_Waitall
        MPI_Waitall(2 * n_msgs_Y, requests_Y.data(), MPI_STATUSES_IGNORE);

        // Unpack the data in Y
        if (launch_y_unpacking)
            CUDA_TRY(cudaGraphLaunch(graph_Y_unpacking, stream));
    }

    // Launch packing kernels in Z
    if (n_msgs_Y && n_msgs_Z) {
        // Wait for Z neighbors to be ready to receive IPC data before we launch packing/IPC
        if (n_ack_Z)
            MPI_Waitall(2 * n_ack_Z, ack_requests_Z.data(), MPI_STATUSES_IGNORE);

        CUDA_TRY(cudaGraphLaunch(graph_Z_packing, stream));
        // Record an event to mark the end of the packing
        CUDA_TRY(cudaEventRecord(end_packing_Z, stream));
    }

    // Launch local halo exchanges in Z, unless the decomposition is 1D,
    // in which case this was already executed in start_update()
    if (n_msgs_Y && exchange_Z_count)
        CUDA_TRY(cudaGraphLaunch(graph_Z_exchange, stream));

    // MPI exchanges in Z
    if (n_msgs_Z) {
        // Only start after the end of the Z packing
        CUDA_TRY(cudaEventSynchronize(end_packing_Z));

        // MPI_Sends, put the requests after the Irecv requests
        for (int i = 0; i < n_msgs_Z; i++)
            MPI_Isend(mpi_msgs_Z[i].send_buf, mpi_msgs_Z[i].send_size, MPI_BYTE, mpi_msgs_Z[i].rank, mpi_msgs_Z[i].tag,
                      comm, &requests_Z[n_msgs_Z + i]);

        // MPI_Waitall
        MPI_Waitall(2 * n_msgs_Z, requests_Z.data(), MPI_STATUSES_IGNORE);

        // Unpack the data in Z
        if (launch_z_unpacking)
            CUDA_TRY(cudaGraphLaunch(graph_Z_unpacking, stream));
    }

    // Wait for Z exchange to complete, so the finish_update is blocking, and will not return until all exchanges are complete.
    CUDA_TRY(cudaStreamSynchronize(stream));
}

bool
haloManager3D_gpu::consistent_halos(decompositionManager3D* decomp, std::vector<std::vector<int>> wSz[3][2])
{
    // For each subdomain and each field, we need to make sure that the halo exchanges are consistent,
    // in order to exchange corners and edges correctly.
    // For a given dimension, the neighbors must have the same exchange sizes for the other 2 dimensions.
    // E.g. if a subdomain has a neighbor subdomain in X, both subdomains must have the same
    // exchange dimensions for Y and Z.

    int nsubdom = decomp->getTotalNumLocalSubDom();
    int nfields = wSz[0][0].size();

    for (int j = 0; j < nsubdom; ++j) {
        int nhSubDomId[_NH_ID_3D_TOTAL];
        decomp->getLocalNeighborID(j, nhSubDomId);

        // Check X neighbors
        std::vector<int> x_neighbors;
        int xm = nhSubDomId[_NH_ID_3D_NEGX_CENY_CENZ];
        int xp = nhSubDomId[_NH_ID_3D_POSX_CENY_CENZ];
        if (xm >= 0)
            x_neighbors.push_back(xm);
        if (xp >= 0)
            x_neighbors.push_back(xp);
        for (auto xn : x_neighbors) {
            for (int i = 0; i < nfields; ++i) {
                if (wSz[1][0][i][j] != wSz[1][0][i][xn] || // Y- halo
                    wSz[1][1][i][j] != wSz[1][1][i][xn] || // Y+ halo
                    wSz[2][0][i][j] != wSz[2][0][i][xn] || // Z- halo
                    wSz[2][1][i][j] != wSz[2][1][i][xn])   // Z+ halo
                    return false;
            }
        }

        // Check Y neighbors
        std::vector<int> y_neighbors;
        int ym = nhSubDomId[_NH_ID_3D_CENX_NEGY_CENZ];
        int yp = nhSubDomId[_NH_ID_3D_CENX_POSY_CENZ];
        if (ym >= 0)
            y_neighbors.push_back(ym);
        if (yp >= 0)
            y_neighbors.push_back(yp);
        for (auto yn : y_neighbors) {
            for (int i = 0; i < nfields; ++i) {
                if (wSz[0][0][i][j] != wSz[0][0][i][yn] || // X- halo
                    wSz[0][1][i][j] != wSz[0][1][i][yn] || // X+ halo
                    wSz[2][0][i][j] != wSz[2][0][i][yn] || // Z- halo
                    wSz[2][1][i][j] != wSz[2][1][i][yn])   // Z+ halo
                    return false;
            }
        }
        // Check Z neighbors
        std::vector<int> z_neighbors;
        int zm = nhSubDomId[_NH_ID_3D_CENX_CENY_NEGZ];
        int zp = nhSubDomId[_NH_ID_3D_CENX_CENY_POSZ];
        if (zm >= 0)
            z_neighbors.push_back(zm);
        if (zp >= 0)
            z_neighbors.push_back(zp);
        for (auto zn : z_neighbors) {
            for (int i = 0; i < nfields; ++i) {
                if (wSz[0][0][i][j] != wSz[0][0][i][zn] || // X- halo
                    wSz[0][1][i][j] != wSz[0][1][i][zn] || // X+ halo
                    wSz[1][0][i][j] != wSz[1][0][i][zn] || // Y- halo
                    wSz[1][1][i][j] != wSz[1][1][i][zn])   // Y+ halo
                    return false;
            }
        }
    }
    return true;
}

void
haloManager3D_gpu::print(FILE* fd)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    int npes;
    MPI_Comm_size(comm, &npes);

    if (rank == 0) {
        fprintf(fd, "---- begin 3D halo manager info ----\n");
    }

    char token = 1;
    if (rank > 0) {
        MPI_Recv(&token, 1, MPI_CHAR, (rank - 1), 1, comm, MPI_STATUS_IGNORE);
    }

    fprintf(fd, "halo X(%d): Reqsz=%d\n", rank, (int)(requests_X.size()));
    fprintf(fd, "halo Y(%d): Reqsz=%d\n", rank, (int)(requests_Y.size()));
    fprintf(fd, "halo Z(%d): Reqsz=%d\n", rank, (int)(requests_Z.size()));
    fflush(fd);

    if (rank < (npes - 1)) {
        MPI_Send(&token, 1, MPI_CHAR, (rank + 1), 1, comm);
    }

    if (rank == (npes - 1)) {
        fprintf(fd, "---- end 3D halo manager info ----\n");
    }
} //end print

// Get IPC info for the received halo of the buffer for a particular dimension
template <int dim, bool lo>
void
haloManager3D_gpu::get_cuda_ipc_info(cuda_ipc_info_t* info, cart_volume<T>* vol, int radius)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();

    // Leading dimensions
    info->ldimx = vol_gpu->ax1()->ntot;
    info->ldimy = vol_gpu->ax2()->ntot;

    // Compute offset of the first point to be received
    if (dim == 0) { // X dim, exclude Y and Z halos
        info->offset = (vol_gpu->ax3()->ibeg * info->ldimy + vol_gpu->ax2()->ibeg) * info->ldimx +
                       (lo ? (vol_gpu->ax1()->ibeg - radius) : (vol_gpu->ax1()->iend + 1));
    } else if (dim == 1) { // Y dim, include X halos but exclude Z halos
        info->offset = vol_gpu->ax3()->ibeg * info->ldimy * info->ldimx +
                       info->ldimx * (lo ? (vol_gpu->ax2()->ibeg - radius) : (vol_gpu->ax2()->iend + 1));
    } else if (dim == 2) { // Z dim, include X and Y halos
        info->offset = info->ldimx * info->ldimy * (lo ? (vol_gpu->ax3()->ibeg - radius) : (vol_gpu->ax3()->iend + 1));
    }

    // Get IPC handle for this pointer. Must be the base pointer, no offset.
    CUDA_TRY(cudaIpcGetMemHandle(&info->handle, vol_gpu->getData()));
} // end get_cuda_ipc_info
