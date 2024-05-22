#ifndef _HALO_MANAGER_3D_GPU
#define _HALO_MANAGER_3D_GPU

#include <mpi.h>
#include "decompositionManager3D.h"
#include "cart_volume.h"
#include "cart_volume_regular_gpu.h"
#include "haloManager3D.h"
#include "haloManager3D_gpu.h"
#include "cuda_utils.h"
#include <type_traits>

// Once there's a common base class we'll use it.
// Maybe templated for the type of the cart_volume<T>
class haloManager3D_gpu : public haloManager3D
{
    using T = realtype;

public:
    /**
      Constructor
decomp: domain decomposition object.
data: list of fields that participate in the halo-exchange.
wSz: Widths of the stencil for each sub-volume of each field in each direction. This is 
typically equal to the number of halo values received, except when there is no adjacent 
sub-volume in that direction. The first dimension is one of X = 0, Y = 1 or Z = 2. The second 
dimension is one of Negative = 0 or Positive = 1.
includeEdges: True to include edges and false otherwise.
includeCorners: True to include corners and false otherwise. Ignored if includeEdges is false.
*/
    haloManager3D_gpu(decompositionManager3D* decomp, std::vector<cart_volume<T>**>& data,
                      std::vector<std::vector<int>> wSz[3][2], std::vector<std::vector<bool>>& includeEdges,
                      std::vector<std::vector<bool>>& includeCorners);

    // We do not want copy operations because of the duplicated comm
    haloManager3D_gpu(const haloManager3D_gpu& other) = delete;
    haloManager3D_gpu& operator=(const haloManager3D_gpu& other) = delete;

    //Destructor
    ~haloManager3D_gpu();

    // default methods are using the default stream 0
    void start_update() override { start_update(0); }
    void finish_update() override { finish_update(0); }

    // GPU-specific methods can use any stream
    void start_update(cudaStream_t stream);
    void finish_update(cudaStream_t stream);

    void print(FILE* fd) override;

private:
    MPI_Comm comm;

    // CUDA graphs to execute echange kernels betwen local subdomains
    cudaGraphExec_t graph_X_exchange;
    cudaGraphExec_t graph_Y_exchange;
    cudaGraphExec_t graph_Z_exchange;
    int exchange_X_count = 0;
    int exchange_Y_count = 0;
    int exchange_Z_count = 0;

    // Structure to hold MPI message parameters
    struct mpi_msg
    {
        mpi_msg(int _rank, int _send_size, int _recv_size, T* _send_buf, T* _recv_buf, int _tag)
            : rank(_rank), send_size(_send_size), recv_size(_recv_size), send_buf(_send_buf), recv_buf(_recv_buf),
              tag(_tag)
        {}
        int rank;      // the other rank
        int send_size; // In bytes, 32-bit int as MPI message sizes are 32-bit!
        int recv_size; // In bytes, 32-bit int as MPI message sizes are 32-bit!
        T* send_buf;
        T* recv_buf;
        int tag;
    };
    std::vector<mpi_msg> mpi_msgs_X;
    std::vector<mpi_msg> mpi_msgs_Y;
    std::vector<mpi_msg> mpi_msgs_Z;

    std::vector<mpi_msg> mpi_ack_X;
    std::vector<mpi_msg> mpi_ack_Y;
    std::vector<mpi_msg> mpi_ack_Z;

    // Communication buffers can be either device or host memory
    static void release_commbuf(T* ptr)
    {
        if (is_device_pointer(ptr)) {
            CUDA_TRY(cudaFree(ptr));
        } else
            free(ptr);
    }

    // Vectors of requests
    std::vector<MPI_Request> requests_X;
    std::vector<MPI_Request> requests_Y;
    std::vector<MPI_Request> requests_Z;

    // Vectors of ack requests
    std::vector<MPI_Request> ack_requests_X;
    std::vector<MPI_Request> ack_requests_Y;
    std::vector<MPI_Request> ack_requests_Z;

    // Events to mark the end of the packing kernels
    cudaEvent_t end_packing_X;
    cudaEvent_t end_packing_Y;
    cudaEvent_t end_packing_Z;

    // CUDA graphs to execute packing and unpacking kernels
    cudaGraphExec_t graph_X_packing;
    cudaGraphExec_t graph_Y_packing;
    cudaGraphExec_t graph_Z_packing;
    cudaGraphExec_t graph_X_unpacking;
    cudaGraphExec_t graph_Y_unpacking;
    cudaGraphExec_t graph_Z_unpacking;

    // Unpacking graphs might be empty if same host
    bool launch_x_unpacking;
    bool launch_y_unpacking;
    bool launch_z_unpacking;

    // Functions to discover the communication patterns and create CUDA graphs
    template <int dim>
    static void create_graph_local_exchanges(int& exchange_count, cudaGraphExec_t& graphexec,
                                             decompositionManager3D* decomp, std::vector<cart_volume<realtype>**>& data,
                                             std::vector<std::vector<int>> wSz[3][2]);

    template <int dim>
    static void create_graph_mpi_exchanges(std::vector<mpi_msg>& mpi_msg_vec, std::vector<mpi_msg>& mpi_ack_vec,
                                           cudaGraphExec_t& graph_pack, cudaGraphExec_t& graph_unpack,
                                           std::vector<T*>& ipc_vec_lo, std::vector<T*>& ipc_vec_hi,
                                           bool& launch_unpacking, decompositionManager3D* decomp,
                                           std::vector<cart_volume<realtype>**>& data,
                                           std::vector<std::vector<int>> wSz[3][2]);

    // Check if the halo exchanges are compatible with the optimized exchange scheme
    static bool consistent_halos(decompositionManager3D* decomp, std::vector<std::vector<int>> wSz[3][2]);

    // 2-way copy of halos between local subdomains (kernels)
    static void exchange_halos_X(cart_volume<T>* lo_vol, cart_volume<T>* hi_vol, int radius_lo, int radius_hi,
                                 cudaStream_t stream);
    static void exchange_halos_Y(cart_volume<T>* lo_vol, cart_volume<T>* hi_vol, int radius_lo, int radius_hi,
                                 cudaStream_t stream);
    static void exchange_halos_Z(cart_volume<T>* lo_vol, cart_volume<T>* hi_vol, int radius_lo, int radius_hi,
                                 cudaStream_t stream);

    // Packing and unpacking kernels (working with an intermediate communication buffer)
    template <bool lo, bool pack>
    static void copy_halos_X(cart_volume<T>* vol, T* combuf, int radius, cudaStream_t stream);
    template <bool lo, bool pack>
    static void copy_halos_Y(cart_volume<T>* vol, T* combuf, int radius, cudaStream_t stream);
    template <bool lo, bool pack>
    static void copy_halos_Z(cart_volume<T>* vol, T* combuf, int radius, cudaStream_t stream);

    // Copy halos straight to a remote buffer
    template <bool lo>
    static void copy_halos_X(cart_volume<T>* vol, T* dest, int dest_ldimx, int dest_ldimy, int radius,
                             cudaStream_t stream);
    template <bool lo>
    static void copy_halos_Y(cart_volume<T>* vol, T* dest, int dest_ldimx, int dest_ldimy, int radius,
                             cudaStream_t stream);
    template <bool lo>
    static void copy_halos_Z(cart_volume<T>* vol, T* dest, int dest_ldimx, int dest_ldimy, int radius,
                             cudaStream_t stream);

    // Size of the X communication buffer, in elements
    static int get_size_combuf_X(cart_volume<T>* vol, int radius);

    // Size of the Y communication buffer, in elements
    static int get_size_combuf_Y(cart_volume<T>* vol, int radius);

    // Size of the Z communication buffer, in elements
    static int get_size_combuf_Z(cart_volume<T>* vol, int radius);

    // Structure for CUDA IPC buffer info
    struct cuda_ipc_info_t
    {
        cudaIpcMemHandle_t handle; // handle from pointer
        int offset;                // Offset where to start the copy
        int ldimx;                 // leading dimension X
        int ldimy;                 // leading dimension Y
    };

    // IPC info for X Y or Z buffers
    template <int dim, bool lo>
    static void get_cuda_ipc_info(cuda_ipc_info_t* info, cart_volume<T>* vol, int radius);

    // Remote buffer pointers mapped from IPC handles
    std::vector<T*> dest_ptr_xlo;
    std::vector<T*> dest_ptr_xhi;
    std::vector<T*> dest_ptr_ylo;
    std::vector<T*> dest_ptr_yhi;
    std::vector<T*> dest_ptr_zlo;
    std::vector<T*> dest_ptr_zhi;
};
#endif // _HALO_MANAGER_3D_GPU
