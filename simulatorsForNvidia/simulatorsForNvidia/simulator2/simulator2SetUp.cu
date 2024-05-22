#include <math.h>
#include "simulator2SetUp.h"
#include "cart_volume_regular_gpu.h"
#include "kernel_type1_gpu.h"
#include "kernel_base.h"
#include "kernel_type2.h"
#include "file_snap_factory.h"
#include "timer.h"

__global__ void
init_cijs_kernel(float vp, float eps, float delta, float vs, float theta, float phi, float* model1, float* model2, float* model3,
                 float* model4, float* model5, float* model6, float* model7, int ixbeg, int ixend, int iybeg, int iyend, int izbeg,
                 int izend, volume_index idx)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    /// becasue use 128x1x1 no need check y and z
    if (ix >= ixbeg && ix <= ixend) { //&& iy >= iybeg && iy <= iyend && iz >= izbeg && iz <= izend) {
        idx(model1, ix, iy, iz) = vp * vp * (1 + 2 * eps);
        idx(model2, ix, iy, iz) = sqrtf((vp * vp - vs * vs) * (vp * vp * (1 + 2 * delta) - vs * vs));
        idx(model3, ix, iy, iz) = vp * vp;
        idx(model4, ix, iy, iz) = vs * vs;
        idx(model5, ix, iy, iz) = sin(theta) * cos(phi);
        idx(model6, ix, iy, iz) = sin(theta) * sin(phi);
        idx(model7, ix, iy, iz) = cos(theta);
    }
}

__global__ void
compute_pressure_kernel(float* pressure, float* field1, float* field2, int xtot, int ytot, int ztot, volume_index idx)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    /// becasue use 128x1x1 no need check y and z
    if (ix < xtot) { // && iy < ytot && iz < ztot) {
        idx(pressure, ix, iy, iz) = -(2 * idx(field1, ix, iy, iz) + idx(field2, ix, iy, iz)) / 3;
    }
}

__global__ void
init_src_kernel(float* field1, float* field2, int x, int y, int z, float press, volume_index idx)
{
    idx(field1, x, y, z) = press;
    idx(field2, x, y, z) = press;
}

simulator2SetUp::simulator2SetUp(int npoints, int npml, int niter, int xcorr_step, int nzones[3], int ranks[3], int radius_lo, int radius_hi, float bitPerFloat, int fwd_only, MPI_Comm communicator):
    _npoints {npoints},
    _npml {npml},
    _niter {niter},
    scratch_folder {"test_snapshot_area"},
    _xcorr_step {xcorr_step},
    _bitPerFloat {bitPerFloat},
    _fwd_only {fwd_only}
{
    EMSL_VERIFY(nzones[0] >= 1 && nzones[1] >= 1 && nzones[2] >= 1);
    EMSL_VERIFY(ranks[0] >= 1 && ranks[1] >= 1 && ranks[2] >= 1);

    _comm = communicator;
    MPI_Comm_rank(_comm, &_rank);
    r_lo = radius_lo;
    r_hi = radius_hi;
    radius = std::max(radius_lo, radius_hi);

    // Define the number of points in each zone
    for (int i = 0; i < 3; i++) {
        int sum = 0;
        numPts[i] = new int[nzones[i]];
        for (int j = 0; j < nzones[i]; j++) {
            numPts[i][j] = get_n_for_zone(j, nzones[i]);
            sum += numPts[i][j];
        }
        n_total[i] = sum;
    }

    numZonesinX = nzones[0];
    numZonesinY = nzones[1];
    numZonesinZ = nzones[2];

    // Costs are all the same = 1
    costs = new float**[numZonesinZ];
    for (int i = 0; i < numZonesinZ; ++i) {
        costs[i] = new float*[numZonesinY];
        for (int j = 0; j < numZonesinY; ++j) {
            costs[i][j] = new float[numZonesinX];
            for (int k = 0; k < numZonesinX; ++k)
                costs[i][j][k] = 1.0f;
        }
    }

    // Same stencil width for all 3D but allow staggered
    for (int axis = 0; axis < 3; ++axis) {
        for (int sign = 0; sign < 2; ++sign) {
            stencilWidths[axis][sign] = new int**[numZonesinZ];
            int r = sign ? radius_lo : radius_hi;
            for (int z = 0; z < numZonesinZ; ++z) {
                stencilWidths[axis][sign][z] = new int*[numZonesinY];
                for (int y = 0; y < numZonesinY; ++y) {
                    stencilWidths[axis][sign][z][y] = new int[numZonesinX];
                    for (int x = 0; x < numZonesinX; ++x)
                        stencilWidths[axis][sign][z][y][x] = r;
                }
            }
        }
    }
    // Create a decomposition manager
    decompmgr = new decompositionManager3D(nzones, numPts, costs, stencilWidths, ranks, communicator, 0);

    float dx = 1.0f;
    float dy = 1.0f;
    float dz = 1.0f;
    // no halo
    axgiX = new axis(0, dx, _npoints);
    axgiY = new axis(0, dy, _npoints);
    axgiZ = new axis(0, dz, _npoints);
    //Global axis including boundaries
    axgX = new axis(axgiX->o - _npml * axgiX->d, axgiX->e + _npml * axgiX->d, axgiX->d);
    axgY = new axis(axgiY->o - _npml * axgiY->d, axgiY->e + _npml * axgiY->d, axgiY->d);
    axgZ = new axis(axgiZ->o - _npml * axgiZ->d, axgiZ->e + _npml * axgiZ->d, axgiZ->d);

    // init Wsi and dt
    allocateAndInitializeWsi();
    float dt_l = 1. / (1. * vp * sqrtf(1. / (dx * dx) + 1. / (dy * dy) + 1. / (dz * dz)));
    dt_l *= 0.77;     // get_dt_limit_order_factor(order);
    dt = 0.91 * dt_l; // cfl_perc * dt_limit;

    // Get the  number of subdomains for each dimension
    int nsubs_per_dim[3];
    decompmgr->getSplitNumLocalSubDom(nsubs_per_dim);

    // For each subdomain of each dimension, create an axis
    std::vector<axis*> local_X_axes(nsubs_per_dim[0]);
    std::vector<axis*> local_X_axes_noHalo(nsubs_per_dim[0]);
    std::vector<axis*> local_Y_axes(nsubs_per_dim[1]);
    std::vector<axis*> local_Y_axes_noHalo(nsubs_per_dim[1]);
    std::vector<axis*> local_Z_axes(nsubs_per_dim[2]);
    std::vector<axis*> local_Z_axes_noHalo(nsubs_per_dim[2]);

    for (int isub = 0; isub < nsubs_per_dim[0]; ++isub) // num sub domains in X axis
    {
        int offset = decompmgr->getOffset(isub, 0); // 0 -- X axis
        float origin = axgX->o + (float(offset) * axgX->d);
        int nxloc = decompmgr->getNumPtsSplit(isub, 0);
        local_X_axes[isub] =
            new axis(origin, dx, nxloc, radius, AlignmentElem(AlignMemBytes::CACHELINE, sizeof(float)));
        //for each subdomain of cpu axis without halo.
        local_X_axes_noHalo[isub] = new axis(origin, dx, nxloc);
    }

    for (int isub = 0; isub < nsubs_per_dim[1]; ++isub) // num sub domains in Y axis
    {
        int offset = decompmgr->getOffset(isub, 1); // 1 -- Y axis
        float origin = axgY->o + (float(offset) * axgY->d);
        int nyloc = decompmgr->getNumPtsSplit(isub, 1);
        local_Y_axes[isub] = new axis(origin, dy, nyloc, radius);
        local_Y_axes_noHalo[isub] = new axis(origin, dy, nyloc);
    }

    for (int isub = 0; isub < nsubs_per_dim[2]; ++isub) // num sub domains in Z axis
    {
        int offset = decompmgr->getOffset(isub, 2); // 2 -- Z axis
        float origin = axgZ->o + (float(offset) * axgZ->d);
        int nzloc = decompmgr->getNumPtsSplit(isub, 2);
        local_Z_axes[isub] = new axis(origin, dz, nzloc, radius);
        local_Z_axes_noHalo[isub] = new axis(origin, dz, nzloc);
    }

    // Total number of subvolumes
    nsubs = nsubs_per_dim[0] * nsubs_per_dim[1] * nsubs_per_dim[2];
    //from kernel test
    for (int n = 0; n < NMECH; n++) {
        Model8[n] = new cart_volume<float>*[nsubs];
        Model9[n] = new cart_volume<float>*[nsubs];
        Model81[n] = new cart_volume<float>*[nsubs];
        Model91[n] = new cart_volume<float>*[nsubs];
        Model10[n] = new cart_volume<float>*[nsubs];
    }
    field1 = new cart_volume<float>*[nsubs];
    field2 = new cart_volume<float>*[nsubs];
    field3 = new cart_volume<float>*[nsubs];
    field4 = new cart_volume<float>*[nsubs];
    field11 = new cart_volume<float>*[nsubs];
    field21 = new cart_volume<float>*[nsubs];
    field31 = new cart_volume<float>*[nsubs];
    field41 = new cart_volume<float>*[nsubs];
    field3_rhs = new cart_volume<float>*[nsubs];
    field4_rhs = new cart_volume<float>*[nsubs];
    Model11 = new cart_volume<float>*[nsubs];
    model1 = new cart_volume<float>*[nsubs];
    model2 = new cart_volume<float>*[nsubs];
    model3 = new cart_volume<float>*[nsubs];
    model4 = new cart_volume<float>*[nsubs];
    dfield1dx = new cart_volume<float>*[nsubs];
    dfield1dy = new cart_volume<float>*[nsubs];
    dfield1dz = new cart_volume<float>*[nsubs];
    dfield2dx = new cart_volume<float>*[nsubs];
    dfield2dy = new cart_volume<float>*[nsubs];
    dfield2dz = new cart_volume<float>*[nsubs];
    model5 = new cart_volume<float>*[nsubs];
    model6 = new cart_volume<float>*[nsubs];
    model7 = new cart_volume<float>*[nsubs];
    field5_gpu = new cart_volume<float>*[nsubs];
    grad1 = new cart_volume<float>*[nsubs];
    grad2 = new cart_volume<float>*[nsubs];

    // We're not using PML for this at all, so set all the pointers to null
    std::memset(&pml, 0, sizeof(pml));

    snap_field3.resize(nsubs, nullptr);
    snap_field4.resize(nsubs, nullptr);
    snap_field1.resize(nsubs, nullptr);
    snap_field2.resize(nsubs, nullptr);

    //create and initialize cart vols
    for (int isub = 0; isub < nsubs; isub++) {
        int subid[3];
        decompmgr->getSplitLocalSubDomID(isub, subid);
        axis* ax1 = local_X_axes[subid[0]];
        axis* ax2 = local_Y_axes[subid[1]];
        axis* ax3 = local_Z_axes[subid[2]];

        // creat cart_vols and set 0
        field1[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field2[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field3[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field4[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field11[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field21[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field31[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field41[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field3_rhs[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field4_rhs[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        Model11[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        model1[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        model2[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        model3[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        model4[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        dfield1dx[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        dfield1dy[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        dfield1dz[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        dfield2dx[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        dfield2dy[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        dfield2dz[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        model5[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        model6[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        model7[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        field5_gpu[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);

        for (int n = 0; n < NMECH; n++) {
            Model81[n][isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
            Model91[n][isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
            Model8[n][isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
            Model8[n][isub]->set_constant(1.0f, false, false);
            Model9[n][isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
            Model9[n][isub]->set_constant(1.0f, false, false);
            Model10[n][isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
            Model10[n][isub]->set_constant(1.0f, false, false);
        }

        init_cijs(isub);

        // Initialize correlation buffer with snapshots
        axis* ax1_noHalo = local_X_axes_noHalo[subid[0]];
        axis* ax2_noHalo = local_Y_axes_noHalo[subid[1]];
        axis* ax3_noHalo = local_Z_axes_noHalo[subid[2]];

        snap_field3[isub] =  new cart_volume_regular_gpu(ax1_noHalo, ax2_noHalo, ax3_noHalo, true);
        snap_field4[isub] =  new cart_volume_regular_gpu(ax1_noHalo, ax2_noHalo, ax3_noHalo, true);
        snap_field1[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        snap_field2[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        grad1[isub] =  new cart_volume_regular_gpu(ax1_noHalo, ax2_noHalo, ax3_noHalo, true);
        grad2[isub] =  new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        corrBuffList.push_back(snap_field3[isub]);
        corrBuffList.push_back(snap_field4[isub]);
        corrBuffList.push_back(snap_field1[isub]);
        corrBuffList.push_back(snap_field2[isub]);
    }
    int nbuf = corrBuffList.size();
    corrBuff_size.resize(nbuf, 0);

    if (_bitPerFloat > 0 && _bitPerFloat < 32) {
        useZFP_ = true;

        zipped_corr_buff_.resize(nbuf, NULL);
        zfpFields_.resize(nbuf, NULL);

        zfpStream_ = zfp_stream_open(NULL);
        zfp_stream_set_rate(zfpStream_, bitPerFloat, zfp_type_float, 3, 0);
    }
    else 
        useZFP_ = false;

    for (int i = 0; i < nbuf; ++i) {
        cart_volume<realtype>* vol = corrBuffList[i];
        // Use nvalid here, so that the snapshot computational volumes can have alignment padding if desired.
        // The snapshot storage will strip any padding.
        int n1 = vol->as<cart_volume_regular>()->ax1()->nvalid;
        int n2 = vol->as<cart_volume_regular>()->ax2()->nvalid;
        int n3 = vol->as<cart_volume_regular>()->ax3()->nvalid;

        if (!useZFP_) {
            corrBuff_size[i] = n1 * n2 * n3;
        } else {
            zfpFields_[i] = zfp_field_3d(NULL, zfp_type_float, n1, n2, n3);

            zfp_field_set_stride_3d(zfpFields_[i], 1, vol->as<cart_volume_regular>()->ax1()->ntot,
                                    vol->as<cart_volume_regular>()->ax1()->ntot *
                                        vol->as<cart_volume_regular>()->ax2()->ntot);

            long nbyte = zfp_stream_maximum_size(zfpStream_, zfpFields_[i]);

            long nfloat = nbyte / sizeof(realtype);
            if (nfloat * sizeof(realtype) < nbyte)
                nfloat += 1;

            corrBuff_size[i] = nfloat;
            zfp_stream_set_execution(zfpStream_, zfp_exec_cuda);
            CUDA_TRY(cudaMallocHost((void**)&zipped_corr_buff_[i], nfloat * sizeof(realtype)));
        }
    }

    // allocateAndInitializeSpongeCoefs();
    // Need to create these vectors for the sponge_coef.
    // The first vector needs to be the size of the number of subdomains used,
    // so the kernel can iterate through it.  The kernel will begin iteration after the
    // number of ghost cells (nghost) used to initialize the axis used in the iteration.
    // The iteration ends after it reaches the end of the axis, which is nghost + n - 1.
    // So the inner vectors need to have size nghost + n.
    // sponge_coef_cpu.z_B_ = std::vector<std::vector<float>>(nsubs, std::vector<float>(axgiz->n + axgiz->nghost, 1.0f));
    sponge_coef_cpu.x_A_ = std::vector<std::vector<float>>(nsubs);
    sponge_coef_cpu.x_B_ = std::vector<std::vector<float>>(nsubs);
    sponge_coef_cpu.y_A_ = std::vector<std::vector<float>>(nsubs);
    sponge_coef_cpu.y_B_ = std::vector<std::vector<float>>(nsubs);
    sponge_coef_cpu.z_A_ = std::vector<std::vector<float>>(nsubs);
    sponge_coef_cpu.z_B_ = std::vector<std::vector<float>>(nsubs);

    for (int isub = 0; isub < nsubs; isub++) {
        int subid[3];
        decompmgr->getSplitLocalSubDomID(isub, subid);
        axis* ax1 = local_X_axes[subid[0]];
        axis* ax2 = local_Y_axes[subid[1]];
        axis* ax3 = local_Z_axes[subid[2]];

        sponge_coef_cpu.x_A_[isub] = std::vector<float>(ax1->n + ax1->nghost, 1.0f);
        sponge_coef_cpu.x_B_[isub] = std::vector<float>(ax1->n + ax1->nghost, 1.0f);
        sponge_coef_cpu.y_A_[isub] = std::vector<float>(ax2->n + ax2->nghost, 1.0f);
        sponge_coef_cpu.y_B_[isub] = std::vector<float>(ax2->n + ax2->nghost, 1.0f);
        sponge_coef_cpu.z_A_[isub] = std::vector<float>(ax3->n + ax3->nghost, 1.0f);
        sponge_coef_cpu.z_B_[isub] = std::vector<float>(ax3->n + ax3->nghost, 1.0f);
    }

    sponge_coef_gpu.setCoeffs(sponge_coef_cpu);

    CUDA_CHECK_ERROR(__FILE__, __LINE__);

    // Create cart_volumes for all s_fields_for_halo, all subdomains.
    s_fields_for_halo.push_back(field1);
    s_fields_for_halo.push_back(field2);
    int nf_s = s_fields_for_halo.size();

    sd_fields_for_halo.push_back(dfield1dx);
    sd_fields_for_halo.push_back(dfield1dy);
    sd_fields_for_halo.push_back(dfield1dz);
    sd_fields_for_halo.push_back(dfield2dz);
    sd_fields_for_halo.push_back(dfield2dy);
    sd_fields_for_halo.push_back(dfield2dz);
    int nf_sd = sd_fields_for_halo.size();

    // Sub-domain Stencil widths
    for (int axis = 0; axis < 3; ++axis) // 0 -- x; 1 -- y, 2 -- z
    {
        for (int sign = 0; sign < 2; ++sign) // 0 -- neg; 1 -- pos
        {
            subDomStencilWidths_s[axis][sign] = std::vector<std::vector<int>>(nf_s, std::vector<int>(nsubs, 0));
            subDomStencilWidths_sd[axis][sign] = std::vector<std::vector<int>>(nf_sd, std::vector<int>(nsubs, 0));
            for (int isub = 0; isub < nsubs; ++isub) {
                subDomStencilWidths_s[axis][sign][0][isub] = radius;
                subDomStencilWidths_s[axis][sign][1][isub] = radius;
                subDomStencilWidths_sd[axis][sign][0][isub] = radius;
                subDomStencilWidths_sd[axis][sign][1][isub] = radius;
                subDomStencilWidths_sd[axis][sign][2][isub] = radius;
                subDomStencilWidths_sd[axis][sign][3][isub] = radius;
                subDomStencilWidths_sd[axis][sign][4][isub] = radius;
                subDomStencilWidths_sd[axis][sign][5][isub] = radius;
            } // isub
        }
    }

    // Create the GPU halo manager
    std::vector<std::vector<bool>> incEdges_s(s_fields_for_halo.size(), std::vector<bool>(nsubs, true));
    std::vector<std::vector<bool>> incCorners_s(s_fields_for_halo.size(), std::vector<bool>(nsubs, true));
    halomgr_s = new haloManager3D_gpu(decompmgr, s_fields_for_halo, subDomStencilWidths_s, incEdges_s, incCorners_s);

    std::vector<std::vector<bool>> incEdges_sd(sd_fields_for_halo.size(), std::vector<bool>(nsubs, true));
    std::vector<std::vector<bool>> incCorners_sd(sd_fields_for_halo.size(), std::vector<bool>(nsubs, true));
    halomgr_sd = new haloManager3D_gpu(decompmgr, sd_fields_for_halo, subDomStencilWidths_sd, incEdges_sd, incCorners_sd);
    
    CUDA_CHECK_ERROR(__FILE__, __LINE__);

    initKernels();

    CUDA_CHECK_ERROR(__FILE__, __LINE__);

    long nfloats;
    for (int i = 0; i < corrBuff_size.size(); ++i) {
        nfloats += corrBuff_size[i];
    }
    file_snap_p = file_snap_factory::create(_comm, const_cast<char*>(scratch_folder.c_str()),
                                            nfloats, memoryPercent, _niter/xcorr_step, snap_type);
    CUDA_TRY(cudaEventCreate(&writeSnapshotsCompleteEvent_));
    CUDA_TRY(cudaEventCreate(&readSnapshotsCompleteEvent_));
    snapReaderWriter = std::make_unique<rtm_snap_gpu>(corrBuffList, writeSnapshotsCompleteEvent_, readSnapshotsCompleteEvent_, useZFP_);  
}

simulator2SetUp::~simulator2SetUp()
{
    for (int i = 0; i < 3; ++i) {
        delete[] numPts[i];
    }

    if (costs != nullptr) {
        for (int i = 0; i < numZonesinZ; ++i) {
            for (int j = 0; j < numZonesinY; ++j) {
                delete[] costs[i][j];
            }
            delete[] costs[i];
        }
        delete[] costs;
    }

    for (int axis = 0; axis < 3; ++axis) {
        for (int sign = 0; sign < 2; ++sign) {
            if (stencilWidths[axis][sign] != nullptr) {
                for (int z = 0; z < numZonesinZ; ++z) {
                    for (int y = 0; y < numZonesinY; ++y) {
                        delete[] stencilWidths[axis][sign][z][y];
                    }
                    delete[] stencilWidths[axis][sign][z];
                }
                delete[] stencilWidths[axis][sign];
            }
        }
    }

    delete axgiX;
    delete axgiY;
    delete axgiZ;
    delete axgX;
    delete axgY;
    delete axgZ;

    delete decompmgr;
    delete halomgr_s;
    delete halomgr_sd;
}


int simulator2SetUp::get_n_for_zone(int izone, int nzones)
{
    if (nzones == 1)
        return _npoints; // 1 zone -> [_npoints]
    else {
        if (nzones == 2) // 2 zones -> [_npoints PML]
            return izone ? _npoints : _npml;
        else // 3+ zones -> [PML _npoints [_npoints...] PML]
            return (izone == 0 || izone == nzones - 1) ? _npml : _npoints;
    }
    EMSL_VERIFY(false); // shouln't be here
    return 0;
}

void simulator2SetUp::init_cijs(int isub)
{
    cart_volume_regular_gpu* model1_gpu = model1[isub]->as<cart_volume_regular_gpu>();

    int ixbeg = model1_gpu->ax1()->ibeg, ixend = model1_gpu->ax1()->iend;
    int iybeg = model1_gpu->ax2()->ibeg, iyend = model1_gpu->ax2()->iend;
    int izbeg = model1_gpu->ax3()->ibeg, izend = model1_gpu->ax3()->iend;

    int xsize = model1_gpu->ax1()->ntot - model1_gpu->ax1()->npad_trailing;

    float* model1data = model1[isub]->as<cart_volume_regular_gpu>()->getData();
    float* model2data = model2[isub]->as<cart_volume_regular_gpu>()->getData();
    float* model3data = model3[isub]->as<cart_volume_regular_gpu>()->getData();
    float* model4data = model4[isub]->as<cart_volume_regular_gpu>()->getData();
    float* model5data = model5[isub]->as<cart_volume_regular_gpu>()->getData();
    float* model6data = model6[isub]->as<cart_volume_regular_gpu>()->getData();
    float* model7data = model7[isub]->as<cart_volume_regular_gpu>()->getData();

    dim3 threads(128, 1, 1);
    dim3 blocks((xsize - 1) / threads.x + 1, (model1_gpu->ax2()->ntot - 1) / threads.y + 1,
                (model1_gpu->ax3()->ntot - 1) / threads.z + 1);

    init_cijs_kernel<<<blocks, threads>>>(vp, eps, delta, vs, theta, phi, model1data, model2data, model3data, model4data,
                                            model5data, model6data, model7data, ixbeg, ixend, iybeg, iyend, izbeg, izend,
                                            model1_gpu->vol_idx());
    CUDA_CHECK_ERROR(__FILE__, __LINE__);
}

void simulator2SetUp::initKernels()
{
    kernel_base_ = new kernel_base(order, bc_opt, nsubs, dt, d1, d2, d3, false, field1, field2, field3, field4, field11, field21,
                                            field31, field41, sponge_coef_gpu, Model8, Model9, Model81, Model91, wsi, Model10);

    kernel = new kernel_type2(bc_opt, order, nsubs, d1, d2, d3, dt, field1,
                                            field2, field3, field4, field3_rhs, field4_rhs, model1, model2, model3, model4, Model11,
                                            model5, model6, model7, dfield1dx, dfield1dy, dfield1dz, dfield2dx, dfield2dy, dfield2dz,
                                            snap_field1, snap_field2, pml, pml_coef, sponge_coef_gpu);


}

void simulator2SetUp::execute(bool has_src_p, bool has_qp_)
{
    float milliseconds_h = 0;
    float milliseconds_v = 0;
    float milliseconds_s = 0;
    float elapsedTime = 0;
    TIMED_BLOCK("Total") {
        float max_pressure = 500.0f;
        TIMED_BLOCK("Forward") {
            if (has_src_p) { // source group in lib/grid/src_group.cpp
                initializeSource(max_pressure);
            }

            for (int iter = 0; iter < _niter; ++iter) {

                if (has_qp_)
                    for (int isub = 0; isub < nsubs; ++isub) {
                        kernel_base_->kernel_loop1(isub);
                    }
                else
                    for (int isub = 0; isub < nsubs; ++isub) {
                        kernel_base_->kernel_loop2(isub);
                    }

                if (has_qp_) {
                    for (int isub = 0; isub < nsubs; ++isub) {
                        kernel_base_->kernel_loop3(isub);
                    }
                }

                cudaEvent_t start_halo, stop_halo;
                cudaEventCreate(&start_halo);
                cudaEventCreate(&stop_halo); 

                cudaEventRecord(start_halo);
                cudaEventSynchronize(start_halo);
                halomgr_s->start_update();
                halomgr_s->finish_update();
                cudaEventRecord(stop_halo);
                cudaEventSynchronize(stop_halo);
                elapsedTime = 0;
                cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                milliseconds_h +=elapsedTime;  

                cudaEvent_t start_v, stop_v;
                cudaEventCreate(&start_v);
                cudaEventCreate(&stop_v);
                cudaEventRecord(start_v);
                for (int isub = 0; isub < nsubs; ++isub) {
                    kernel_base_->kernel_loop4(isub);
                } // isub

                for (int isub = 0; isub < nsubs; ++isub) {
                    kernel->fwd_main_loop(isub, false, false, false, false);

                } //nsubs
                cudaEventRecord(stop_v);
                cudaEventSynchronize(stop_v);
                elapsedTime = 0;
                cudaEventElapsedTime(&elapsedTime, start_v, stop_v);
                milliseconds_v += elapsedTime;

                cudaEventRecord(start_halo);
                cudaEventSynchronize(start_halo);
                halomgr_sd->start_update();
                halomgr_sd->finish_update();
                cudaEventRecord(stop_halo);
                cudaEventSynchronize(stop_halo);
                elapsedTime = 0;
                cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                milliseconds_h +=elapsedTime;  

                cudaEvent_t start_s, stop_s;
                cudaEventCreate(&start_s);
                cudaEventCreate(&stop_s);
                cudaEventRecord(start_s);                    
                for (int isub = 0; isub < nsubs; ++isub) {
                    kernel->kernel_loop4_derivative(isub);
                } // isub

                for (int isub = 0; isub < nsubs; ++isub) {
                    kernel->fwd_main_loop_2(isub, false, false, false, false);
                }
                cudaEventRecord(stop_s);
                cudaEventSynchronize(stop_s);   
                elapsedTime = 0;
                cudaEventElapsedTime(&elapsedTime, start_s, stop_s);
                milliseconds_s += elapsedTime; 

                if (!_fwd_only) {               
                    if (iter == next_snap) {

                        if (!useZFP_) {
                            snapReaderWriter->lock_for_write_uncompressed_fwi_snapshot();
                        } else {
                            snapReaderWriter->lock_for_write_compressed_fwi_snapshot();
                        }
                        bool skip_halo = true;
                        bool no_skip_halo = false;
                        for (int isub = 0; isub < nsubs; ++isub) {
                            // THIS IS NOT COMPUTATIONALLY CORRECT. We need the correct zone id. Taking this approach as we are not worried about correctness.
                            snap_field3[isub]->copyData(field3[isub], skip_halo);
                            snap_field4[isub]->copyData(field4[isub], skip_halo);
                        }

                        cudaEventRecord(start_halo);
                        cudaEventSynchronize(start_halo);
                        halomgr_s->start_update();
                        halomgr_s->finish_update();
                        cudaEventRecord(stop_halo);
                        cudaEventSynchronize(stop_halo);
                        elapsedTime = 0;
                        cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                        milliseconds_h +=elapsedTime;  

                        for (int isub = 0; isub < nsubs; ++isub) {
                            snap_field1[isub]->copyData(field1[isub], no_skip_halo);
                            snap_field2[isub]->copyData(field2[isub], no_skip_halo);
                        }

                        CUDA_TRY(cudaEventRecord(writeSnapshotsCompleteEvent_, 0));
                        if (!useZFP_) {
                            snapReaderWriter->start_write_uncompressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p);
                        } else {
                            snapReaderWriter->start_write_compressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p,
                                                                                zipped_corr_buff_, zfpFields_, zfpStream_);
                        }                    
                        next_snap += _xcorr_step;
                    }

                    for (int isub = 0; isub < nsubs; ++isub) {
                        computefield5(isub);
                    }
                } // !_fwd_only
            } // end iter
        } // End TIMED_BLOCK ("Forward")

        if (!_fwd_only) {
            if (!useZFP_) {
                snapReaderWriter->start_read_uncompressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p);
            } else {
                snapReaderWriter->start_read_compressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p,
                                                                        zipped_corr_buff_, zfpFields_, zfpStream_);
            }

            TIMED_BLOCK("Adjoint") {
                next_snap -= _xcorr_step;
                for (int iter = _niter - 1; iter >= 0; --iter) {        
                    cudaEvent_t start_s, stop_s;
                    cudaEventCreate(&start_s);
                    cudaEventCreate(&stop_s);
                    cudaEventRecord(start_s);   
                    if (has_qp_) {
                        for (int isub = 0; isub < nsubs; ++isub) {
                            kernel_base_->adj_kernel_loop3(isub);
                        }
                    }
                    
                    for (int isub = 0; isub < nsubs; ++isub) {
                        kernel->adj_main_loop(isub, false, false, false, false);
                    }
                    cudaEventRecord(stop_s);
                    cudaEventSynchronize(stop_s); 
                    elapsedTime = 0;
                    cudaEventElapsedTime(&elapsedTime, start_s, stop_s);
                    milliseconds_s += elapsedTime; 

                    cudaEvent_t start_halo, stop_halo;
                    cudaEventCreate(&start_halo);
                    cudaEventCreate(&stop_halo); 
                    
                    cudaEventRecord(start_halo);
                    cudaEventSynchronize(start_halo);
                    halomgr_sd->start_update();
                    halomgr_sd->finish_update();
                    cudaEventRecord(stop_halo);
                    cudaEventSynchronize(stop_halo);
                    elapsedTime = 0;
                    cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                    milliseconds_h +=elapsedTime;  

                    cudaEvent_t start_v, stop_v;
                    cudaEventCreate(&start_v);
                    cudaEventCreate(&stop_v);
                    cudaEventRecord(start_v);                       
                    for (int isub = 0; isub < nsubs; ++isub) {
                        kernel->kernel_loop4_derivative(isub);
                    } // isub

                    for (int isub = 0; isub < nsubs; ++isub) {
                        kernel->adj_main_loop_2(isub, false, false, false, false);
                    }

                    if (has_qp_) {
                        for (int isub = 0; isub < nsubs; ++isub) {
                            kernel_base_->adj_kernel_loop1(isub);
                        }
                    } else {
                        for (int isub = 0; isub < nsubs; ++isub) {
                            kernel_base_->kernel_loop2(isub);
                        }
                    }
                    cudaEventRecord(stop_v);
                    cudaEventSynchronize(stop_v);     
                    elapsedTime = 0;
                    cudaEventElapsedTime(&elapsedTime, start_v, stop_v);
                    milliseconds_v += elapsedTime;             

                    cudaEventRecord(start_halo);
                    cudaEventSynchronize(start_halo);
                    halomgr_s->start_update();
                    halomgr_s->finish_update();
                    cudaEventRecord(stop_halo);
                    cudaEventSynchronize(stop_halo);
                    elapsedTime = 0;
                    cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                    milliseconds_h +=elapsedTime; 

                    for (int isub = 0; isub < nsubs; ++isub) {
                        kernel_base_->kernel_loop4(isub);
                    }

                    if (iter == next_snap) {
                        if (!useZFP_) {
                            snapReaderWriter->finish_read_uncompressed_fwi_snapshot();
                        } else {
                            snapReaderWriter->finish_read_compressed_fwi_snapshot();
                        }
                        for (int isub = 0; isub < nsubs; ++isub) {
                            kernel->update_grad_loop_1(isub, snap_field3[isub], snap_field4[isub], grad1[isub], field3[isub], field4[isub]);
                        } //isub
                        for (int isub = 0; isub < nsubs; ++isub) {
                            kernel->snaps_loop(isub);
                        } // isub
                        
                        cudaEventRecord(start_halo);
                        cudaEventSynchronize(start_halo);
                        halomgr_sd->start_update();
                        halomgr_sd->finish_update();
                        cudaEventRecord(stop_halo);
                        cudaEventSynchronize(stop_halo);
                        elapsedTime = 0;
                        cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                        milliseconds_h +=elapsedTime; 

                        for (int isub = 0; isub < nsubs; ++isub) {
                            kernel->kernel_loop4_derivative(isub);
                        } // isub
                        for (int isub = 0; isub < nsubs; ++isub) {
                            kernel->update_grad_loop_2(isub, dfield1dx[isub], dfield1dy[isub], dfield1dz[isub],
                                                            dfield2dx[isub], dfield2dy[isub], dfield2dz[isub], grad2[isub]);
                        } // isub
                        CUDA_TRY(cudaEventRecord(readSnapshotsCompleteEvent_, 0));
                        next_snap -= _xcorr_step;
                        if (next_snap >= 0) {
                            if (!useZFP_) {
                                snapReaderWriter->start_read_uncompressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p);
                            } else {
                                snapReaderWriter->start_read_compressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p,
                                                                                        zipped_corr_buff_, zfpFields_, zfpStream_);
                            }
                        }
                    }                
                } // end iter
            } //end TIMED_BLOCK ("Adjoint")
        }
    } // End TIMED_BLOCK ("Total")

    timer locTotal = timer::get_timer("Total");
    timer locForward = timer::get_timer("Forward");
    double locTotalTime = locTotal.get_elapsed();
    double locForwardTime = locForward.get_elapsed();

    double maxTotalTime, maxForwardTime, maxAdjointTime;
    float max_milliseconds_s, max_milliseconds_v, max_milliseconds_h;
    MPI_Reduce(&locTotalTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, _comm);
    
    MPI_Reduce(&locForwardTime, &maxForwardTime, 1, MPI_DOUBLE, MPI_MAX, 0, _comm);
    MPI_Reduce(&milliseconds_v, &max_milliseconds_v, 1, MPI_FLOAT, MPI_MAX, 0, _comm);
    MPI_Reduce(&milliseconds_s, &max_milliseconds_s, 1, MPI_FLOAT, MPI_MAX, 0, _comm);
    MPI_Reduce(&milliseconds_h, &max_milliseconds_h, 1, MPI_FLOAT, MPI_MAX, 0, _comm);

    if (!_fwd_only) {
        timer locAdjoint = timer::get_timer("Adjoint");    
        double locAdjointTime = locAdjoint.get_elapsed();
        MPI_Reduce(&locAdjointTime, &maxAdjointTime, 1, MPI_DOUBLE, MPI_MAX, 0, _comm);
    }

    if(_rank == 0) {
        std::cout<<"Total Time: "<<maxTotalTime<<std::endl;
        std::cout<<"Max time for Forward: "<<maxForwardTime<<std::endl;
        if (!_fwd_only)
            std::cout<<"Max time for Adjoint: "<<maxAdjointTime<<std::endl;
        std::cout<<"Max time for velocity: "<<max_milliseconds_v/1000.0f<<std::endl;
        std::cout<<"Max time for stress: "<<max_milliseconds_s/1000.0f<<std::endl;
        std::cout<<"Max time for halo-exchange: "<<max_milliseconds_h/1000.0f<<std::endl;
    }
}

void simulator2SetUp::initializeSource(float pressure)
{
    //find the cart_vol that has the point position _npoints/2 + 4;
    for (int i = 0; i < nsubs; i++) {
        int splitId[3];
        decompmgr->getSplitLocalSubDomID(i, splitId);
        int stOff[3];
        int endOff[3];
        for (int d = 0; d < 3; ++d) {
            stOff[d] = decompmgr->getOffset(splitId[d], d);
            endOff[d] = stOff[d] + decompmgr->getNumPtsSplit(splitId[d], d) - 1;
        } //end d
        if (stOff[0] <= n_total[0] / 2 && endOff[0] >= n_total[0] / 2 && stOff[1] <= n_total[1] / 2 &&
            endOff[1] >= n_total[1] / 2 && stOff[2] <= n_total[2] / 2 &&
            endOff[2] >= n_total[2] / 2) { // found that point
            int x = n_total[0] / 2 - stOff[0] + radius;
            int y = n_total[1] / 2 - stOff[1] + radius;
            int z = n_total[2] / 2 - stOff[2] + radius;
            init_src_kernel<<<1, 1>>>(field1[i]->as<cart_volume_regular_gpu>()->getData(),
                                        field2[i]->as<cart_volume_regular_gpu>()->getData(), x, y, z, pressure,
                                        field1[i]->as<cart_volume_regular_gpu>()->vol_idx());
            CUDA_CHECK_ERROR(__FILE__, __LINE__);
        }
    }
}

void simulator2SetUp::computefield5(int isub)
{
    axis* ax1 = field5_gpu[isub]->as<cart_volume_regular_gpu>()->ax1();
    axis* ax2 = field5_gpu[isub]->as<cart_volume_regular_gpu>()->ax2();
    axis* ax3 = field5_gpu[isub]->as<cart_volume_regular_gpu>()->ax3();
    dim3 threads(128, 1, 1);
    dim3 blocks((ax1->ntot - 1) / threads.x + 1, (ax2->ntot - 1) / threads.y + 1, (ax3->ntot - 1) / threads.z + 1);

    compute_pressure_kernel<<<blocks, threads>>>(
        field5_gpu[isub]->as<cart_volume_regular_gpu>()->getData(),
        field1[isub]->as<cart_volume_regular_gpu>()->getData(), field2[isub]->as<cart_volume_regular_gpu>()->getData(),
        ax1->ntot, ax2->ntot, ax3->ntot, field5_gpu[isub]->as<cart_volume_regular_gpu>()->vol_idx());
    CUDA_CHECK_ERROR(__FILE__, __LINE__);

    CUDA_CHECK_ERROR(__FILE__, __LINE__);
}