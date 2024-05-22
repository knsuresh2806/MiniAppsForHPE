#ifndef SIMULATOR3SETUP_H
#define SIMULATOR3SETUP_H

#include "cuda_utils.h"
#include "mpi_utils.h"
#include "axis.h"
#include "cart_volume.h"
#include "decompositionManager3D.h"
#include "haloManager3D_gpu.h"

#include <vector>
#include "rtm_snap.h"                                                                                                    
#include "rtm_snap_gpu.h" 
#include <string>
#include <zfp.h>

class simulator3SetUp
{
public:
    // Set up a decomposed domain with variable number of ranks per dimension
    simulator3SetUp() = delete;
    simulator3SetUp(int npoints, int npml, int niter, int xcorr_step, int nzones[3], int ranks[3], int radius_lo, int radius_hi, float bitPerFloat, int fwd_only, MPI_Comm communicator);
    ~simulator3SetUp();
    void execute(bool has_src_p, bool has_qp_);

private:
    int get_n_for_zone(int izone, int nzones);
    void init_cijs(int isub);
    void initializeSource(float p);
    void computeP(int isub);
    
    MPI_Comm _comm;
    int _rank;

    std::vector<axis*> local_X_axes;
    std::vector<axis*> local_X_axes_noHalo;
    std::vector<axis*> local_Y_axes;
    std::vector<axis*> local_Y_axes_noHalo;
    std::vector<axis*> local_Z_axes;
    std::vector<axis*> local_Z_axes_noHalo;

    double d1 = 1.0;
    double d2 = 1.0;
    double d3 = 1.0;

    float vp = 1500.0f; 
    float eps = 0.0f;
    float delta = 0.0f;
    float vs = 0.0f;
    float theta = 0.0f;
    float phi = 0.0f;

    int bc_opt = 0;
    int NMECH = 3;
    int isub = 0;
    axis* axgiX = nullptr;
    axis* axgiY = nullptr;
    axis* axgiZ = nullptr;
    axis* axgX = nullptr;
    axis* axgY = nullptr;
    axis* axgZ = nullptr;

    cart_volume<float>** f1;
    cart_volume<float>** f2;
    cart_volume<float>** f3;
    cart_volume<float>** f4;
    cart_volume<float>** f5;
    cart_volume<float>** f6;
    cart_volume<float>** f7;
    cart_volume<float>** f8;
    cart_volume<float>** f9;
    cart_volume<float>** M1;
    cart_volume<float>** M2;
    cart_volume<float>** M3;
    cart_volume<float>** M4;
    cart_volume<float>** M5;
    cart_volume<float>** M6;
    cart_volume<float>** M7;
    cart_volume<float>** M8;
    cart_volume<float>** M9;
    cart_volume<float>** M10;
    cart_volume<float>** M11;
    cart_volume<float>** M12;
    cart_volume<float>** M13;
    cart_volume<float>** M14;
    cart_volume<float>** M15;
    cart_volume<float>** M16;
    cart_volume<float>** M17;
    cart_volume<float>** M18;
    cart_volume<float>** M19;
    cart_volume<float>** M20;
    cart_volume<float>** M21;
    cart_volume<float>** M22;
    cart_volume<float>** d4;
    cart_volume<float>** d5;
    cart_volume<float>** d6;
    cart_volume<float>** d7;
    cart_volume<float>** d8;
    cart_volume<float>** d9;
    cart_volume<float>** f10;
    cart_volume<float>** f11;
    cart_volume<float>** f12;
    cart_volume<float>** f13;
    cart_volume<float>** f14;
    cart_volume<float>** f15;
    cart_volume<float>** f16;
    cart_volume<float>** f17;
    cart_volume<float>** f18;
    cart_volume<float>** f19;
    cart_volume<float>** f20;
    cart_volume<float>** f21;
    float **zprime;

    cart_volume<float>** p_gpu;
    std::vector<cart_volume<float>*> snap_f4;
    std::vector<cart_volume<float>*> snap_f5;
    std::vector<cart_volume<float>*> snap_f6;
    std::vector<cart_volume<float>*> snap_f7;
    std::vector<cart_volume<float>*> snap_f8;
    std::vector<cart_volume<float>*> snap_f9;
    const int order = 8;

    std::vector<cart_volume<float>*> snap_f1;
    std::vector<cart_volume<float>*> snap_f2;
    std::vector<cart_volume<float>*> snap_f3;
    cart_volume<float>** g1;
    cart_volume<float>** g2;
    std::vector<cart_volume<float>*> corrBuffList;
    std::vector<realtype*> zipped_corr_buff_; // compressed correlation buffer
    std::vector<int> corrBuff_size;
    float memoryPercent = 0.88f;
    int numSnaps;
    int snap_type = 0; // original without compression
    std::unique_ptr<rtm_snap> snapReaderWriter;
    cudaEvent_t writeSnapshotsCompleteEvent_;
    cudaEvent_t readSnapshotsCompleteEvent_;
    file_snap* file_snap_p;
    std::string scratch_folder;
    int next_snap = 0;
    int _xcorr_step;
    float _bitPerFloat;
    bool useZFP_; // whether use ZFP library to compress correlation buffer
    std::vector<zfp_field*> zfpFields_;
    zfp_stream* zfpStream_;

    int numZonesinX;
    int numZonesinY;
    int numZonesinZ;

    int* numPts[3] = { nullptr, nullptr, nullptr };
    int _npml;
    int _npoints;
    int _niter;
    int _fwd_only;
    int radius, r_lo, r_hi;
    int nsubs;
    int nf;
    int n_total[3];
    float*** costs = nullptr;
    int*** stencilWidths[3][2] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    std::vector<cart_volume<float>**> sf;
    std::vector<cart_volume<float>**> vf;
    std::vector<cart_volume<float>**> vdf;
    std::vector<std::vector<int>> subDomStencilWidths_s[3][2];
    std::vector<std::vector<int>> subDomStencilWidths_v[3][2];
    std::vector<std::vector<int>> subDomStencilWidths_vd[3][2];
    decompositionManager3D* decompmgr = nullptr;
    haloManager3D_gpu* halomgr_s = nullptr;
    haloManager3D_gpu* halomgr_v = nullptr;
    haloManager3D_gpu* halomgr_vd = nullptr;
};

#endif