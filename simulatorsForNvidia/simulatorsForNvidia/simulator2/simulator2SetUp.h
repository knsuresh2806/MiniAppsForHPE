#ifndef SIMULATOR2SETUP_H
#define SIMULATOR2SETUP_H

#include "cuda_utils.h"
#include "mpi_utils.h"
#include "axis.h"
#include "cart_volume.h"
#include "decompositionManager3D.h"
#include "haloManager3D_gpu.h"

#include "kernel_type1_base.h"
#include "kernel_abstract_base.h"
#include "pml_volumes_cg3d.h"
#include "pml_coef3d_gpu.h"
#include "sponge_coef3d.h"
#include "sponge_coef3d_gpu.h"
#include <vector>
#include "kernel_type2_base.h"
#include "kernel_type2.h"
#include "rtm_snap.h"                                                                                                    
#include "rtm_snap_gpu.h" 
#include <string>
#include <zfp.h>

class simulator2SetUp
{
public:
    // Set up a decomposed domain with variable number of ranks per dimension
    simulator2SetUp() = delete;
    simulator2SetUp(int npoints, int npml, int niter, int xcorr_step, int nzones[3], int ranks[3], int radius_lo, int radius_hi, float bitPerFloat, int fwd_only, MPI_Comm communicator);
    ~simulator2SetUp();
    void execute(bool has_src_p, bool has_qp_);

private:
    int get_n_for_zone(int izone, int nzones);
    void allocateAndInitializeWsi() { wsi = { 1.0f, 1.0f, 1.0f }; }
    void init_cijs(int isub);
    void initKernels();
    void initializeSource(float pressure);
    void computefield5(int isub);
    
    MPI_Comm _comm;
    int _rank;

    double d1 = 1.0;
    double d2 = 1.0;
    double d3 = 1.0;
    float dt = 0.002734f;

    float vp = 1500.0f; // constant velocity
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

    cart_volume<float>** field1;
    cart_volume<float>** field11;
    cart_volume<float>** field2;
    cart_volume<float>** field21;
    cart_volume<float>** field3;
    cart_volume<float>** field31;
    cart_volume<float>** field3_rhs;
    cart_volume<float>** field4;
    cart_volume<float>** field41;
    cart_volume<float>** field4_rhs;
    cart_volume<float>** model1;
    cart_volume<float>** model2;
    cart_volume<float>** model3;
    cart_volume<float>** model4;
    cart_volume<float>** Model11;
    cart_volume<float>** dfield1dx;
    cart_volume<float>** dfield1dy;
    cart_volume<float>** dfield1dz;
    cart_volume<float>** dfield2dx;
    cart_volume<float>** dfield2dy;
    cart_volume<float>** dfield2dz;
    cart_volume<float>** model5;
    cart_volume<float>** model6;
    cart_volume<float>** model7;
    cart_volume<float>** field5_gpu;
    std::vector<cart_volume<float>*> snap_field1;
    std::vector<cart_volume<float>*> snap_field2;
    std::vector<cart_volume<float>**> Model8{ nullptr, nullptr, nullptr };
    std::vector<cart_volume<float>**> Model9{ nullptr, nullptr, nullptr };
    std::vector<cart_volume<float>**> Model81{ nullptr, nullptr, nullptr };
    std::vector<cart_volume<float>**> Model91{ nullptr, nullptr, nullptr };
    std::vector<cart_volume<float>**> Model10{ nullptr, nullptr, nullptr };
    pml_volumes_cg3d pml;
    const pml_coef3d_gpu pml_coef;
    kernel_abstract_base* kernel_base_;
    kernel_type2* kernel;
    const int order = 8;

    std::vector<cart_volume<float>*> snap_field3;
    std::vector<cart_volume<float>*> snap_field4;
    cart_volume<float>** grad1;
    cart_volume<float>** grad2;
    std::vector<cart_volume<float>*> corrBuffList;
    std::vector<realtype*> zipped_corr_buff_; // compressed correlation buffer
    std::vector<int> corrBuff_size;
    float memoryPercent = 0.88f;
    int numSnaps;
    int snap_type = 0; // original without compression. Do not change this!
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
    sponge_coef3d_gpu sponge_coef_gpu;
    sponge_coef3d sponge_coef_cpu;
    std::vector<float> wsi;

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
    std::vector<cart_volume<float>**> s_fields_for_halo;
    std::vector<cart_volume<float>**> sd_fields_for_halo;
    std::vector<std::vector<int>> subDomStencilWidths_s[3][2];
    std::vector<std::vector<int>> subDomStencilWidths_sd[3][2];
    decompositionManager3D* decompmgr = nullptr;
    haloManager3D_gpu* halomgr_s = nullptr;
    haloManager3D_gpu* halomgr_sd = nullptr;
};

#endif