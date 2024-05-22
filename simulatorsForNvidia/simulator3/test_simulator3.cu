#include <stdio.h>
#include "simulator3SetUp.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    // int commSize;
    int rank;
    // int ngpus;
    // CUDA_TRY(cudaGetDeviceCount(&ngpus));
    // int has_gpu = mpi_utils::get_mpi_node_rank(MPI_COMM_WORLD) < ngpus;
    // MPI_Comm comm1, comm2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc < 8) {
      if(!rank) {
        std::cout<<"Usage: <exe> npe_x npe_y npe_z npoints npml niter xcorr_step"<<std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    int npe_x = std::atoi(argv[1]);
    int npe_y = std::atoi(argv[2]);
    int npe_z = std::atoi(argv[3]);
    int npoints = std::atoi(argv[4]);
    int npml = std::atoi(argv[5]);
    int niter = std::atoi(argv[6]);
    int xcorr_step = std::atoi(argv[7]);
    float bitPerFloat;
    int fwd_only;

    if(argc > 8)
      bitPerFloat = std::atof(argv[8]);
    else
      bitPerFloat = 32.0f;
    if(argc > 9)
      fwd_only = std::atoi(argv[4]);
    else 
      fwd_only = 0;

    if (!rank)
    {
      std::cout << " ================================ " << std:: endl;
      std::cout << "npe_x : " << npe_x << std::endl;
      std::cout << "npe_y : " << npe_y << std::endl;
      std::cout << "npe_z : " << npe_z << std::endl;
      std::cout << "npoints : " << npoints << std::endl;
      std::cout << "npml : " << npml << std::endl;
      std::cout << "niter : " << niter << std::endl;
      std::cout << "xcorr_step : " << xcorr_step << std::endl;
      std::cout << "bitPerFloat : " << bitPerFloat << std::endl;
    }
    int nzones[3] = { 3, 3, 3 };
    int ranks[3] = { npe_x, npe_y, npe_z };
    simulator3SetUp* simulator3SetUp_obj 
      = new simulator3SetUp(npoints, npml, niter, xcorr_step, nzones, ranks, 4, 4, bitPerFloat, fwd_only, MPI_COMM_WORLD);
    bool has_qp_ = false;
    bool has_src_p = true;
    simulator3SetUp_obj->execute(has_src_p, has_qp_);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) {
        printf("Done\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank)
      std::cout << " ================================ " << std:: endl;
    
    MPI_Finalize();
    return 0;
}