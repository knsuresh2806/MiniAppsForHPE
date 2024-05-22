#include <stdio.h>
#include "simulator2SetUp.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    // int commSize;
    int rank;
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
      fwd_only = std::atoi(argv[9]);
    else 
      fwd_only = 0;

    // MPI_Comm_split(MPI_COMM_WORLD, has_gpu, rank, &comm1);
    // if (has_gpu) {
    //     MPI_Comm_size(comm1, &commSize);
    //     int needcpu = 4;
    //     if (commSize < needcpu){
    //         printf("Not enough ranks with GPUs for this fixture\n");
    //         return 1;
    //     }
    //     MPI_Comm_split(comm1, rank < needcpu, rank, &comm2);
    //     if (rank < needcpu) {
            int nzones[3] = { 3, 3, 3 };
            int ranks[3] = { npe_x, npe_y, npe_z };
            simulator2SetUp* simulatorSetUp_obj = new simulator2SetUp(npoints, npml, niter, xcorr_step, nzones, ranks, 4, 4, bitPerFloat, fwd_only, MPI_COMM_WORLD);
            bool has_qp_ = false;
            bool has_src_p = true;
            simulatorSetUp_obj->execute(has_src_p, has_qp_);
            MPI_Barrier(MPI_COMM_WORLD);
            if (!rank) {
                printf("Done\n");
            }
            return 0;
    //     }
    // }
}