
/*
Author: Rahul S. Sampath
*/

#include <mpi.h>
#include "decompositionManager.h"
#include "emsl_error.h"
#include <cstdio>

decompositionManager ::decompositionManager(DIMENSION _dimen, int* _numZones, int** numPts, int* npe,
                                            MPI_Comm _comm_world)
{
    dimen = _dimen;

    if (dimen == _2D) {
        numDim = 2;
    } else {
        numDim = 3;
    }

    numZones = new int[numDim];
    for (int i = 0; i < numDim; ++i) {
        numZones[i] = _numZones[i];
    } //end i

    comm_world = _comm_world;

    MPI_Comm_rank(comm_world, &world_rank);

    MPI_Comm_size(comm_world, &npes);

    //Basic checks on input
    for (int i = 0; i < numDim; ++i) {
        EMSL_VERIFY(numZones[i] > 0);
        for (int j = 0; j < numZones[i]; ++j) {
            EMSL_VERIFY(numPts[i][j] > 0);
        } //end j
    }     //end i

    int prod_npe = 1;
    for (int i = 0; i < numDim; ++i) {
        prod_npe *= npe[i];
    }

    if (prod_npe != npes) {
        fprintf(stderr, "ERROR: decompositionManager: rank: %d, npes: %d", world_rank, npes);
        for (int i = 0; i < numDim; ++i) {
            fprintf(stderr, ", npe_%d: %d", i, npe[i]);
        }
        fprintf(stderr, "\n");
        EMSL_VERIFY(false);
    }

    //Create cartesian topology
    dims = new int[numDim];
    for (int i = 0; i < numDim; ++i) {
        dims[i] = npe[i];
    } //end i

    int* periods = new int[numDim];
    for (int i = 0; i < numDim; ++i) {
        periods[i] = 0;
    } //end i

    MPI_Cart_create(comm_world, numDim, dims, periods, 1, &cart_comm);
    EMSL_VERIFY(cart_comm != MPI_COMM_NULL);

    delete[] periods;

    MPI_Comm_rank(cart_comm, &cart_rank);

    coords = new int[numDim];
    MPI_Cart_coords(cart_comm, cart_rank, numDim, coords);

    MPI_Comm_group(cart_comm, &cart_group);

    numLocal = new int[numDim];

    //Global offsets for each zone in each dimension
    zoneOff = new IntPtr[numDim];
    for (int i = 0; i < numDim; ++i) {
        zoneOff[i] = new int[numZones[i]];
        zoneOff[i][0] = 0;
        for (int j = 1; j < numZones[i]; ++j) {
            zoneOff[i][j] = zoneOff[i][j - 1] + numPts[i][j - 1];
        } //end j
    }     //end i

    numLocalSplit = new IntPtr[numDim];
    numNeighborPts = new IntPtr[numDim];
    neighborSubDom = new IntPtr[numDim];
    for (int i = 0; i < numDim; ++i) {
        numLocalSplit[i] = new int[numZones[i]];
        numNeighborPts[i] = new int[2];
        neighborSubDom[i] = new int[2];
    } //end i
}

decompositionManager ::~decompositionManager()
{
    delete[] numZones;
    delete[] dims;
    delete[] coords;
    delete[] numLocal;

    for (int i = 0; i < numDim; ++i) {
        delete[](numLocalSplit[i]);
        delete[](numNeighborPts[i]);
        delete[](neighborSubDom[i]);
        delete[](zoneOff[i]);
    } //end i
    delete[] numLocalSplit;
    delete[] numNeighborPts;
    delete[] neighborSubDom;
    delete[] zoneOff;

    MPI_Group_free(&cart_group);

    MPI_Comm_free(&cart_comm);
}
