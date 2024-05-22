
#ifndef _DECOMPOSITION_MANAGER_
#define _DECOMPOSITION_MANAGER_

/*
   Abstract base class for simple (tensor-product) decomposition managers.
Author: Rahul S. Sampath
*/

#include <mpi.h>
#include "std_const.h"

class decompositionManager
{
public:
    decompositionManager(DIMENSION _dimen, int* _numZones, int** numPts, int* npe, MPI_Comm _comm_world);

    virtual ~decompositionManager();

    //Returns rank in comm_world.
    int get_world_rank() { return world_rank; }

    //Returns the number of MPI ranks
    int get_npes() { return npes; }

    //Returns rank in cartesian topology
    int get_rank() { return cart_rank; }

    //Returns the communicator with cartesian topology
    MPI_Comm get_comm() { return cart_comm; }

    int getTotalNumLocalSubDom() { return totalNumLocalSubDom; }

    //Return the number of points on this rank in each direction
    int getNumPts(int direction) { return numLocal[direction]; }

protected:
    //Return the number of points on this rank in each direction in each zone
    int getNumPtsInZone(int direction, int zone) { return numLocalSplit[direction][zone]; }

    MPI_Comm comm_world;
    int world_rank; //rank in comm_world
    int npes;       //number of MPI ranks in comm_world (same in cart_comm)

    MPI_Comm cart_comm;   //comm using cartesian topology
    int cart_rank;        //rank in cart_comm
    MPI_Group cart_group; //group for cart_comm

    int* coords;      //coordinates in processor grid
    int* dims;        //number of MPI ranks in each dimension
    int numNeighbors; //number of neighboring processors in cart_comm

    int* numZones; //Global number of zones in each dimension

    int** zoneOff; //Global offsets for each zone in each dimension

    int* numLocal; //Number of points on this rank in each dimension

    int** numLocalSplit; //Number of points on this rank in each zone in each dimension

    /*
       Number of points (in the dimension of interest) in the domains that are adjacent to this rank.
       The first dimension identifies the axis.
       The second dimension identifies the neighbor: Negative = 0 or Positive = 1.
       */
    int** numNeighborPts;

    /*
       Zone ID for the domains that are adjacent to this rank.
       The first dimension identifies the axis.
       The second dimension identifies the neighbor: Negative = 0 or Positive = 1.
       */
    int** neighborSubDom;

    int totalNumLocalSubDom; //total number of subdomains on this processor

private:
    DIMENSION dimen;

    int numDim; //number of dimensions
};

#endif
