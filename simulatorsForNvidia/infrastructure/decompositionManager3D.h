
/*
   Simple (tensor-product) decomposition manager for 3-D domains.
Author: Rahul S. Sampath
*/

#ifndef _DECOMPOSITION_MANAGER_3D
#define _DECOMPOSITION_MANAGER_3D

#include <mpi.h>
#include "decompositionManager.h"
#include "std_const.h"
#include "mpi_const.h"
#include "grid_const.h"
#include "emsl_error.h"
#include <algorithm>

class decompositionManager3D : public decompositionManager
{
public:
    /*
     * Constructor
     * numZones: Number of non-empty zones (including boundary) in X, Y and Z directions.
     * numPts: Global number of points in X, Y and Z directions for each zone.
     * npe: Number of MPI ranks in X, Y and Z directions.
     * costs: Weight per point (used in partitioning) for each zone.
     * sw: Maximum stencil widths in each direction ([X, Y and Z] x [Negative and Positive]) for each zone.
     * comm_world: Global communicator.
     * Notes:
     * Uses Right-Handed-Coordinate System:
     * Left = -ve X, Right = +ve X
     * Back = -ve Y, Front = +ve Y
     * Top = -ve Z, Bottom = +ve Z
     * (X = 0 and Y = 1 and Z = 2)
     * (Negative = 0 and Positive = 1)
     */
    decompositionManager3D(int _numZones[3], int* numPts[3], realtype*** costs, int*** sw[3][2], int npe[3],
                           MPI_Comm _comm_world, int hack_first_interior_z_rank_max_nz);

    //Destructor
    ~decompositionManager3D();

    //Prints the domain decomposition information for each rank
    void print(FILE* fd);

    //Returns true if the sub-domain has a neighbor (either on the same rank or on the adjacent rank) in a particular direction and false otherwise
    void has_neighbor(int id, bool hasNeighbor[3][2]);

    //Returns the ranks (-1 if it does not exist) of adjacent processors in the cartesian topology
    void getNeighbors(int nh[_NH_ID_3D_TOTAL])
    {
        for (int i = 0; i < _NH_ID_3D_TOTAL; ++i) {
            nh[i] = neighbor[i];
        } //end i
    }

    //Returns the rank in dimension 'd' containing the point 'pt'
    int getRankContainingPt(int d, int pt)
    {
        int* pos = std::lower_bound(offsets[d], offsets[d] + dims[d], pt);
        if (pos == (offsets[d] + dims[d])) {
            return (dims[d] - 1);
        }
        EMSL_VERIFY((*pos) >= pt);
        if ((*pos) == pt) {
            return (pos - offsets[d]);
        }
        return (pos - offsets[d] - 1);
    }

    //Returns the subset of cart_comm with same X coordinate in the processor grid
    MPI_Comm get_xcomm()
    {
        if (!xcommSet) {
            int remain_dims[3] = { 0, 1, 1 };
            MPI_Cart_sub(cart_comm, remain_dims, &xcomm);
            xcommSet = true;
        }
        return xcomm;
    }

    //Returns the subset of cart_comm with same Y coordinate in the processor grid
    MPI_Comm get_ycomm()
    {
        if (!ycommSet) {
            int remain_dims[3] = { 1, 0, 1 };
            MPI_Cart_sub(cart_comm, remain_dims, &ycomm);
            ycommSet = true;
        }
        return ycomm;
    }

    //Returns the subset of cart_comm with same Z coordinate in the processor grid
    MPI_Comm get_zcomm()
    {
        if (!zcommSet) {
            int remain_dims[3] = { 1, 1, 0 };
            MPI_Cart_sub(cart_comm, remain_dims, &zcomm);
            zcommSet = true;
        }
        return zcomm;
    }

    //Returns the subset of cart_comm containing ranks within selected subgrid
    //activeRange (X, Y, Z/ min,max) - range containing non-empty groups
    //groupRange (X, Y, Z/ min,max) - range of group including my rank
    MPI_Comm createSubComm(const int activeRange[3][2], const int groupRange[3][2]);

    //Split the number of local sub-domains into its components in each dimension.
    void getSplitNumLocalSubDom(int nLocSubDom[3])
    {
        nLocSubDom[0] = numLocalSubDom[0];
        nLocSubDom[1] = numLocalSubDom[1];
        nLocSubDom[2] = numLocalSubDom[2];
    }

    //Split local sub-domain id into its components (in local ordering) in each dimension.
    void getSplitLocalSubDomID(int id, int splitLocalSubDomId[3])
    {
        splitLocalSubDomId[0] = id % (numLocalSubDom[0]);
        splitLocalSubDomId[1] = (id % (numLocalSubDom[1] * numLocalSubDom[0])) / (numLocalSubDom[0]);
        splitLocalSubDomId[2] = id / (numLocalSubDom[1] * numLocalSubDom[0]);
    }

    //Split local sub-domain id into its components (in global ordering) in each dimension.
    void getSplitGlobalSubDomID(int id, int splitGlobalSubDomId[3])
    {
        getSplitLocalSubDomID(id, splitGlobalSubDomId);
        for (int i = 0; i < 3; ++i) {
            splitGlobalSubDomId[i] += splitGlobalSubDomIdOff[i];
        } //end i
    }

    //Combine the component sub-domain id in each dimension to give an index into the list of local sub-domains.
    int getSubDomIDfromSplit(int idX, int idY, int idZ)
    {
        return ((((idZ * numLocalSubDom[1]) + idY) * numLocalSubDom[0]) + idX);
    }

    int getSubDomIDtoSend(NH_ID_3D nh, RECV_TYPE_3D rType, int othIdX, int othIdY, int othIdZ);

    //Get ids (-1 if it does not exist) of local sub-domains adjacent to a given sub-domain.
    void getLocalNeighborID(int id, int localNhId[_NH_ID_3D_TOTAL]);

    //Get coordinates (X, Y, Z) of this rank in processor grid
    void get_coords(int _coords[3])
    {
        _coords[0] = coords[0];
        _coords[1] = coords[1];
        _coords[2] = coords[2];
    }

    //Return the number of points on this rank in each direction based on split sub-domain id
    int getNumPtsSplit(int splitSubDomId, int direction)
    {
        return numLocalSplit[direction][splitSubDomLocalToGlobalMap[direction][splitSubDomId]];
    }

    //Return the number of points on the given 1D rank in the given direction
    int getNumPtsOnRank(int direction, int rank1D) { return counts[direction][rank1D]; }

    //Returns the offset in each direction based on split sub-domain id
    int getOffset(int splitSubDomId, int direction) { return splitOff[direction][splitSubDomId]; }

    //Get the starting and end offsets in each direction for given sub-domain id
    void getStartAndEndOffsets(int id, int stOff[3], int endOff[3])
    {
        int splitId[3];
        getSplitLocalSubDomID(id, splitId);
        for (int d = 0; d < 3; ++d) {
            stOff[d] = splitOff[d][splitId[d]];
            endOff[d] = gN[d] - (stOff[d] + numLocalSplit[d][splitSubDomLocalToGlobalMap[d][splitId[d]]]);
        } //end d
    }

    //Return the offset on the given 1D rank in the given direction
    int getOffsetForRank(int direction, int rank1D) { return offsets[direction][rank1D]; }

private:
    void compute_neighbors();
    void compute_partition(int* numPts[3], realtype*** costs, int*** sw[3][2], int hack_first_interior_z_rank_max_nz);
    void create_subdomains();
    void check_grain_size(int*** sw[3][2]);

    //Return the local subdomain id (-1 if it does not exist) for given zone
    int getSubDomIDofZone(int zoneX, int zoneY, int zoneZ) { return subDomGlobalToLocalMap[zoneZ][zoneY][zoneX]; }

    MPI_Comm xcomm; //subset of cart_comm with same X coordinate in the processor grid
    MPI_Comm ycomm; //subset of cart_comm with same Y coordinate in the processor grid
    MPI_Comm zcomm; //subset of cart_comm with same Z coordinate in the processor grid

    bool xcommSet;
    bool ycommSet;
    bool zcommSet;

    int neighbor[_NH_ID_3D_TOTAL]; //ranks (-1 if it does not exist) of neighboring processors in cart_comm.

    int gN[3]; //global number of points in each dimension (X, Y, Z)

    int off[3]; //Offset (X, Y, Z) for first point on this rank

    int* splitOff[3]; //Offset for each sub-domain on this rank in each dimension (X, Y, Z)

    int* counts[3]; //Number of points on each rank in each dimension (X, Y, Z)

    int* offsets[3]; //Offsets for each rank in each dimension (X, Y, Z)

    int splitGlobalSubDomIdOff
        [3]; //Offset for the sub-domains (in global ordering) on this rank in each dimension (X, Y, Z)

    int numLocalSubDom[3]; //number of subdomains on this processor in each dimension (X, Y, Z)

    /* 
       Map each of the possible zones (global sub-domains) to an index (-1 or (0 to (totalNumLocalSubDom - 1))) into the list of local subdomains.
       -1 indicates that the sub-domain does not exist on this rank. 
       The first (slowest) dimension is Z, the second (middle) dimension is Y and the third (fastest) dimension is X.
       */
    int*** subDomGlobalToLocalMap;

    //Map each of the local sub-domains to the corresponding global domains in each direction (X, Y, Z).
    int* subDomLocalToGlobalMap[3];

    //Map each of the local split sub-domain id to the corresponding global domains in each direction (X, Y, Z).
    int* splitSubDomLocalToGlobalMap[3];

    friend class ZoneManager3D;
};

#endif
