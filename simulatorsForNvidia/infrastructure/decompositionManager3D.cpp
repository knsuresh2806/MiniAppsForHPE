
/*
Author: Rahul S. Sampath
*/

#include "decompositionManager3D.h"
#include "partUtils1D.h"
#include <vector>
#include <cstdio>
#include <algorithm>

decompositionManager3D ::decompositionManager3D(int _numZones[3], int* numPts[3], realtype*** costs, int*** sw[3][2],
                                                int npe[3], MPI_Comm _comm_world, int hack_first_interior_z_rank_max_nz)
    : decompositionManager(_3D, _numZones, numPts, npe, _comm_world)
{
    //Basic checks on input
    for (int k = 0; k < numZones[2]; ++k) {
        for (int j = 0; j < numZones[1]; ++j) {
            for (int i = 0; i < numZones[0]; ++i) {
                EMSL_VERIFY(costs[k][j][i] >= 0.0f);
            } //end i
        }     //end j
    }         //end k

    for (int d2 = 0; d2 < 3; ++d2) {
        for (int d1 = 0; d1 < 2; ++d1) {
            for (int k = 0; k < numZones[2]; ++k) {
                for (int j = 0; j < numZones[1]; ++j) {
                    for (int i = 0; i < numZones[0]; ++i) {
                        EMSL_VERIFY(sw[d2][d1][k][j][i] >= 0);
                    } //end i
                }     //end j
            }         //end k
        }             //end d1
    }                 //end d2

    //Get neighboring ranks in cart_comm
    compute_neighbors();

    //Partition the domain
    compute_partition(numPts, costs, sw, hack_first_interior_z_rank_max_nz);

    //Create local subdomains
    create_subdomains();

    //Catch over decomposition
    check_grain_size(sw);

    xcommSet = false;
    ycommSet = false;
    zcommSet = false;

    xcomm = MPI_COMM_NULL;
    ycomm = MPI_COMM_NULL;
    zcomm = MPI_COMM_NULL;
}

decompositionManager3D ::~decompositionManager3D()
{
    if (xcomm != MPI_COMM_NULL) {
        MPI_Comm_free(&xcomm);
    }

    if (ycomm != MPI_COMM_NULL) {
        MPI_Comm_free(&ycomm);
    }

    if (zcomm != MPI_COMM_NULL) {
        MPI_Comm_free(&zcomm);
    }

    for (int k = 0; k < numZones[2]; ++k) {
        for (int j = 0; j < numZones[1]; ++j) {
            delete[](subDomGlobalToLocalMap[k][j]);
        } //end j
        delete[](subDomGlobalToLocalMap[k]);
    } //end k
    delete[] subDomGlobalToLocalMap;

    for (int i = 0; i < 3; ++i) {
        delete[](counts[i]);
        delete[](offsets[i]);
        delete[](subDomLocalToGlobalMap[i]);
        delete[](splitSubDomLocalToGlobalMap[i]);
        delete[](splitOff[i]);
    } //end i
}

void
decompositionManager3D ::compute_partition(int* numPts[3], realtype*** costs, int*** sw[3][2],
                                           int hack_first_interior_z_rank_max_nz)
{
    //global size
    for (int i = 0; i < 3; ++i) {
        gN[i] = 0;
        for (int j = 0; j < numZones[i]; ++j) {
            gN[i] += numPts[i][j];
        } //end j
    }     //end i

    //List of weights for each point in each dimension
    realtype* wts[3];
    for (int i = 0; i < 3; ++i) {
        wts[i] = new realtype[(gN[i])];
    } //end i

    for (int i = 0, currPt = 0; i < numZones[0]; ++i) {
        realtype sum = 0.0f;
        for (int l = 0; l < numZones[2]; ++l) {
            for (int k = 0; k < numZones[1]; ++k) {
                sum += (costs[l][k][i] * ((realtype)(numPts[1][k] * numPts[2][l])));
            } //end k
        }     //end l
        for (int j = 0; j < numPts[0][i]; ++j, ++currPt) {
            wts[0][currPt] = sum / ((realtype)(gN[1] * gN[2]));
        } //end j
    }     //end i

    for (int i = 0, currPt = 0; i < numZones[1]; ++i) {
        realtype sum = 0.0f;
        for (int l = 0; l < numZones[2]; ++l) {
            for (int k = 0; k < numZones[0]; ++k) {
                sum += (costs[l][i][k] * ((realtype)(numPts[0][k] * numPts[2][l])));
            } //end k
        }     //end l
        for (int j = 0; j < numPts[1][i]; ++j, ++currPt) {
            wts[1][currPt] = sum / ((realtype)(gN[0] * gN[2]));
        } //end j
    }     //end i

    for (int i = 0, currPt = 0; i < numZones[2]; ++i) {
        realtype sum = 0.0f;
        for (int l = 0; l < numZones[1]; ++l) {
            for (int k = 0; k < numZones[0]; ++k) {
                sum += (costs[i][l][k] * ((realtype)(numPts[0][k] * numPts[1][l])));
            } //end k
        }     //end l
        for (int j = 0; j < numPts[2][i]; ++j, ++currPt) {
            wts[2][currPt] = sum / ((realtype)(gN[0] * gN[1]));
        } //end j
    }     //end i

    //Maximum stencil width for each zone in each dimension
    int* maxWidth[3][2];
    for (int d2 = 0; d2 < 3; ++d2) {
        for (int d1 = 0; d1 < 2; ++d1) {
            maxWidth[d2][d1] = new int[numZones[d2]];
        } //end d1
    }     //end d2

    for (int d1 = 0; d1 < 2; ++d1) {
        for (int i = 0; i < numZones[0]; ++i) {
            for (int k = 0; k < numZones[2]; ++k) {
                for (int j = 0; j < numZones[1]; ++j) {
                    if ((k == 0) && (j == 0)) {
                        maxWidth[0][d1][i] = sw[0][d1][k][j][i];
                    } else {
                        if (sw[0][d1][k][j][i] > maxWidth[0][d1][i]) {
                            maxWidth[0][d1][i] = sw[0][d1][k][j][i];
                        }
                    }
                } //end j
            }     //end k
        }         //end i
    }             //end d1

    for (int d1 = 0; d1 < 2; ++d1) {
        for (int i = 0; i < numZones[1]; ++i) {
            for (int k = 0; k < numZones[2]; ++k) {
                for (int j = 0; j < numZones[0]; ++j) {
                    if ((k == 0) && (j == 0)) {
                        maxWidth[1][d1][i] = sw[1][d1][k][i][j];
                    } else {
                        if (sw[1][d1][k][i][j] > maxWidth[1][d1][i]) {
                            maxWidth[1][d1][i] = sw[1][d1][k][i][j];
                        }
                    }
                } //end j
            }     //end k
        }         //end i
    }             //end d1

    for (int d1 = 0; d1 < 2; ++d1) {
        for (int i = 0; i < numZones[2]; ++i) {
            for (int k = 0; k < numZones[1]; ++k) {
                for (int j = 0; j < numZones[0]; ++j) {
                    if ((k == 0) && (j == 0)) {
                        maxWidth[2][d1][i] = sw[2][d1][i][k][j];
                    } else {
                        if (sw[2][d1][i][k][j] > maxWidth[2][d1][i]) {
                            maxWidth[2][d1][i] = sw[2][d1][i][k][j];
                        }
                    }
                } //end j
            }     //end k
        }         //end i
    }             //end d1

    //Number of points on each rank in each dimension
    for (int i = 0; i < 3; ++i) {
        counts[i] = new int[(dims[i])];
    } //end i

    for (int i = 0; i < 3; ++i) {
        computeWeightedPartition1D(counts[i], gN[i], wts[i], dims[i]);

        bool result = checkPart1D(dims[i], counts[i], numZones[i], numPts[i], maxWidth[i][0], maxWidth[i][1]);

        if (result == false) {
            std::vector<int> superPtMap;
            std::vector<realtype> tmpWts;

            createSuperPtsForWpart1D(superPtMap, tmpWts, gN[i], wts[i], numZones[i], numPts[i], maxWidth[i][0],
                                     maxWidth[i][1]);

            computeWeightedPartition1D(counts[i], superPtMap.size(), &(tmpWts[0]), dims[i]);

            splitSuperPtsForWpart1D(counts[i], superPtMap, dims[i]);
        } //end if failed checkPart1D
    }     //end i

    //TEMPORARY HACK TO OVERCOME A LIMITATION IN CURRENT DE-REMIG IMPLEMENTATION.
    //SHOULD BE REMOVED SOON AFTER THE DE-REMIG FIX AND DEFINITELY BEFORE MULTI-PHYSICS.
    if (hack_first_interior_z_rank_max_nz > 0) {
        int nbdyTop = 0;
        if (numZones[2] == 3) {
            nbdyTop = numPts[2][0];
        } //end if has Top Boundary
        int sum = 0;
        int firstInteriorZrank = 0;
        while (firstInteriorZrank < dims[2]) {
            sum += counts[2][firstInteriorZrank];
            if (sum > nbdyTop) {
                break;
            } else {
                ++firstInteriorZrank;
            }
        }
        EMSL_VERIFY(firstInteriorZrank < dims[2]);
        if (counts[2][firstInteriorZrank] > hack_first_interior_z_rank_max_nz) {
            fprintf(
                stderr,
                "WARNING: ADJUSTING NZ ON FIRST INTERIOR Z RANK ( %d ) world rank(%d) TO BE: %d INITIALLY WAS: %d \n",
                firstInteriorZrank, world_rank, hack_first_interior_z_rank_max_nz, counts[2][firstInteriorZrank]);
            int numExtra = counts[2][firstInteriorZrank] - hack_first_interior_z_rank_max_nz;
            EMSL_VERIFY(firstInteriorZrank < (dims[2] - 1));
            counts[2][firstInteriorZrank] = hack_first_interior_z_rank_max_nz;

            int nrank_z_2share = dims[2] - 1 - firstInteriorZrank;
            int nExtra_per_rank = numExtra / nrank_z_2share;
            int nLeftover = numExtra - nExtra_per_rank * nrank_z_2share;

            int z_rank_count = 0;
            for (int iz_rank = (firstInteriorZrank + 1); iz_rank < dims[2]; iz_rank++) {
                counts[2][iz_rank] += nExtra_per_rank;
                if (z_rank_count < nLeftover)
                    counts[2][iz_rank] += 1;
                z_rank_count++;
            }
        }
    } //END HACK

    //Offsets for each rank in each dimension
    for (int i = 0; i < 3; ++i) {
        offsets[i] = new int[(dims[i])];
        offsets[i][0] = 0;
        for (int j = 1; j < dims[i]; ++j) {
            offsets[i][j] = offsets[i][j - 1] + counts[i][j - 1];
        } //end j
    }     //end i

    //Offsets for first point on this rank
    for (int j = 0; j < 3; ++j) {
        off[j] = offsets[j][coords[j]];
    } //end j

    //Total number of points on this rank in each dimension
    for (int j = 0; j < 3; ++j) {
        numLocal[j] = counts[j][coords[j]];
    } //end j

    //Number of local points in each zone in each dimension
    for (int i = 0; i < 3; ++i) {
        int sum = 0;
        for (int j = 0; j < numZones[i]; ++j) {
            int currZoneEnd = zoneOff[i][j] + numPts[i][j];
            if (currZoneEnd <= (off[i] + sum)) {
                numLocalSplit[i][j] = 0;
            } else {
                numLocalSplit[i][j] = std::min((currZoneEnd - (off[i] + sum)), (numLocal[i] - sum));
            }
            sum += numLocalSplit[i][j];
        } //end j
    }     //end i

    //Number of points on the domains adjacent to this rank
    for (int i = 0; i < 3; ++i) {
        //Negative direction
        if (coords[i] > 0) {
            for (int j = 0; j < numZones[i]; ++j) {
                if ((zoneOff[i][j] + numPts[i][j]) >= off[i]) {
                    numNeighborPts[i][0] = std::min((off[i] - zoneOff[i][j]), counts[i][coords[i] - 1]);
                    neighborSubDom[i][0] = j;
                    break;
                }
            } //end j
        } else {
            numNeighborPts[i][0] = 0;
            neighborSubDom[i][0] = -1;
        } //end if first rank

        //Positive direction
        if (coords[i] < (dims[i] - 1)) {
            for (int j = (numZones[i] - 1); j >= 0; --j) {
                if (zoneOff[i][j] <= (off[i] + numLocal[i])) {
                    int currZoneEnd = zoneOff[i][j] + numPts[i][j];
                    numNeighborPts[i][1] = std::min((currZoneEnd - (off[i] + numLocal[i])), counts[i][coords[i] + 1]);
                    neighborSubDom[i][1] = j;
                    break;
                }
            } //end j
        } else {
            numNeighborPts[i][1] = 0;
            neighborSubDom[i][1] = -1;
        } //end if last rank
    }     //end i

    for (int d2 = 0; d2 < 3; ++d2) {
        for (int d1 = 0; d1 < 2; ++d1) {
            delete[](maxWidth[d2][d1]);
        } //end d1
    }     //end d2

    for (int i = 0; i < 3; ++i) {
        delete[](wts[i]);
    } //end i
} //compute_partition

void
decompositionManager3D ::create_subdomains()
{
    //Number of sub-domains in each dimension
    for (int j = 0; j < 3; ++j) {
        numLocalSubDom[j] = 0;
        for (int i = 0; i < numZones[j]; ++i) {
            if (numLocalSplit[j][i] > 0) {
                ++(numLocalSubDom[j]);
            }
        } //end i
    }     //end j

    totalNumLocalSubDom = numLocalSubDom[0] * numLocalSubDom[1] * numLocalSubDom[2];

    for (int i = 0; i < 3; ++i) {
        subDomLocalToGlobalMap[i] = new int[totalNumLocalSubDom];
        splitSubDomLocalToGlobalMap[i] = new int[numLocalSubDom[i]];
    } //end i

    subDomGlobalToLocalMap = new Int2Ptr[numZones[2]];
    for (int k = 0; k < numZones[2]; ++k) {
        subDomGlobalToLocalMap[k] = new IntPtr[numZones[1]];
        for (int j = 0; j < numZones[1]; ++j) {
            subDomGlobalToLocalMap[k][j] = new int[numZones[0]];
        } //end j
    }     //end k

    int subDomCnt = 0;
    for (int k = 0; k < numZones[2]; ++k) {
        for (int j = 0; j < numZones[1]; ++j) {
            for (int i = 0; i < numZones[0]; ++i) {
                if ((numLocalSplit[2][k] > 0) && (numLocalSplit[1][j] > 0) && (numLocalSplit[0][i] > 0)) {
                    subDomGlobalToLocalMap[k][j][i] = subDomCnt;
                    subDomLocalToGlobalMap[0][subDomCnt] = i;
                    subDomLocalToGlobalMap[1][subDomCnt] = j;
                    subDomLocalToGlobalMap[2][subDomCnt] = k;
                    ++subDomCnt;
                } else {
                    subDomGlobalToLocalMap[k][j][i] = -1;
                }
            } //end i
        }     //end j
    }         //end k

    EMSL_VERIFY(subDomCnt == totalNumLocalSubDom);

    for (int d = 0; d < 3; ++d) {
        int splitSubDomCnt = 0;
        for (int i = 0; i < numZones[d]; ++i) {
            if (numLocalSplit[d][i] > 0) {
                splitSubDomLocalToGlobalMap[d][splitSubDomCnt] = i;
                ++splitSubDomCnt;
            }
        } //end i
        EMSL_VERIFY(splitSubDomCnt == numLocalSubDom[d]);
    } //end d

    //Offset for each sub-domain in each dimension
    for (int d = 0; d < 3; ++d) {
        splitOff[d] = new int[numLocalSubDom[d]];
        splitOff[d][0] = off[d];
        for (int i = 1; i < numLocalSubDom[d]; ++i) {
            splitOff[d][i] = splitOff[d][i - 1] + getNumPtsSplit(i - 1, d);
        } //end i
    }     //end d

    //Compute global sub-domain id offsets
    for (int i = 0; i < 3; ++i) {
        splitGlobalSubDomIdOff[i] = 0;
        int prevLastZone = 0;
        int currOff = 0;
        for (int j = 0; j < coords[i]; ++j) {
            int currFirstZone = -1;
            for (int k = prevLastZone; k < numZones[i]; ++k) {
                if ((k == (numZones[i] - 1)) || ((zoneOff[i][k] <= currOff) && (zoneOff[i][k + 1] > currOff))) {
                    currFirstZone = k;
                    break;
                }
            } //end k
            EMSL_VERIFY(currFirstZone >= 0);
            int currLastZone = -1;
            int currEnd = currOff + counts[i][j] - 1;
            for (int k = currFirstZone; k < numZones[i]; ++k) {
                if ((k == (numZones[i] - 1)) || ((zoneOff[i][k] <= currEnd) && (zoneOff[i][k + 1] > currEnd))) {
                    currLastZone = k;
                    break;
                }
            } //end k
            EMSL_VERIFY(currLastZone >= currFirstZone);
            splitGlobalSubDomIdOff[i] += (currLastZone - currFirstZone + 1);
            currOff += counts[i][j];
            prevLastZone = currLastZone;
        } //end j
    }     //end i
}

void
decompositionManager3D ::getLocalNeighborID(int id, int localNhId[_NH_ID_3D_TOTAL])
{
    //Set non-existing neigbors to -1
    for (int i = 0; i < _NH_ID_3D_TOTAL; ++i) {
        localNhId[i] = -1;
    } //end i

    int splitSubDomId[3];
    getSplitLocalSubDomID(id, splitSubDomId);

    if (splitSubDomId[2] > 0) {
        if (splitSubDomId[1] > 0) {
            if (splitSubDomId[0] > 0) {
                localNhId[_NH_ID_3D_NEGX_NEGY_NEGZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1] - 1), (splitSubDomId[2] - 1));
            }

            localNhId[_NH_ID_3D_CENX_NEGY_NEGZ] =
                getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1] - 1), (splitSubDomId[2] - 1));

            if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
                localNhId[_NH_ID_3D_POSX_NEGY_NEGZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1] - 1), (splitSubDomId[2] - 1));
            }
        }

        if (splitSubDomId[0] > 0) {
            localNhId[_NH_ID_3D_NEGX_CENY_NEGZ] =
                getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1]), (splitSubDomId[2] - 1));
        }

        localNhId[_NH_ID_3D_CENX_CENY_NEGZ] =
            getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1]), (splitSubDomId[2] - 1));

        if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
            localNhId[_NH_ID_3D_POSX_CENY_NEGZ] =
                getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1]), (splitSubDomId[2] - 1));
        }

        if (splitSubDomId[1] < (numLocalSubDom[1] - 1)) {
            if (splitSubDomId[0] > 0) {
                localNhId[_NH_ID_3D_NEGX_POSY_NEGZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1] + 1), (splitSubDomId[2] - 1));
            }

            localNhId[_NH_ID_3D_CENX_POSY_NEGZ] =
                getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1] + 1), (splitSubDomId[2] - 1));

            if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
                localNhId[_NH_ID_3D_POSX_POSY_NEGZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1] + 1), (splitSubDomId[2] - 1));
            }
        }
    }

    if (splitSubDomId[1] > 0) {
        if (splitSubDomId[0] > 0) {
            localNhId[_NH_ID_3D_NEGX_NEGY_CENZ] =
                getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1] - 1), (splitSubDomId[2]));
        }

        localNhId[_NH_ID_3D_CENX_NEGY_CENZ] =
            getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1] - 1), (splitSubDomId[2]));

        if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
            localNhId[_NH_ID_3D_POSX_NEGY_CENZ] =
                getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1] - 1), (splitSubDomId[2]));
        }
    }

    if (splitSubDomId[0] > 0) {
        localNhId[_NH_ID_3D_NEGX_CENY_CENZ] =
            getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1]), (splitSubDomId[2]));
    }

    if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
        localNhId[_NH_ID_3D_POSX_CENY_CENZ] =
            getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1]), (splitSubDomId[2]));
    }

    if (splitSubDomId[1] < (numLocalSubDom[1] - 1)) {
        if (splitSubDomId[0] > 0) {
            localNhId[_NH_ID_3D_NEGX_POSY_CENZ] =
                getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1] + 1), (splitSubDomId[2]));
        }

        localNhId[_NH_ID_3D_CENX_POSY_CENZ] =
            getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1] + 1), (splitSubDomId[2]));

        if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
            localNhId[_NH_ID_3D_POSX_POSY_CENZ] =
                getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1] + 1), (splitSubDomId[2]));
        }
    }

    if (splitSubDomId[2] < (numLocalSubDom[2] - 1)) {
        if (splitSubDomId[1] > 0) {
            if (splitSubDomId[0] > 0) {
                localNhId[_NH_ID_3D_NEGX_NEGY_POSZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1] - 1), (splitSubDomId[2] + 1));
            }

            localNhId[_NH_ID_3D_CENX_NEGY_POSZ] =
                getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1] - 1), (splitSubDomId[2] + 1));

            if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
                localNhId[_NH_ID_3D_POSX_NEGY_POSZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1] - 1), (splitSubDomId[2] + 1));
            }
        }

        if (splitSubDomId[0] > 0) {
            localNhId[_NH_ID_3D_NEGX_CENY_POSZ] =
                getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1]), (splitSubDomId[2] + 1));
        }

        localNhId[_NH_ID_3D_CENX_CENY_POSZ] =
            getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1]), (splitSubDomId[2] + 1));

        if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
            localNhId[_NH_ID_3D_POSX_CENY_POSZ] =
                getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1]), (splitSubDomId[2] + 1));
        }

        if (splitSubDomId[1] < (numLocalSubDom[1] - 1)) {
            if (splitSubDomId[0] > 0) {
                localNhId[_NH_ID_3D_NEGX_POSY_POSZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] - 1), (splitSubDomId[1] + 1), (splitSubDomId[2] + 1));
            }

            localNhId[_NH_ID_3D_CENX_POSY_POSZ] =
                getSubDomIDfromSplit((splitSubDomId[0]), (splitSubDomId[1] + 1), (splitSubDomId[2] + 1));

            if (splitSubDomId[0] < (numLocalSubDom[0] - 1)) {
                localNhId[_NH_ID_3D_POSX_POSY_POSZ] =
                    getSubDomIDfromSplit((splitSubDomId[0] + 1), (splitSubDomId[1] + 1), (splitSubDomId[2] + 1));
            }
        }
    }
}

void
decompositionManager3D ::compute_neighbors()
{
    //Set non-existing neigbors to -1
    for (int i = 0; i < _NH_ID_3D_TOTAL; ++i) {
        neighbor[i] = -1;
    } //end i

    numNeighbors = 0;

    int tmpCoords[3];
    tmpCoords[0] = coords[0];
    tmpCoords[1] = coords[1];

    if (coords[2] > 0) {
        tmpCoords[2] = coords[2] - 1;
        MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_CENY_NEGZ]));
        ++numNeighbors;
    }

    if (coords[2] < (dims[2] - 1)) {
        tmpCoords[2] = coords[2] + 1;
        MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_CENY_POSZ]));
        ++numNeighbors;
    }

    if (coords[1] > 0) {
        tmpCoords[1] = coords[1] - 1;
        tmpCoords[2] = coords[2];
        MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_NEGY_CENZ]));
        ++numNeighbors;

        if (coords[2] > 0) {
            tmpCoords[2] = coords[2] - 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_NEGY_NEGZ]));
            ++numNeighbors;
        }

        if (coords[2] < (dims[2] - 1)) {
            tmpCoords[2] = coords[2] + 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_NEGY_POSZ]));
            ++numNeighbors;
        }
    }

    if (coords[1] < (dims[1] - 1)) {
        tmpCoords[1] = coords[1] + 1;
        tmpCoords[2] = coords[2];
        MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_POSY_CENZ]));
        ++numNeighbors;

        if (coords[2] > 0) {
            tmpCoords[2] = coords[2] - 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_POSY_NEGZ]));
            ++numNeighbors;
        }

        if (coords[2] < (dims[2] - 1)) {
            tmpCoords[2] = coords[2] + 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_CENX_POSY_POSZ]));
            ++numNeighbors;
        }
    }

    if (coords[0] > 0) {
        tmpCoords[0] = coords[0] - 1;
        tmpCoords[1] = coords[1];
        tmpCoords[2] = coords[2];
        MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_CENY_CENZ]));
        ++numNeighbors;

        if (coords[2] > 0) {
            tmpCoords[2] = coords[2] - 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_CENY_NEGZ]));
            ++numNeighbors;
        }

        if (coords[2] < (dims[2] - 1)) {
            tmpCoords[2] = coords[2] + 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_CENY_POSZ]));
            ++numNeighbors;
        }

        if (coords[1] > 0) {
            tmpCoords[1] = coords[1] - 1;
            tmpCoords[2] = coords[2];
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_NEGY_CENZ]));
            ++numNeighbors;

            if (coords[2] > 0) {
                tmpCoords[2] = coords[2] - 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_NEGY_NEGZ]));
                ++numNeighbors;
            }

            if (coords[2] < (dims[2] - 1)) {
                tmpCoords[2] = coords[2] + 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_NEGY_POSZ]));
                ++numNeighbors;
            }
        }

        if (coords[1] < (dims[1] - 1)) {
            tmpCoords[1] = coords[1] + 1;
            tmpCoords[2] = coords[2];
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_POSY_CENZ]));
            ++numNeighbors;

            if (coords[2] > 0) {
                tmpCoords[2] = coords[2] - 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_POSY_NEGZ]));
                ++numNeighbors;
            }

            if (coords[2] < (dims[2] - 1)) {
                tmpCoords[2] = coords[2] + 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_NEGX_POSY_POSZ]));
                ++numNeighbors;
            }
        }
    }

    if (coords[0] < (dims[0] - 1)) {
        tmpCoords[0] = coords[0] + 1;
        tmpCoords[1] = coords[1];
        tmpCoords[2] = coords[2];
        MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_CENY_CENZ]));
        ++numNeighbors;

        if (coords[2] > 0) {
            tmpCoords[2] = coords[2] - 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_CENY_NEGZ]));
            ++numNeighbors;
        }

        if (coords[2] < (dims[2] - 1)) {
            tmpCoords[2] = coords[2] + 1;
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_CENY_POSZ]));
            ++numNeighbors;
        }

        if (coords[1] > 0) {
            tmpCoords[1] = coords[1] - 1;
            tmpCoords[2] = coords[2];
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_NEGY_CENZ]));
            ++numNeighbors;

            if (coords[2] > 0) {
                tmpCoords[2] = coords[2] - 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_NEGY_NEGZ]));
                ++numNeighbors;
            }

            if (coords[2] < (dims[2] - 1)) {
                tmpCoords[2] = coords[2] + 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_NEGY_POSZ]));
                ++numNeighbors;
            }
        }

        if (coords[1] < (dims[1] - 1)) {
            tmpCoords[1] = coords[1] + 1;
            tmpCoords[2] = coords[2];
            MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_POSY_CENZ]));
            ++numNeighbors;

            if (coords[2] > 0) {
                tmpCoords[2] = coords[2] - 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_POSY_NEGZ]));
                ++numNeighbors;
            }

            if (coords[2] < (dims[2] - 1)) {
                tmpCoords[2] = coords[2] + 1;
                MPI_Cart_rank(cart_comm, tmpCoords, &(neighbor[_NH_ID_3D_POSX_POSY_POSZ]));
                ++numNeighbors;
            }
        }
    }
}

void
decompositionManager3D ::print(FILE* fd)
{
    if (world_rank == 0) {
        fprintf(fd, "------ begin 3D domain decompostion info ------\n");
        fprintf(fd, "npes=%d, npe_x=%d, npe_y=%d, npe_z=%d\n", npes, dims[0], dims[1], dims[2]);
        fprintf(fd, "gN_x=%d, gN_y=%d, gN_z=%d\n", gN[0], gN[1], gN[2]);
        fprintf(fd, "numZones_x=%d, numZones_y=%d, numZones_z=%d\n", numZones[0], numZones[1], numZones[2]);
    }

    char token = 1;
    if (world_rank > 0) {
        MPI_Recv(&token, 1, MPI_CHAR, (world_rank - 1), 1, comm_world, MPI_STATUS_IGNORE);
    }

    fprintf(fd, "decomp(%d): cart_rank = %d, coords (X, Y, Z) in processor grid = %d, %d, %d\n", world_rank, cart_rank,
            coords[0], coords[1], coords[2]);
    fprintf(fd, "decomp(%d): totalNumSubDom = %d, numSubDom in each dimension (X, Y, Z) = %d, %d, %d\n", world_rank,
            totalNumLocalSubDom, numLocalSubDom[0], numLocalSubDom[1], numLocalSubDom[2]);
    fprintf(fd, "decomp(%d): Offset for first local subdomain (X, Y, Z) = (%d, %d, %d)\n", world_rank,
            splitGlobalSubDomIdOff[0], splitGlobalSubDomIdOff[1], splitGlobalSubDomIdOff[2]);
    fprintf(fd, "------ begin local subdomain to zone map along each dimension on rank %d ----------\n", world_rank);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < numLocalSubDom[i]; ++j) {
            fprintf(fd, "decomp(%d): subdomain[%s][%d] = %d\n", world_rank, _DIM_STR_3D[i], j,
                    splitSubDomLocalToGlobalMap[i][j]);
        } //end j
    }     //end i
    fprintf(fd, "------ end local subdomain to zone map along each dimension on rank %d ----------\n", world_rank);
    fprintf(fd, "decomp(%d): Offset for first local point (X, Y, Z) = (%d, %d, %d)\n", world_rank, off[0], off[1],
            off[2]);
    fprintf(fd, "------ begin number of points in each zone along each dimension on rank %d ----------\n", world_rank);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < numZones[i]; ++j) {
            fprintf(fd, "decomp(%d): numPoints[%s][%d] = %d\n", world_rank, _DIM_STR_3D[i], j, numLocalSplit[i][j]);
        } //end j
    }     //end i
    fprintf(fd, "------ end number of points in each zone along each dimension on rank %d ----------\n", world_rank);
    fprintf(fd, "------ begin neighbors (Ranks in Cart_Comm) for rank %d ----------\n", world_rank);
    fprintf(fd, "decomp(%d): numNeighbors = %d\n", world_rank, numNeighbors);
    for (int i = 0; i < _NH_ID_3D_TOTAL; ++i) {
        if (neighbor[i] >= 0) {
            fprintf(fd, "decomp(%d): neighbor[%s] = %d\n", world_rank, _NH_ID_STR_3D[i], neighbor[i]);
        }
    } //end i
    fprintf(fd, "------ end neighbors for rank %d ----------\n", world_rank);
    fprintf(fd, "------ begin neighboring subdomain info for rank %d ----------\n", world_rank);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (neighborSubDom[i][j] >= 0) {
                fprintf(fd, "decomp(%d): neighbor[%s][%s]: zone = %d, numPoints = %d\n", world_rank, _DIM_STR_3D[i],
                        _NH_DIR_STR[j], neighborSubDom[i][j], numNeighborPts[i][j]);
            }
        } //end j
    }     //end i
    fprintf(fd, "------ end neighboring subdomain info for rank %d ----------\n", world_rank);
    fprintf(fd, "\n");
    fflush(fd);

    if (world_rank < (npes - 1)) {
        MPI_Send(&token, 1, MPI_CHAR, (world_rank + 1), 1, comm_world);
    }

    if (world_rank == (npes - 1)) {
        fprintf(fd, "------ end 3D domain decompostion info ------\n");
    }
}

int
decompositionManager3D ::getSubDomIDtoSend(NH_ID_3D nh, RECV_TYPE_3D rType, int othIdX, int othIdY, int othIdZ)
{
    int idX = -1;
    int idY = -1;
    int idZ = -1;

    switch (rType) {
        case _RECV_TYPE_3D_NEGX: {
            EMSL_VERIFY(nh == _NH_ID_3D_POSX_CENY_CENZ);
            idX = numLocalSubDom[0] - 1;
            idY = othIdY;
            idZ = othIdZ;
            break;
        }
        case _RECV_TYPE_3D_POSX: {
            EMSL_VERIFY(nh == _NH_ID_3D_NEGX_CENY_CENZ);
            idX = 0;
            idY = othIdY;
            idZ = othIdZ;
            break;
        }
        case _RECV_TYPE_3D_NEGY: {
            EMSL_VERIFY(nh == _NH_ID_3D_CENX_POSY_CENZ);
            idX = othIdX;
            idY = numLocalSubDom[1] - 1;
            idZ = othIdZ;
            break;
        }
        case _RECV_TYPE_3D_POSY1: {
            EMSL_VERIFY(nh == _NH_ID_3D_CENX_NEGY_CENZ);
            idX = othIdX;
            idY = 0;
            idZ = othIdZ;
            break;
        }
        case _RECV_TYPE_3D_POSY2: {
            EMSL_VERIFY(nh == _NH_ID_3D_CENX_NEGY_CENZ);
            idX = othIdX;
            idY = 0;
            idZ = othIdZ;
            break;
        }
        case _RECV_TYPE_3D_NEGZ: {
            EMSL_VERIFY(nh == _NH_ID_3D_CENX_CENY_POSZ);
            idX = othIdX;
            idY = othIdY;
            idZ = numLocalSubDom[2] - 1;
            break;
        }
        case _RECV_TYPE_3D_POSZ: {
            EMSL_VERIFY(nh == _NH_ID_3D_CENX_CENY_NEGZ);
            idX = othIdX;
            idY = othIdY;
            idZ = 0;
            break;
        }
        case _RECV_TYPE_3D_NEGY_NEGZ: {
            idX = othIdX;
            if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idY = othIdY - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idY >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_CENX_POSY_POSZ);
                idY = numLocalSubDom[1] - 1;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGY_POSZ: {
            idX = othIdX;
            if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idY = othIdY - 1;
                idZ = 0;
                EMSL_VERIFY(idY >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_CENX_POSY_NEGZ);
                idY = numLocalSubDom[1] - 1;
                idZ = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_POSY_NEGZ: {
            idX = othIdX;
            if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idY = 0;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idY = othIdY + 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_CENX_NEGY_POSZ);
                idY = 0;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_POSY_POSZ: {
            idX = othIdX;
            if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idY = 0;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idY = othIdY + 1;
                idZ = 0;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_CENX_NEGY_NEGZ);
                idY = 0;
                idZ = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_NEGY: {
            idZ = othIdZ;
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY - 1;
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idX = othIdX - 1;
                idY = numLocalSubDom[1] - 1;
                EMSL_VERIFY(idX >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_POSY_CENZ);
                idX = numLocalSubDom[0] - 1;
                idY = numLocalSubDom[1] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_POSY1: {
            idZ = othIdZ;
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY + 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX - 1;
                idY = 0;
                EMSL_VERIFY(idX >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_NEGY_CENZ);
                idX = numLocalSubDom[0] - 1;
                idY = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_POSY2: {
            idZ = othIdZ;
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY + 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX - 1;
                idY = 0;
                EMSL_VERIFY(idX >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_NEGY_CENZ);
                idX = numLocalSubDom[0] - 1;
                idY = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_NEGY: {
            idZ = othIdZ;
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idY = othIdY - 1;
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idX = othIdX + 1;
                idY = numLocalSubDom[1] - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_POSY_CENZ);
                idX = 0;
                idY = numLocalSubDom[1] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_POSY1: {
            idZ = othIdZ;
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idY = othIdY + 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX + 1;
                idY = 0;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_NEGY_CENZ);
                idX = 0;
                idY = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_POSY2: {
            idZ = othIdZ;
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idY = othIdY + 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX + 1;
                idY = 0;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_NEGY_CENZ);
                idX = 0;
                idY = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_NEGZ: {
            idY = othIdY;
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idX = othIdX - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_CENY_POSZ);
                idX = numLocalSubDom[0] - 1;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_POSZ: {
            idY = othIdY;
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idX = othIdX - 1;
                idZ = 0;
                EMSL_VERIFY(idX >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_CENY_NEGZ);
                idX = numLocalSubDom[0] - 1;
                idZ = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_NEGZ: {
            idY = othIdY;
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idX = othIdX + 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_CENY_POSZ);
                idX = 0;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_POSZ: {
            idY = othIdY;
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idX = othIdX + 1;
                idZ = 0;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_CENY_NEGZ);
                idX = 0;
                idZ = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_NEGY_NEGZ: {
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idY >= 0);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idX = othIdX - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idX = othIdX - 1;
                idY = othIdY - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_POSZ) {
                idX = othIdX - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX >= 0);
            } else if (nh == _NH_ID_3D_POSX_CENY_POSZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_POSX_POSY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_POSY_POSZ);
                idX = numLocalSubDom[0] - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_NEGY_POSZ: {
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idY >= 0);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idX = othIdX - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idX = othIdX - 1;
                idY = othIdY - 1;
                idZ = 0;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_NEGZ) {
                idX = othIdX - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = 0;
                EMSL_VERIFY(idX >= 0);
            } else if (nh == _NH_ID_3D_POSX_CENY_NEGZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY - 1;
                idZ = 0;
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_POSX_POSY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_POSY_NEGZ);
                idX = numLocalSubDom[0] - 1;
                idY = numLocalSubDom[1] - 1;
                idZ = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_POSY_NEGZ: {
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY + 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX - 1;
                idY = 0;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idX = othIdX - 1;
                idY = othIdY + 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_POSZ) {
                idX = othIdX - 1;
                idY = 0;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX >= 0);
            } else if (nh == _NH_ID_3D_POSX_CENY_POSZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY + 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_POSX_NEGY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = 0;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_NEGY_POSZ);
                idX = numLocalSubDom[0] - 1;
                idY = 0;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_NEGX_POSY_POSZ: {
            if (nh == _NH_ID_3D_POSX_CENY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY + 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX - 1;
                idY = 0;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idX = othIdX - 1;
                idY = othIdY + 1;
                idZ = 0;
                EMSL_VERIFY(idX >= 0);
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_NEGZ) {
                idX = othIdX - 1;
                idY = 0;
                idZ = 0;
                EMSL_VERIFY(idX >= 0);
            } else if (nh == _NH_ID_3D_POSX_CENY_NEGZ) {
                idX = numLocalSubDom[0] - 1;
                idY = othIdY + 1;
                idZ = 0;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_POSX_NEGY_CENZ) {
                idX = numLocalSubDom[0] - 1;
                idY = 0;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_POSX_NEGY_NEGZ);
                idX = numLocalSubDom[0] - 1;
                idY = 0;
                idZ = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_NEGY_NEGZ: {
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idY = othIdY - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idY >= 0);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idX = othIdX + 1;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idX = othIdX + 1;
                idY = othIdY - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_POSZ) {
                idX = othIdX + 1;
                idY = numLocalSubDom[1] - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else if (nh == _NH_ID_3D_NEGX_CENY_POSZ) {
                idX = 0;
                idY = othIdY - 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_NEGX_POSY_CENZ) {
                idX = 0;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_POSY_POSZ);
                idX = 0;
                idY = numLocalSubDom[1] - 1;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_NEGY_POSZ: {
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idY = othIdY - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idY >= 0);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_POSY_CENZ) {
                idX = othIdX + 1;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idX = othIdX + 1;
                idY = othIdY - 1;
                idZ = 0;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_CENX_POSY_NEGZ) {
                idX = othIdX + 1;
                idY = numLocalSubDom[1] - 1;
                idZ = 0;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else if (nh == _NH_ID_3D_NEGX_CENY_NEGZ) {
                idX = 0;
                idY = othIdY - 1;
                idZ = 0;
                EMSL_VERIFY(idY >= 0);
            } else if (nh == _NH_ID_3D_NEGX_POSY_CENZ) {
                idX = 0;
                idY = numLocalSubDom[1] - 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_POSY_NEGZ);
                idX = 0;
                idY = numLocalSubDom[1] - 1;
                idZ = 0;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_POSY_NEGZ: {
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idY = othIdY + 1;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX + 1;
                idY = 0;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idZ >= 0);
            } else if (nh == _NH_ID_3D_CENX_CENY_POSZ) {
                idX = othIdX + 1;
                idY = othIdY + 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_POSZ) {
                idX = othIdX + 1;
                idY = 0;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else if (nh == _NH_ID_3D_NEGX_CENY_POSZ) {
                idX = 0;
                idY = othIdY + 1;
                idZ = numLocalSubDom[2] - 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_NEGX_NEGY_CENZ) {
                idX = 0;
                idY = 0;
                idZ = othIdZ - 1;
                EMSL_VERIFY(idZ >= 0);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_NEGY_POSZ);
                idX = 0;
                idY = 0;
                idZ = numLocalSubDom[2] - 1;
            }
            break;
        }
        case _RECV_TYPE_3D_POSX_POSY_POSZ: {
            if (nh == _NH_ID_3D_NEGX_CENY_CENZ) {
                idX = 0;
                idY = othIdY + 1;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_CENZ) {
                idX = othIdX + 1;
                idY = 0;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else if (nh == _NH_ID_3D_CENX_CENY_NEGZ) {
                idX = othIdX + 1;
                idY = othIdY + 1;
                idZ = 0;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_CENX_NEGY_NEGZ) {
                idX = othIdX + 1;
                idY = 0;
                idZ = 0;
                EMSL_VERIFY(idX < numLocalSubDom[0]);
            } else if (nh == _NH_ID_3D_NEGX_CENY_NEGZ) {
                idX = 0;
                idY = othIdY + 1;
                idZ = 0;
                EMSL_VERIFY(idY < numLocalSubDom[1]);
            } else if (nh == _NH_ID_3D_NEGX_NEGY_CENZ) {
                idX = 0;
                idY = 0;
                idZ = othIdZ + 1;
                EMSL_VERIFY(idZ < numLocalSubDom[2]);
            } else {
                EMSL_VERIFY(nh == _NH_ID_3D_NEGX_NEGY_NEGZ);
                idX = 0;
                idY = 0;
                idZ = 0;
            }
            break;
        }
        default: {
            //Should not come here!
            EMSL_VERIFY(false);
        }
    } //end switch

    return getSubDomIDfromSplit(idX, idY, idZ);
}

MPI_Comm
decompositionManager3D ::createSubComm(const int activeRange[3][2], const int groupRange[3][2])
{
    MPI_Group subGroup;

    if ((coords[0] >= activeRange[0][0]) && (coords[0] <= activeRange[0][1]) && (coords[1] >= activeRange[1][0]) &&
        (coords[1] <= activeRange[1][1]) && (coords[2] >= activeRange[2][0]) && (coords[2] <= activeRange[2][1])) {
        int numX = groupRange[0][1] - groupRange[0][0] + 1;
        int numY = groupRange[1][1] - groupRange[1][0] + 1;
        int numZ = groupRange[2][1] - groupRange[2][0] + 1;

        int groupSize = numX * numY * numZ;
        int* groupRanks = new int[groupSize];

        int cnt = 0;
        int tmpCoords[3];
        for (int iz = groupRange[2][0]; iz <= groupRange[2][1]; ++iz) {
            tmpCoords[2] = iz;
            for (int iy = groupRange[1][0]; iy <= groupRange[1][1]; ++iy) {
                tmpCoords[1] = iy;
                for (int ix = groupRange[0][0]; ix <= groupRange[0][1]; ++ix) {
                    tmpCoords[0] = ix;
                    MPI_Cart_rank(cart_comm, tmpCoords, &(groupRanks[cnt]));
                    ++cnt;
                } //end ix
            }     //end iy
        }         //end iz

        MPI_Group_incl(cart_group, groupSize, groupRanks, &subGroup);

        delete[] groupRanks;
    } else {
        subGroup = MPI_GROUP_EMPTY;
    } //end if active

    MPI_Comm subComm;
    MPI_Comm_create(cart_comm, subGroup, &subComm);

    MPI_Group_free(&subGroup);

    return subComm;
}

void
decompositionManager3D ::has_neighbor(int id, bool hasNeighbor[3][2])
{
    int splitId[3];
    getSplitLocalSubDomID(id, splitId);
    for (int i = 0; i < 3; ++i) {
        if ((coords[i] == 0) && (splitId[i] == 0)) {
            hasNeighbor[i][0] = false;
        } else {
            hasNeighbor[i][0] = true;
        } //end if first rank and first sub-domain
        if ((coords[i] == (dims[i] - 1)) && (splitId[i] == (numLocalSubDom[i] - 1))) {
            hasNeighbor[i][1] = false;
        } else {
            hasNeighbor[i][1] = true;
        } //end if last rank and last sub-domain
    }     //end i
} //has_neighbor

void
decompositionManager3D ::check_grain_size(int*** sw[3][2])
{
    //Ensure halo exchange is only between adjacent domains - required for DataBlock
    for (int k = 0; k < numZones[2]; ++k) {
        for (int j = 0; j < numZones[1]; ++j) {
            for (int i = 0; i < numZones[0]; ++i) {
                if ((numLocalSplit[0][i] > 0) && (numLocalSplit[1][j] > 0) && (numLocalSplit[2][k] > 0)) {
                    //Negative-X
                    if ((i > 0) && (numLocalSplit[0][i - 1] > 0)) {
                        EMSL_VERIFY(sw[0][0][k][j][i] <= numLocalSplit[0][i - 1]);
                    } else {
                        if (numNeighborPts[0][0] > 0) {
                            EMSL_VERIFY(sw[0][0][k][j][i] <= numNeighborPts[0][0]);
                        }
                    }

                    //Positive-X
                    if ((i < (numZones[0] - 1)) && (numLocalSplit[0][i + 1] > 0)) {
                        EMSL_VERIFY(sw[0][1][k][j][i] <= numLocalSplit[0][i + 1]);
                    } else {
                        if (numNeighborPts[0][1] > 0) {
                            EMSL_VERIFY(sw[0][1][k][j][i] <= numNeighborPts[0][1]);
                        }
                    }

                    //Negative-Y
                    if ((j > 0) && (numLocalSplit[1][j - 1] > 0)) {
                        EMSL_VERIFY(sw[1][0][k][j][i] <= numLocalSplit[1][j - 1]);
                    } else {
                        if (numNeighborPts[1][0] > 0) {
                            EMSL_VERIFY(sw[1][0][k][j][i] <= numNeighborPts[1][0]);
                        }
                    }

                    //Positive-Y
                    if ((j < (numZones[1] - 1)) && (numLocalSplit[1][j + 1] > 0)) {
                        EMSL_VERIFY(sw[1][1][k][j][i] <= numLocalSplit[1][j + 1]);
                    } else {
                        if (numNeighborPts[1][1] > 0) {
                            EMSL_VERIFY(sw[1][1][k][j][i] <= numNeighborPts[1][1]);
                        }
                    }

                    //Negative-Z
                    if ((k > 0) && (numLocalSplit[2][k - 1] > 0)) {
                        EMSL_VERIFY(sw[2][0][k][j][i] <= numLocalSplit[2][k - 1]);
                    } else {
                        if (numNeighborPts[2][0] > 0) {
                            EMSL_VERIFY(sw[2][0][k][j][i] <= numNeighborPts[2][0]);
                        }
                    }

                    //Positive-Z
                    if ((k < (numZones[2] - 1)) && (numLocalSplit[2][k + 1] > 0)) {
                        EMSL_VERIFY(sw[2][1][k][j][i] <= numLocalSplit[2][k + 1]);
                    } else {
                        if (numNeighborPts[2][1] > 0) {
                            EMSL_VERIFY(sw[2][1][k][j][i] <= numNeighborPts[2][1]);
                        }
                    }
                } //if zone exists
            }     //end i
        }         //end j
    }             //end k

    //Ensure negative and positive send buffers don't overlap - required for DataBlock
    for (int k = 0; k < numZones[2]; ++k) {
        for (int j = 0; j < numZones[1]; ++j) {
            for (int i = 0; i < numZones[0]; ++i) {
                if ((numLocalSplit[0][i] > 0) && (numLocalSplit[1][j] > 0) && (numLocalSplit[2][k] > 0)) {
                    //Negative-X
                    int negXsendSz = 0;
                    if ((i > 0) && (numLocalSplit[0][i - 1] > 0)) {
                        negXsendSz = sw[0][1][k][j][i - 1];
                    } else {
                        if (neighborSubDom[0][0] != -1) {
                            negXsendSz = sw[0][1][k][j][neighborSubDom[0][0]];
                        }
                    }

                    //Positive-X
                    int posXsendSz = 0;
                    if ((i < (numZones[0] - 1)) && (numLocalSplit[0][i + 1] > 0)) {
                        posXsendSz = sw[0][0][k][j][i + 1];
                    } else {
                        if (neighborSubDom[0][1] != -1) {
                            posXsendSz = sw[0][0][k][j][neighborSubDom[0][1]];
                        }
                    }

                    EMSL_VERIFY(numLocalSplit[0][i] > (negXsendSz + posXsendSz));

                    //Negative-Y
                    int negYsendSz = 0;
                    if ((j > 0) && (numLocalSplit[1][j - 1] > 0)) {
                        negYsendSz = sw[1][1][k][j - 1][i];
                    } else {
                        if (neighborSubDom[1][0] != -1) {
                            negYsendSz = sw[1][1][k][neighborSubDom[1][0]][i];
                        }
                    }

                    //Positive-Y
                    int posYsendSz = 0;
                    if ((j < (numZones[1] - 1)) && (numLocalSplit[1][j + 1] > 0)) {
                        posYsendSz = sw[1][0][k][j + 1][i];
                    } else {
                        if (neighborSubDom[1][1] != -1) {
                            posYsendSz = sw[1][0][k][neighborSubDom[1][1]][i];
                        }
                    }

                    EMSL_VERIFY(numLocalSplit[1][j] > (negYsendSz + posYsendSz));

                    //Negative-Z
                    int negZsendSz = 0;
                    if ((k > 0) && (numLocalSplit[2][k - 1] > 0)) {
                        negZsendSz = sw[2][1][k - 1][j][i];
                    } else {
                        if (neighborSubDom[2][0] != -1) {
                            negZsendSz = sw[2][1][neighborSubDom[2][0]][j][i];
                        }
                    }

                    //Positive-Z
                    int posZsendSz = 0;
                    if ((k < (numZones[2] - 1)) && (numLocalSplit[2][k + 1] > 0)) {
                        posZsendSz = sw[2][0][k + 1][j][i];
                    } else {
                        if (neighborSubDom[2][1] != -1) {
                            posZsendSz = sw[2][0][neighborSubDom[2][1]][j][i];
                        }
                    }

                    EMSL_VERIFY(numLocalSplit[2][k] > (negZsendSz + posZsendSz));
                } //if zone exists
            }     //end i
        }         //end j
    }             //end k

    //Ensure there is at least one independent point in each domain - required for overlapping communication with computation.
    for (int k = 0; k < numZones[2]; ++k) {
        for (int j = 0; j < numZones[1]; ++j) {
            for (int i = 0; i < numZones[0]; ++i) {
                if ((numLocalSplit[0][i] > 0) && (numLocalSplit[1][j] > 0) && (numLocalSplit[2][k] > 0)) {
                    int negXrecvSz = sw[0][0][k][j][i];
                    if ((coords[0] == 0) && (i == 0)) {
                        negXrecvSz = 0;
                    }

                    int posXrecvSz = sw[0][1][k][j][i];
                    if ((coords[0] == (dims[0] - 1)) && (i == (numZones[0] - 1))) {
                        posXrecvSz = 0;
                    }

                    EMSL_VERIFY(numLocalSplit[0][i] > (negXrecvSz + posXrecvSz));

                    int negYrecvSz = sw[1][0][k][j][i];
                    if ((coords[1] == 0) && (j == 0)) {
                        negYrecvSz = 0;
                    }

                    int posYrecvSz = sw[1][1][k][j][i];
                    if ((coords[1] == (dims[1] - 1)) && (j == (numZones[1] - 1))) {
                        posYrecvSz = 0;
                    }

                    EMSL_VERIFY(numLocalSplit[1][j] > (negYrecvSz + posYrecvSz));

                    int negZrecvSz = sw[2][0][k][j][i];
                    if ((coords[2] == 0) && (k == 0)) {
                        negZrecvSz = 0;
                    }

                    int posZrecvSz = sw[2][1][k][j][i];
                    if ((coords[2] == (dims[2] - 1)) && (k == (numZones[2] - 1))) {
                        posZrecvSz = 0;
                    }

                    EMSL_VERIFY(numLocalSplit[2][k] > (negZrecvSz + posZrecvSz));
                } //if zone exists
            }     //end i
        }         //end j
    }             //end k
} //check_grain_size
