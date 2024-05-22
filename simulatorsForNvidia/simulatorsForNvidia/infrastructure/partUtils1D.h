
/*
 * Collection of 1D partitioning utility functions.

Author: Rahul S. Sampath
*/

#ifndef _PART_UTILS_1D
#define _PART_UTILS_1D

#include <vector>
#include "std_const.h"

bool checkPart1D(int npes, const int* counts, int numRegions, const int* regionSizes, const int* negStencilSz,
                 const int* posStencilSz);

void createSuperPtsForWpart1D(std::vector<int>& superPtMap, std::vector<realtype>& outWts, int inNumPts,
                              const realtype* inWts, int numRegions, const int* regionSizes, const int* negStencilSz,
                              const int* posStencilSz);

//A generic algorithm to partition a sequence of weighted points that preserves their relative order.
void computeWeightedPartition1D(int* counts, int numPts, const realtype* wts, int npes);

void splitSuperPtsForWpart1D(int* counts, const std::vector<int>& superPtMap, int npes);

void computeSendRecvLists(std::vector<int>& sendNegRanks, std::vector<int>& sendNegCnts, std::vector<int>& recvNegRanks,
                          std::vector<int>& recvNegCnts, std::vector<int>& sendPosRanks, std::vector<int>& sendPosCnts,
                          std::vector<int>& recvPosRanks, std::vector<int>& recvPosCnts, std::vector<int> const& counts,
                          int rank, int maxNegRecv, int maxPosRecv, int firstRankToRecv, int lastRankToRecv);

#endif
