
/*
Author: Rahul S. Sampath
*/

#include "partUtils1D.h"
#include "emsl_error.h"
#include <algorithm>
#include <vector>
#include <cstdio>

void
computeSendRecvLists(std::vector<int>& sendNegRanks, std::vector<int>& sendNegCnts, std::vector<int>& recvNegRanks,
                     std::vector<int>& recvNegCnts, std::vector<int>& sendPosRanks, std::vector<int>& sendPosCnts,
                     std::vector<int>& recvPosRanks, std::vector<int>& recvPosCnts, std::vector<int> const& counts,
                     int rank, int maxNegRecv, int maxPosRecv, int firstRankToRecv, int lastRankToRecv)
{
    sendNegRanks.clear();
    sendPosRanks.clear();
    sendNegCnts.clear();
    sendPosCnts.clear();
    recvNegRanks.clear();
    recvPosRanks.clear();
    recvNegCnts.clear();
    recvPosCnts.clear();

    //Send and Recv Neg
    if (rank > 0) {
        if (rank <= lastRankToRecv) {
            int recvRem = maxNegRecv;
            for (int other = (rank - 1); (other >= 0) && (recvRem > 0); --other) {
                recvNegRanks.push_back(other);
                recvNegCnts.push_back(std::min(recvRem, counts[other]));
                recvRem -= recvNegCnts.back();
            } //end other
        }

        int sendRem = maxPosRecv;
        for (int other = (rank - 1); (other >= firstRankToRecv) && (sendRem > 0); --other) {
            sendNegRanks.push_back(other);
            sendNegCnts.push_back(std::min(sendRem, counts[rank]));
            sendRem -= counts[other];
        } //end other
    }     //end neg

    //Send and Recv Pos
    if (rank < (int)(counts.size() - 1)) {
        if (rank >= firstRankToRecv) {
            int recvRem = maxPosRecv;
            for (int other = (rank + 1); (other < (int)counts.size()) && (recvRem > 0); ++other) {
                recvPosRanks.push_back(other);
                recvPosCnts.push_back(std::min(recvRem, counts[other]));
                recvRem -= recvPosCnts.back();
            } //end other
        }

        int sendRem = maxNegRecv;
        for (int other = (rank + 1); (other <= lastRankToRecv) && (sendRem > 0); ++other) {
            sendPosRanks.push_back(other);
            sendPosCnts.push_back(std::min(sendRem, counts[rank]));
            sendRem -= counts[other];
        } //end other
    }     //end pos
} //computeSendRecvLists

//A generic algorithm to partition a sequence of weighted points that preserves their relative order.
void
computeWeightedPartition1D(int* counts, int numPts, const realtype* wts, int npes)
{
    EMSL_VERIFY(numPts > 0);
    EMSL_VERIFY(npes > 0);
    if (numPts < npes) {
        fprintf(stderr, "Over decomposition. numPts=%d, npes=%d.\n", numPts, npes);
        EMSL_VERIFY(false);
    }

    //Cumulative sum of the weights
    realtype* scan = new realtype[numPts];
    scan[0] = wts[0];
    EMSL_VERIFY(wts[0] >= 0);
    for (int i = 1; i < numPts; ++i) {
        scan[i] = scan[i - 1] + wts[i];
        EMSL_VERIFY(wts[i] >= 0);
    }

    realtype totalWt = scan[numPts - 1];

    //Initialize counts
    for (int i = 0; i < npes; ++i) {
        counts[i] = -1;
    } //end i

    //The first point is assigned to rank 0.
    realtype currWt = wts[0];
    int currRank = 0;
    counts[0] = 1;

    //The maximum of the weights on ranks less than currRank
    realtype maxWt = 0;

    //The minimum of the weights on ranks less than currRank
    realtype minWt = 0;

    //For each point, determine if it should be assigned the same
    //rank as the previous point or the next higher rank.
    for (int i = 1; i < numPts; ++i) {
        if (currRank == (npes - 1)) {
            //There is no choice if you've reached the last rank already.
            ++(counts[currRank]);
        } else {
            if ((numPts - i) == (npes - currRank - 1)) {
                //There is no choice if the number of points remaining (including the current point) is
                //equal to the number of ranks remaining (not including current rank).
                //Increment currRank
                ++currRank;

                //This point will be the only point assigned to the new currRank
                counts[currRank] = 1;
            } else if ((numPts - i) < (npes - currRank - 1)) {
                //Number of points remaining (including the current point) is less than the number of
                //ranks remaining (not including current rank). Some rank will be empty, which is not allowed.
                EMSL_VERIFY(false);
            } else {
                if ((currRank > 0) && ((currWt + wts[i]) <= maxWt)) {
                    //Can add to currRank as the weight of currRank will still not exceed maxWt.
                    currWt += wts[i];
                    ++(counts[currRank]);
                } else {
                    //Case-1: The point is assigned to currRank
                    //Case-2: The point is assigned to currRank + 1.

                    //Average weight on the ranks higher than currRank in each case
                    realtype avg1 = (totalWt - scan[i]) / (npes - currRank - 1);
                    realtype avg2 = (totalWt - scan[i - 1]) / (npes - currRank - 1);

                    //Max and min weights across all ranks in each case
                    realtype max1, min1, max2, min2;
                    if (currRank > 0) {
                        //Max of maxWt, (currWt + wts[i]) and avg1.
                        max1 = std::max((std::max(maxWt, (currWt + wts[i]))), avg1);

                        //Min of minWt, (currWt + wts[i]) and avg1.
                        min1 = std::min((std::min(minWt, (currWt + wts[i]))), avg1);

                        //Max of maxWt, currWt and avg2.
                        max2 = std::max((std::max(maxWt, currWt)), avg2);

                        //Min of minWt, currWt and avg2.
                        min2 = std::min((std::min(minWt, currWt)), avg2);
                    } else {
                        //Max of (currWt + wts[i]) and avg1.
                        max1 = std::max((currWt + wts[i]), avg1);

                        //Min of (currWt + wts[i]) and avg1.
                        min1 = std::min((currWt + wts[i]), avg1);

                        //Max of currWt and avg2.
                        max2 = std::max(currWt, avg2);

                        //Min of currWt and avg2.
                        min2 = std::min(currWt, avg2);
                    }

                    if ((max1 - min1) <= (max2 - min2)) {
                        //Case-1 is more balanced, so the point is assigned to currRank
                        currWt += wts[i];
                        ++(counts[currRank]);
                    } else {
                        //Case-2 is more balanced, so the point is assigned to currRank + 1.
                        //Reset maxWt and minWt
                        if (currRank == 0) {
                            maxWt = currWt;
                            minWt = currWt;
                        } else {
                            maxWt = std::max(maxWt, currWt);
                            minWt = std::min(minWt, currWt);
                        }

                        //Increment currRank
                        ++currRank;

                        //This point will be the first point assigned to the new currRank
                        currWt = wts[i];
                        counts[currRank] = 1;
                    } //end if case-1
                }     //end if can add to currRank without exceeding maxWt
            }         //end if number of points remaining equals number of ranks remaining
        }             //end if last rank
    }                 //end i

    EMSL_VERIFY(currRank == (npes - 1));

    int totCnt = 0;
    for (int i = 0; i < npes; ++i) {
        EMSL_VERIFY(counts[i] > 0);
        totCnt += counts[i];
    } //end i
    EMSL_VERIFY(totCnt == numPts);

    delete[] scan;
} //computeWeightedPartition1D

void
splitSuperPtsForWpart1D(int* counts, const std::vector<int>& superPtMap, int npes)
{
    for (int i = 0, pt = 0; i < npes; ++i) {
        int initialCount = counts[i];
        for (int j = 0; j < initialCount; ++j, ++pt) {
            counts[i] += superPtMap[pt] - 1;
        } //end j
    }     //end i
} //splitSuperPtsForWpart1D

void
createSuperPtsForWpart1D(std::vector<int>& superPtMap, std::vector<realtype>& outWts, int inNumPts,
                         const realtype* inWts, int numRegions, const int* regionSizes, const int* negStencilSz,
                         const int* posStencilSz)
{
    EMSL_VERIFY(numRegions > 0);

    //Ensure no region is empty and numPts is consistent with regionSizes
    int sum = 0;
    for (int i = 0; i < numRegions; ++i) {
        EMSL_VERIFY(regionSizes[i] > 0);
        sum += regionSizes[i];
    } //end i
    EMSL_VERIFY(sum == inNumPts);

    //Ensure halo exchange is only between adjacent regions - required for Data-Block.
    for (int i = 0; i < numRegions; ++i) {
        //Negative
        if (i > 0) {
            EMSL_VERIFY(negStencilSz[i] <= regionSizes[i - 1]);
        }

        //Positive
        if (i < (numRegions - 1)) {
            EMSL_VERIFY(posStencilSz[i] <= regionSizes[i + 1]);
        }
    } //end i

    superPtMap.clear();
    outWts.clear();

    for (int i = 0, currPt = 0; i < numRegions; ++i) {
        int negSendSz = 0;
        if (i > 0) {
            negSendSz = posStencilSz[i - 1];
        }

        int posSendSz = 0;
        if (i < (numRegions - 1)) {
            posSendSz = negStencilSz[i + 1];
        }

        int negRecvSz = negStencilSz[i];
        if (i == 0) {
            negRecvSz = 0;
        }

        int posRecvSz = posStencilSz[i];
        if (i == (numRegions - 1)) {
            posRecvSz = 0;
        }

        //Ensure negative and positive send buffers don't overlap - required for Data-Block.
        //Ensure each region contains at least one independent point - required for overlapping communication with computation.
        EMSL_VERIFY(regionSizes[i] > std::max((negSendSz + posSendSz), (negRecvSz + posRecvSz)));

        int regMinNegSz = std::max((negSendSz + negStencilSz[i]), (negRecvSz + posStencilSz[i])) + 1;
        int regMinPosSz = std::max((posStencilSz[i] + posSendSz), (negStencilSz[i] + posRecvSz)) + 1;

        if (regionSizes[i] < (regMinNegSz + regMinPosSz)) {
            superPtMap.push_back(regionSizes[i]);
            realtype currSuperPtWt = 0;
            for (int j = 0; j < regionSizes[i]; ++j) {
                currSuperPtWt += inWts[currPt];
                ++currPt;
            } //end j
            outWts.push_back(currSuperPtWt);
        } else {
            int regMinMidSz = negStencilSz[i] + posStencilSz[i] + 1;
            int numMid = (regionSizes[i] - regMinNegSz - regMinPosSz) / regMinMidSz;
            int rem = regionSizes[i] - regMinNegSz - regMinPosSz - (numMid * regMinMidSz);

            int avgRem = rem / (numMid + 2);

            regMinNegSz += avgRem;
            regMinMidSz += avgRem;
            regMinPosSz += avgRem;

            rem = rem - ((numMid + 2) * avgRem);

            EMSL_VERIFY(rem < (numMid + 2));

            if (rem > 0) {
                ++regMinNegSz;
                --rem;
            }

            if (rem > 0) {
                ++regMinPosSz;
                --rem;
            }

            int numMid1 = std::min(numMid, rem);

            numMid = numMid - numMid1;

            rem = rem - numMid1;

            EMSL_VERIFY(rem == 0);

            realtype currSuperPtWt;

            superPtMap.push_back(regMinNegSz);
            currSuperPtWt = 0;
            for (int j = 0; j < regMinNegSz; ++j) {
                currSuperPtWt += inWts[currPt];
                ++currPt;
            } //end j
            outWts.push_back(currSuperPtWt);

            while ((numMid > 0) || (numMid1 > 0)) {
                if (numMid > 0) {
                    superPtMap.push_back(regMinMidSz);
                    currSuperPtWt = 0;
                    for (int j = 0; j < regMinMidSz; ++j) {
                        currSuperPtWt += inWts[currPt];
                        ++currPt;
                    } //end j
                    outWts.push_back(currSuperPtWt);
                    --numMid;
                }
                if (numMid1 > 0) {
                    superPtMap.push_back(regMinMidSz + 1);
                    currSuperPtWt = 0;
                    for (int j = 0; j < (regMinMidSz + 1); ++j) {
                        currSuperPtWt += inWts[currPt];
                        ++currPt;
                    } //end j
                    outWts.push_back(currSuperPtWt);
                    --numMid1;
                }
            } //end while

            superPtMap.push_back(regMinPosSz);
            currSuperPtWt = 0;
            for (int j = 0; j < regMinPosSz; ++j) {
                currSuperPtWt += inWts[currPt];
                ++currPt;
            } //end j
            outWts.push_back(currSuperPtWt);
        } //end if region can be split
    }     //end i
} //createSuperPtsForWpart1D

bool
checkPart1D(int npes, const int* counts, int numRegions, const int* regionSizes, const int* negStencilSz,
            const int* posStencilSz)
{
    EMSL_VERIFY(numRegions > 0);

    std::vector<std::vector<int>> subDomainSizes(npes);
    std::vector<std::vector<int>> subDomainRegId(npes);

    int totalNumPts = 0;
    int sum = 0;
    int currReg = 0;
    int currRegSz = regionSizes[currReg];
    for (int i = 0; i < npes; ++i) {
        int currCount = counts[i];
        totalNumPts += currCount;
        while (currCount > 0) {
            while ((currReg < numRegions) && (currRegSz == 0)) {
                ++currReg;
                EMSL_VERIFY(currReg < numRegions);
                currRegSz = regionSizes[currReg];
            }
            EMSL_VERIFY(currReg < numRegions);
            int newSz = std::min(currCount, currRegSz);
            sum += newSz;
            subDomainSizes[i].push_back(newSz);
            subDomainRegId[i].push_back(currReg);
            currRegSz = currRegSz - newSz;
            currCount = currCount - newSz;
        } //end while
    }     //end i
    EMSL_VERIFY(totalNumPts == sum);

    //Ensure halo exchange is only between adjacent domains - required for DataBlock
    for (int i = 0; i < npes; ++i) {
        for (size_t j = 0; j < subDomainSizes[i].size(); ++j) {
            //Negative
            if (j > 0) {
                if (negStencilSz[subDomainRegId[i][j]] > subDomainSizes[i][j - 1]) {
                    return false;
                }
            } else if (i > 0) {
                if (negStencilSz[subDomainRegId[i][j]] > subDomainSizes[i - 1][subDomainSizes[i - 1].size() - 1]) {
                    return false;
                }
            }

            //Positive
            if (j < (subDomainSizes[i].size() - 1)) {
                if (posStencilSz[subDomainRegId[i][j]] > subDomainSizes[i][j + 1]) {
                    return false;
                }
            } else if (i < (npes - 1)) {
                if (posStencilSz[subDomainRegId[i][j]] > subDomainSizes[i + 1][0]) {
                    return false;
                }
            }
        } //end j
    }     //end i

    //Ensure negative and positive send buffers don't overlap - required for DataBlock
    for (int i = 0; i < npes; ++i) {
        for (size_t j = 0; j < subDomainSizes[i].size(); ++j) {
            int negSendSz = 0;
            if (j > 0) {
                negSendSz = posStencilSz[subDomainRegId[i][j - 1]];
            } else if (i > 0) {
                negSendSz = posStencilSz[subDomainRegId[i - 1][subDomainRegId[i - 1].size() - 1]];
            }

            int posSendSz = 0;
            if (j < (subDomainSizes[i].size() - 1)) {
                posSendSz = negStencilSz[subDomainRegId[i][j + 1]];
            } else if (i < (npes - 1)) {
                posSendSz = negStencilSz[subDomainRegId[i + 1][0]];
            }

            if (subDomainSizes[i][j] <= (negSendSz + posSendSz)) {
                return false;
            }
        } //end j
    }     //end i

    //Ensure there is at least one independent point in each domain - required for overlapping communication with computation.
    for (int i = 0; i < npes; ++i) {
        for (size_t j = 0; j < subDomainSizes[i].size(); ++j) {
            int negRecvSz = negStencilSz[subDomainRegId[i][j]];
            if ((i == 0) && (j == 0)) {
                negRecvSz = 0;
            }

            int posRecvSz = posStencilSz[subDomainRegId[i][j]];
            if ((i == (npes - 1)) && (j == (subDomainSizes[i].size() - 1))) {
                posRecvSz = 0;
            }

            if (subDomainSizes[i][j] <= (negRecvSz + posRecvSz)) {
                return false;
            }
        } //end j
    }     //end i

    return true;
} //checkPart1D
