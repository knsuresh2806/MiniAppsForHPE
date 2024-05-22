
/**
 * @author: Rahul S. Sampath
 */

#include "std_utils.h"
#include "emsl_error.h"
#include <vector>
#include <string>

void
createQtree(unsigned int myYoff, unsigned int myZoff, unsigned int myYlen, unsigned int myZlen,
            std::vector<unsigned int>& yOffList, std::vector<unsigned int>& zOffList)
{
    EMSL_VERIFY(myYlen > 0);
    EMSL_VERIFY(myZlen > 0);

    unsigned int maxVal = std::max(myYlen, myZlen);

    unsigned int qTreeSz;
    if (isPowerOfTwo(maxVal)) {
        qTreeSz = maxVal;
    } else {
        qTreeSz = getNextPowerOfTwo(maxVal);
    }
    EMSL_VERIFY(qTreeSz >= maxVal);

    if (qTreeSz == 1) {
        EMSL_VERIFY(myYlen == 1);
        EMSL_VERIFY(myZlen == 1);
        yOffList.push_back(myYoff);
        zOffList.push_back(myZoff);
    } else {
        //Add sub quadtrees recursively (Y is the first/faster direction)
        unsigned int halfSz = qTreeSz / 2;
        unsigned int splitYsz = std::min(myYlen, halfSz);
        unsigned int splitZsz = std::min(myZlen, halfSz);
        createQtree(myYoff, myZoff, splitYsz, splitZsz, yOffList, zOffList);
        if (myYlen > splitYsz) {
            createQtree((myYoff + splitYsz), myZoff, (myYlen - splitYsz), splitZsz, yOffList, zOffList);
        }
        if (myZlen > splitZsz) {
            createQtree(myYoff, (myZoff + splitZsz), splitYsz, (myZlen - splitZsz), yOffList, zOffList);
            if (myYlen > splitYsz) {
                createQtree((myYoff + splitYsz), (myZoff + splitZsz), (myYlen - splitYsz), (myZlen - splitZsz),
                            yOffList, zOffList);
            }
        }
    } //end if base case
} //end createQtree

void
createQtreeCompressed(unsigned int myYoff, unsigned int myZoff, unsigned int myYlen, unsigned int myZlen,
                      std::vector<unsigned int>& yOffList, std::vector<unsigned int>& zOffList,
                      std::vector<unsigned int>& segLenList)
{
    unsigned int minVal = std::min(myYlen, myZlen);
    EMSL_VERIFY(minVal > 0);

    unsigned int splitSz;
    if (isPowerOfTwo(minVal)) {
        splitSz = minVal;
    } else {
        splitSz = getPrevPowerOfTwo(minVal);
    }

    unsigned int maxVal = std::max(myYlen, myZlen);

    unsigned int qTreeSz;
    if (isPowerOfTwo(maxVal)) {
        qTreeSz = maxVal;
    } else {
        qTreeSz = getNextPowerOfTwo(maxVal);
    }

    //Add current quadrant
    yOffList.push_back(myYoff);
    zOffList.push_back(myZoff);
    segLenList.push_back(splitSz * splitSz);

    //Add siblings of current quadrant, siblings of current quadrant's parent, siblings of current quadrant's grand-parent, etc.
    //(Y is the first/faster direction)
    while (splitSz < qTreeSz) {
        unsigned int remY = 0;
        if (myYlen > splitSz) {
            remY = std::min(splitSz, (myYlen - splitSz));
        }
        unsigned int remZ = 0;
        if (myZlen > splitSz) {
            remZ = std::min(splitSz, (myZlen - splitSz));
        }
        if (remY) {
            createQtreeCompressed((myYoff + splitSz), myZoff, remY, (std::min(splitSz, myZlen)), yOffList, zOffList,
                                  segLenList);
        }
        if (remZ) {
            createQtreeCompressed(myYoff, (myZoff + splitSz), (std::min(splitSz, myYlen)), remZ, yOffList, zOffList,
                                  segLenList);
            if (remY) {
                createQtreeCompressed((myYoff + splitSz), (myZoff + splitSz), remY, remZ, yOffList, zOffList,
                                      segLenList);
            }
        }
        splitSz *= 2;
    } //end while
} //end createQtreeCompressed

std::string
removeExtraWhiteSpaces(std::string const& str)
{
    std::string const whitespaces(" \t\f\v\n\r");
    size_t stPos = str.find_first_not_of(whitespaces);
    if (stPos == std::string::npos) {
        return "";
    }
    size_t endPos = str.find_last_not_of(whitespaces);
    if (endPos == std::string::npos) {
        return "";
    }
    return (str.substr(stPos, (endPos - stPos + 1)));
}

void
splitString(std::string const& str, char delim, std::vector<std::string>& out)
{
    std::string tmpString = removeExtraWhiteSpaces(str);
    while (tmpString.empty() == false) {
        size_t pos = tmpString.find(delim, 0);
        std::string sub = tmpString.substr(0, pos);
        if (out.empty() == false) {
            out.push_back(removeExtraWhiteSpaces(sub));
        }
        if (pos != std::string::npos) {
            std::string rem = tmpString.substr((pos + 1), std::string::npos);
            tmpString = removeExtraWhiteSpaces(rem);
        } else {
            tmpString.clear();
        }
    } //end while
}
