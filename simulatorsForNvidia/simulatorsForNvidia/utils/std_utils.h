/*************************************************************************
 *  Standard Utilities
 *
 * Author: Rahul Sampath    
 *************************************************************************/

#ifndef _STD_UTILS
#define _STD_UTILS

#include "std_const.h"
#include <vector>
#include <string>

inline float
big_float_min(float a, float b)
{
    float ret;
    if (a == _BIG_FLOAT)
        ret = b;
    else if (b == _BIG_FLOAT)
        ret = a;
    else if (a < b)
        ret = a;
    else
        ret = b;
    return ret;
}

inline float
big_float_max(float a, float b)
{
    float ret;
    if (a == _BIG_FLOAT)
        ret = b;
    else if (b == _BIG_FLOAT)
        ret = a;
    else if (a > b)
        ret = a;
    else
        ret = b;
    return ret;
}

//Assume n > 0
inline unsigned int
getPrevPowerOfTwo(unsigned int n)
{
    --n;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    ++n;
    return (n >> 1);
}

//Assume n > 0
inline unsigned int
getNextPowerOfTwo(unsigned int n)
{
    --n;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    ++n;
    return n;
}

//Assume n > 0
inline bool
isPowerOfTwo(unsigned int n)
{
    return (!(n & (n - 1)));
}

//Works even for non-square domains and even odd numbered dimensions. Square sub-regions are stored compressed (i.e., only the first quadrant is stored).
void createQtreeCompressed(unsigned int myYoff, unsigned int myZoff, unsigned int myYlen, unsigned int myZlen,
                           std::vector<unsigned int>& yOffList, std::vector<unsigned int>& zOffList,
                           std::vector<unsigned int>& segLenList);

//Works even for non-square domains and even odd numbered dimensions.
void createQtree(unsigned int myYoff, unsigned int myZoff, unsigned int myYlen, unsigned int myZlen,
                 std::vector<unsigned int>& yOffList, std::vector<unsigned int>& zOffList);

template <typename F, typename S>
bool
firstLessThan(std::pair<F, S> const& x, std::pair<F, S> const& y)
{
    return (x.first < y.first);
}

template <typename F, typename S>
bool
firstEqualTo(std::pair<F, S> const& x, std::pair<F, S> const& y)
{
    return (x.first == y.first);
}

template <typename F, typename S>
bool
secondLessThan(std::pair<F, S> const& x, std::pair<F, S> const& y)
{
    return (x.second < y.second);
}

template <typename F, typename S>
bool
secondEqualTo(std::pair<F, S> const& x, std::pair<F, S> const& y)
{
    return (x.second == y.second);
}

template <typename T>
void
map3Dto2D(T val3d[3][2], T val2d[2][2])
{
    val2d[0][0] = val3d[0][0];
    val2d[0][1] = val3d[0][1];
    val2d[1][0] = val3d[2][0];
    val2d[1][1] = val3d[2][1];
}

template <typename T>
void
map2Dto3D(T val2d[2][2], T val3d[3][2], T zero)
{
    val3d[0][0] = val2d[0][0];
    val3d[0][1] = val2d[0][1];
    val3d[1][0] = zero;
    val3d[1][1] = zero;
    val3d[2][0] = val2d[1][0];
    val3d[2][1] = val2d[1][1];
}

template <typename T>
bool
bitwiseEqual(T const& a, T const& b)
{
    unsigned char* aChr = (unsigned char*)(&a);
    unsigned char* bChr = (unsigned char*)(&b);
    for (int i = 0; i < sizeof(T); ++i) {
        if ((*(aChr + i)) != (*(bChr + i))) {
            return false;
        }
    } //end i
    return true;
}

template <typename EnumType, EnumType Last>
EnumType
intToEnumType(int i)
{
    EnumType res = Last;
    if ((i >= 0) && (i < Last)) {
        res = static_cast<EnumType>(i);
    }
    return res;
}

template <typename T>
void
deleteList(std::vector<T*>& list)
{
    for (int i = 0; i < list.size(); ++i) {
        if (list[i])
            delete (list[i]);
    } //end i
    list.clear();
}

// Removes leading and trailing whitespaces
std::string removeExtraWhiteSpaces(std::string const& str);

//Split input string into sub-strings using the specified delimiter and append the non-empty sub-strings to the output vector.
void splitString(std::string const& str, char delim, std::vector<std::string>& out);

#endif
