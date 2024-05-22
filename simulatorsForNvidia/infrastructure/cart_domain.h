
/*************************************************************************
 * Base class for Cartesian grids 
 *
 * Author:  Rahul S. Sampath
 * Change Log:
 *
 *************************************************************************/

#ifndef _CART_DOMAIN
#define _CART_DOMAIN

#include <cstdio>
#include "std_const.h"

template <class T>
class cart_domain
{
public:
    cart_domain() {}

    virtual ~cart_domain() {}

    //! set all values to 0
    void zero(bool skipHalos = true, bool skipInterleaving = false) { set_constant(T{}, skipHalos, skipInterleaving); }

    //! set all values to a constant
    virtual void set_constant(T val, bool skipHalos = true, bool skipInterleaving = false) = 0;
};

#endif
