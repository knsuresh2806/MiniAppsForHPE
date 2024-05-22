
/*************************************************************************
 * Cartesian volume module 
 *
 * Author: Rahul S. Sampath
 * Change Log:
 *
 *************************************************************************/

#ifndef _CART_VOLUME
#define _CART_VOLUME

#include "emsl_error.h"
#include "cart_domain.h"
#include "grid_const.h"
#include <map>
#include <utility>
#include <vector>
#include "Array.h"

// FIXME this class has a number of extremely bad problems and
// unfortunately, it is a fairly low-level class for the EMSL
// libraries.
//
// - Constructors modifying external state
//
// - Multiple sources of truth for information that are not
//   protected to ensure concistency: .
//
// - String member data that should not be part of this class. They
//   are not included in any of the constructors, but you can just
//   set them whenever because they are not protected.
//
// - Member functions that should be outside of the class definition
//   because they are not part of the minimal but complete iterface.
//
// - Explicit use of 32 integer values for quantities may be insufficient
//   and there is no plausible justification for not using a larger
//   type.
//
// All of these factors and more contribute to EMSL code being fragile, and
// reduce the effectivness developers and researchers working in this code.
//! Cartesian Volume
template <class T>
class cart_volume : public cart_domain<T>
{
public:
    //! Destructor
    virtual ~cart_volume() {}

    template <class U>
    U* as()
    {
        auto dyncast_ret = dynamic_cast<U*>(this);
        EMSL_VERIFY(dyncast_ret);
        return dyncast_ret;
    }

    template <class U>
    bool is()
    {
        return dynamic_cast<U*>(this) != nullptr;
    }

    //! set all values to a constant
    void set_constant(T val, bool skipHalos = true, bool skipInterleaving = false) override = 0;

    //! copy data from argument volume (used for checkpointing)
    virtual void copyData(cart_volume<T>* vol, bool skipHalos = true, bool skipInterleaving = false) = 0;

    /* Writable dimensions. */
    int Nx; //unaligned writable X dimension. (Note: X derivative loops can't be aligned)
    int Ny;
    int Nz;
};

#endif
