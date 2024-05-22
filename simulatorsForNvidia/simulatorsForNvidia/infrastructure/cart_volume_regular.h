
#ifndef _CART_VOLUME_REGULAR
#define _CART_VOLUME_REGULAR

#include "cart_volume.h"
#include <functional>

//Forward declarations
class axis;

class cart_volume_regular : public cart_volume<realtype>
{

public:
    // Destructor
    virtual ~cart_volume_regular();

    // As function to access the variables
    axis* ax1() const { return this->_ax1; }
    axis* ax2() const { return this->_ax2; }
    axis* ax3() const { return this->_ax3; }

protected:
    // FIXME The is no plausible justification for storing the axis as pointers.
    axis* _ax1; // local axes
    axis* _ax2;
    axis* _ax3;
};
#endif //_CART_VOLUME_REGULAR
