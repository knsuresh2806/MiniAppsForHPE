
#include "cart_volume_regular.h"
#include "axis.h"
#include "emsl_error.h"

cart_volume_regular::~cart_volume_regular()
{
    if (_ax1)
        delete _ax1;

    if (_ax2)
        delete _ax2;

    if (_ax3)
        delete _ax3;
}
