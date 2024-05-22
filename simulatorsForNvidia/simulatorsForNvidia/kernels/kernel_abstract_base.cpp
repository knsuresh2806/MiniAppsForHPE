#include "kernel_abstract_base.h"
#include "cart_volume.h"

const int kernel_abstract_base::NMECH = 3;

kernel_abstract_base::kernel_abstract_base(int order, int bc_opt) : order{ order }, bc_opt{ bc_opt }
{}
