#pragma once
#include "volume_index.h"

void launch_update_grad_loop_kernel(volume_index idx, volume_index idx_adj, float* __restrict__ snap_x,
                                        float* __restrict__ snap_z, float* __restrict__ grad_Vp,
                                        float* __restrict__ adj_x, float* __restrict__ adj_z, int ixbeg, int ixend,
                                        int iybeg, int iyend, int izbeg, int izend);
