#ifndef KERNEL_TYPE1_GPU_ADJ_H
#define KERNEL_TYPE1_GPU_ADJ_H

namespace adj_main_loop_2_inner_gpu {

void launch(float* field3, float* field4, const float* field1, const float* field2, const float* model1, const float* model2,
            const float* model3, const float* model4, const float* model5, const float* model6, const float* model7, const float2* ABx,
            const float2* ABy, const float2* ABz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend,
            int ldimx, int ldimy, int ldimz, float dt, double dx, double dy, double dz, int order, cudaStream_t stream,
            bool use_sponge, bool simple = false);

}

#endif
