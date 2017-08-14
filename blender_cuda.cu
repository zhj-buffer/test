#include "blender_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>

class MatAddAction
{
    public:
        static __device__ __forceinline__   void MatAdd(int width, int height,short*src_laplace, float* src_weight, short*dst_laplace, float* dst_weight)
        {
            int i = threadIdx.x+blockIdx.x*blockDim.x;
            int j = threadIdx.y+blockIdx.y*blockDim.y;
            if(i<width && j<height)
            {
                (dst_laplace+j*width*3)[i*3+0] += (short)(src_laplace+j*width*3)[i*3+0]*(src_weight+j*width)[i];
                (dst_laplace+j*width*3)[i*3+1] += (short)(src_laplace+j*width*3)[i*3+1]*(src_weight+j*width)[i];
                (dst_laplace+j*width*3)[i*3+2] += (short)(src_laplace+j*width*3)[i*3+2]*(src_weight+j*width)[i];
                (dst_weight+j*width)[i]		+= (src_weight+j*width)[i];
                //if (i < 5 && j < 5)
                    //printf("i:<%d, %d> %d,%d,%d, %f\n", i, j, (src_laplace+j*width*3)[i*3+0], (src_laplace+j*width*3)[i*3+1], (src_laplace+j*width*3)[i*3+2], (src_weight+j*width)[i]);
            } 

        }
};


__global__ void MatAdd_(int width, int height,short*src_laplace, float* src_weight, short*dst_laplace, float* dst_weight)
{
    MatAddAction::MatAdd(width,height,src_laplace,src_weight,dst_laplace,dst_weight);
}

#define BLOCK 32

void MatAddEx(int width, int height, short*src_laplace, float* src_weight, short*dst_laplace, float* dst_weight)
{
    dim3 threadperblocks(BLOCK,BLOCK);
    dim3 blockspergrid((width+BLOCK - 1 )/BLOCK,(height+ BLOCK - 1)/BLOCK);

    MatAdd_<<<blockspergrid,threadperblocks>>>(width,height,src_laplace,src_weight,dst_laplace,dst_weight);
}
