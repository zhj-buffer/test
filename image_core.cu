#include "cuda_runtime.h"           //CUDA运行时ACUDART_PI_F  
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <math_constants.h>
#include <math_functions_dbl_ptx3.h>
//#include <math.h>

#include "device_launch_parameters.h"     
#include <malloc.h>
#include <stdio.h> 
#include <stdlib.h>
#include <sys/time.h>

#include "image_core.h"

#define CAM_W 1920
#define CAM_H 1080
#define BLOCK_NUM  64
#define THREAD_NUM 512

static const float WEIGHT_EPS = 1e-5f;

static struct timeval tv0;
static struct timeval tv1;
static struct timezone tz;

static const int IPL_DEPTH_SIGN = 0x80000000;

__device__ __host__ int getWidth(int w, int channel)
{
	int depth = 8;
	int align = 4;

	return  (((w * channel * (depth & ~IPL_DEPTH_SIGN) + 7)/8)+ align - 1) & (~(align - 1)); 
}

__global__ void cudaResizeLinear(float *src, float *dst, int w0, int h0, int w1, int h1)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int i;

	int y1,y2, x1,x2,  x, y;
	float fx1, fx2, fy1, fy2;

	for (i = bid * THREAD_NUM + tid; i < w1 * h1; i += BLOCK_NUM * THREAD_NUM)
	{
		x = (i) % w1;
		y = (i) / w1;

		x1 = (int)(x* ((float)w0 / (float)w1));
		x2 = (int)(x* ((float)w0 / (float)w1)) + 1;
		y1 = (int)(y* ((float)h0 / (float)h1));
		y2 = (int)(y* ((float)h0 / (float)h1)) + 1;

		fx1 = (((float)x* (((float)w0) / (float)w1))) - (int)(x * (((float)w0) / (float)w1));
		fx2 = 1.0f - fx1;
		fy1 = (((float)y* (((float)h0) / (float)h1))) - (int)(y * (((float)h0) / (float)h1));
		fy2 = 1.0f - fy1;

		float s1 = fx1*fy1;
		float s2 = fx2*fy1;
		float s3 = fx2*fy2;
		float s4 = fx1*fy2;

		dst[i * 3 + 0] = (src[y1 * w0 * 3 + x1 * 3 + 0]) * s3 + (src[y1 * w0 *3 + x2*3 + 0]) * s4 + (src[y2 * w0*3 + x1*3 + 0]) * s2 + (src[y2 * w0 *3 + x2 *3 + 0]) * s1;
		dst[i * 3 + 1] = (src[y1 * w0 * 3 + x1 * 3 + 1]) * s3 + (src[y1 * w0 *3 + x2*3 + 1]) * s4 + (src[y2 * w0*3 + x1*3 + 1]) * s2 + (src[y2 * w0 *3 + x2 *3 + 1]) * s1;
		dst[i * 3 + 2] = (src[y1 * w0 * 3 + x1 * 3 + 2]) * s3 + (src[y1 * w0 *3 + x2*3 + 2]) * s4 + (src[y2 * w0*3 + x1*3 + 2]) * s2 + (src[y2 * w0 *3 + x2 *3 + 2]) * s1;
	}
}

__global__ void addKernel(float *a,  const char *b, int w, int h)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;

    for (i = bid * THREAD_NUM + tid; i < w * h / 2; i += BLOCK_NUM * THREAD_NUM) {

        a[i*6 + 0]=(10000*b[i*4 + 1]+14075*(b[i*4 + 2]-128))/10000;
        a[i*6 + 1]=(10000*b[i*4 + 1]-3455*( b[i*4 + 0]-128)-7169*(b[i*4 + 2]-128))/10000;
        a[i*6 + 2]=(10000*b[i*4 + 1]+17990*(b[i*4 + 0]-128))/10000;
        a[i*6 + 3]=(10000*b[i*4 + 3]+14075*(b[i*4 + 2]-128))/10000;
        a[i*6 + 4]=(10000*b[i*4 + 3]-3455*( b[i*4 + 0]-128)-7169*(b[i*4 + 2]-128))/10000;
        a[i*6 + 5]=(10000*b[i*4 + 3]+17990*(b[i*4 + 0]-128))/10000;

        if(a[i*6 + 0]>255) a[i*6 + 0]=255; if(a[i*6 + 0]<0) a[i*6 + 0]=0;
        if(a[i*6 + 1]>255) a[i*6 + 1]=255; if(a[i*6 + 1]<0) a[i*6 + 1]=0;
        if(a[i*6 + 2]>255) a[i*6 + 2]=255; if(a[i*6 + 2]<0) a[i*6 + 2]=0;
        if(a[i*6 + 3]>255) a[i*6 + 3]=255; if(a[i*6 + 3]<0) a[i*6 + 3]=0;
        if(a[i*6 + 4]>255) a[i*6 + 4]=255; if(a[i*6 + 4]<0) a[i*6 + 4]=0;
        if(a[i*6 + 5]>255) a[i*6 + 5]=255; if(a[i*6 + 5]<0) a[i*6 + 5]=0;

    }
}

__global__ void cudasplice(char *src0, char *src1, char *src2, char * src3, char *dst, int w, int h)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i, h1, w1;
    int size = w * h;
    int bw = 2 * w;

    for (i = bid * THREAD_NUM + tid; i < size * 4; i += BLOCK_NUM * THREAD_NUM) {
        w1 = i % bw;
        h1 = i / bw;
        if (((i % bw) < w ) && (i / bw < h)) {
            dst[i * 2 + 0] = src0[(h1 * w + w1) * 2 + 0];
            dst[i * 2 + 1] = src0[(h1 * w + w1) * 2 + 1];
        } else if ((i % bw >= w ) && (i / bw < h)) {
            dst[i * 2 + 0] = src1[(h1 * w + w1) * 2 + 0];
            dst[i * 2 + 1] = src1[(h1 * w + w1) * 2 + 1];
        } else if ((i % bw < w ) && (i / bw >= h)) {
            h1 = h1 - h;
            dst[i * 2 + 0] = src2[(h1 * w + w1) * 2 + 0];
            dst[i * 2 + 1] = src2[(h1 * w + w1) * 2 + 1];
        } else if ((i % bw >= w ) && (i / bw >= h)) {
            h1 = h1 - h;
            dst[i * 2 + 0] = src3[(h1 * w + w1) * 2 + 0];
            dst[i * 2 + 1] = src3[(h1 * w + w1) * 2 + 1];
        } else {
            printf(" Should not be here\n");
        }
    }
}
__global__ void cudaShowconvert(char *dst, const float *src, int w, int h)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;

    for (i = bid * THREAD_NUM + tid; i < w * h; i += BLOCK_NUM * THREAD_NUM) {
        dst[i * 3 + 0] = src[i * 3 + 2];
        dst[i * 3 + 1] = src[i * 3 + 1];
        dst[i * 3 + 2] = src[i * 3 + 0];
    }
}

__global__ void cudabgr2rgb(float *dst, const float *src, int w, int h)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i, j, c;

    for (c = 0; c < 3; c++)
        for (j = 0; j < h; j++)
            for (i = bid * THREAD_NUM + tid; i < w; i += BLOCK_NUM * THREAD_NUM) {
                dst[i  + j *  w  + c * h * w] = src[i * 3 + 3 * w * j + c] / 255.;
            }
}


__global__ void cudaswapfloat(float *dst, int w, int h)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;
    float swap;

    for (i = bid * THREAD_NUM + tid; i < w * h; i += BLOCK_NUM * THREAD_NUM) {
        swap = dst[i];
        dst[i] = dst[i + w*h*2];
        dst[i + w*h*2] = swap;
    }
}

bool InitCUDA(void)
{
    int count = 0;
    int i = 0;
    cudaGetDeviceCount(&count); //看看有多少个设备?
    if(count == 0)   //哈哈~~没有设备.
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    cudaDeviceProp prop;
    for(i = 0; i < count; i++)  //逐个列出设备属性:
    {
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if(prop.major >= 1)
            {
                break;
            }
        }
    }
    if(i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }
    cudaSetDevice(0);

    cudaDeviceProp sDevProp = prop;

    printf( "\n\nGPU Num: %d \n", i);
    printf( "Device name: %s\n", sDevProp.name );
    printf( "Device memory: %lu\n", sDevProp.totalGlobalMem );
    printf( "Memory per-block: %lu\n", sDevProp.sharedMemPerBlock );
    printf( "Register per-block: %u\n", sDevProp.regsPerBlock );
    printf( "Warp size: %u\n", sDevProp.warpSize );
    printf( "Memory pitch: %lu\n", sDevProp.memPitch );
    printf( "Constant Memory: %lu\n", sDevProp.totalConstMem );
    printf( "Max thread per-block: %u\n", sDevProp.maxThreadsPerBlock );
    printf( "Max thread dim: ( %d, %d, %d )\n", sDevProp.maxThreadsDim[0],
            sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2] );
    printf( "Max grid size: ( %d, %d, %d )\n", sDevProp.maxGridSize[0],  
            sDevProp.maxGridSize[1], sDevProp.maxGridSize[2] );
    printf( "Ver: %d.%d\n", sDevProp.major, sDevProp.minor );
    printf( "Clock: %d\n", sDevProp.clockRate );
    printf( "textureAlignment: %lu\n", sDevProp.textureAlignment );
    printf( "CUDART_VERSION: %d\n", CUDART_VERSION);

    if (!prop.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", i);

    } else {
        printf("Device %d support mapping CPU host memory!\n", i);
    }

    cudaSetDeviceFlags(cudaDeviceMapHost);


    printf("\nCUDA initialized.\n\n");
    return true;
}

void cudayuv2rgb(float *dev_a, const char *dev_b, int w, int h)
{
    gettimeofday(&tv0, &tz);
	addKernel<<<BLOCK_NUM, THREAD_NUM>>>(dev_a, dev_b, w, h);
    gettimeofday(&tv1, &tz);
    //printf("\n kernel running Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);
}

void cuda_resize(float *src, float *dst, int src_w, int src_h, int dst_w, int dst_h)
{
    gettimeofday(&tv0, &tz);
	cudaResizeLinear<<<BLOCK_NUM, THREAD_NUM>>>(src, dst, src_w, src_h, dst_w, dst_h);
    gettimeofday(&tv1, &tz);
    //printf("\n kernel running Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);
}

void cudaShowCon(char *dst, const float *src, int w, int h)
{
    gettimeofday(&tv0, &tz);
    cudaShowconvert<<<BLOCK_NUM, THREAD_NUM>>>(dst, src, w, h);
    gettimeofday(&tv1, &tz);
//    printf("\n kernel show convert Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);
}
void cudabgrtorgb(float *dst, const float *src, int w, int h)
{
    gettimeofday(&tv0, &tz);
    cudabgr2rgb<<<BLOCK_NUM, THREAD_NUM>>>(dst, src, w, h);
    gettimeofday(&tv1, &tz);
//    printf("\n kernel show convert Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);
}
void cudaswap(float *dst, int w, int h)
{
    gettimeofday(&tv0, &tz);
    cudaswapfloat<<<BLOCK_NUM, THREAD_NUM>>>(dst,w, h);
    gettimeofday(&tv1, &tz);
//    printf("\n kernel show convert Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);
}

void cuda_splice(char *src, int w, int h)
{
    int offset = w * h * 2;
    gettimeofday(&tv0, &tz);
    cudasplice<<<BLOCK_NUM, THREAD_NUM>>>(src, src + offset * 1, src + offset * 2, src + offset * 3, src + offset * 4, w, h);
    gettimeofday(&tv1, &tz);
//    printf("\n kernel show convert Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);
}

void cuda_splice_four(char *src0, char *src1, char *src2, char *src3, char *dst, int w, int h)
{
    //int offset = w * h * 2;
    gettimeofday(&tv0, &tz);
    cudasplice<<<BLOCK_NUM, THREAD_NUM>>>(src0, src1, src2, src3, dst, w, h);
    gettimeofday(&tv1, &tz);
//    printf("\n kernel show convert Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);
}


__global__ void
standardCircleKernel(void *in, int in_widthStep, int w, int h, void *out, int out_widthStep, int offset, int n_left)
{
    char *src = (char *)in;
    char *dest = (char *)out;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < w && col < h)
    {
        for (int k = 0; k < 3; k++)
        {
//            pdata[width * height * k + row * width + col] =
//                ((float)*(psrcdata + row * pitch + col * 4 + (3 - 1 - k))) / 255.;
//            pdata[width * height * k + row * width + col] =
//                ((float)*(psrcdata + row * pitch + col * 4 + (k))) / 255.;
//            show_buf[width * row * 3 + col * 3 + k] = *(psrcdata + row * pitch + col * 4 + (3 - 1 - k));
            (dest + (col + offset) * out_widthStep)[row * 3 + k] = (src + (col) * in_widthStep)[(row + n_left) * 3 + k];
        }
    }
}

#if 1
int standardCircle(void *in, int in_widthStep, void *out, int out_widthSetp)
{
    int n_right = 1710;
    int n_left = 266;
    int n_bottom = 1080;
    int n_top = 0;
    int w = n_right - n_left;
    int h = n_bottom - n_top;
    //h = 1080;
    int offset = (w - h) / 2; // == 182   1080/2 = 540  => 572 ... 
    offset = 182 - (572 - 540); // == 182   1080/2 = 540  => 572 ...    
    // circle center(266 + 722, 572) r= 722

    dim3 threadsPerBlock(32, 32);
    dim3 blocks(w/threadsPerBlock.x, h/threadsPerBlock.y);
    cudaStream_t stream;

    //    convertIntToFloatKernel<<<blocks, threadsPerBlock>>>(pDevPtr, width,
    //                height, cuda_buf, pitch, show_buf);

    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "!!! cannot create stream\n");
        return -1;
    }

//    printf("ins: %d, w:%d, h:%d, ous: %d\n", in_widthStep, w, h, out_widthSetp);
    standardCircleKernel<<<blocks, threadsPerBlock, 0, stream>>>
        (in, in_widthStep, w, h, out, out_widthSetp, offset, n_left);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 0;
}
#endif
__device__  void auxFunc(double w, double phi, double *ret)
{
    double l;
   l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(CUDART_PI_F - w - atan((1 - sin(phi)) / abs(cos(phi))));
    if (phi > CUDART_PI_F / 2)
        l = -l;
    *ret = l;
}

//double auxFunc(double w, double phi)
#define AUXFUNC(w, phi) \
{ \
    double l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(CUDART_PI_F - w - atan((1 - sin(phi)) / abs(cos(phi)))); \
    if (phi > CUDART_PI_F / 2) \
        l = -l; \
    return l; \
}

#if 1

__device__ void func1(double l, double phi, double w, double *ret)
{
    double limit = 0.0;
    auxFunc(w, 0, &limit);
    double result = 0.0;
    auxFunc(w, phi, &result);

    result = l - limit + result;

    *ret = result;
}

__device__ void getPhi1(double l, double w, double *ret)
//#define GETPHI(l, w)
{
    int N_lim = 100;
    int N = 0;
    //printf("========1 ===============");
    double lim = 0.0;
    auxFunc(w, 0, &lim);

    double head = 0;
    double tail = 0;
    double mid = 0;
    double result = 0;
    if (l >= 0 && l < lim)
    {
        head = 0;
        tail = CUDART_PI_F / 2;
        mid = head;
        func1(l, mid, w, &result);
        while (abs(result)>LIMIT && N++ < N_lim)
        {
            mid = (tail + head) / 2;
            func1(l, mid, w, &result);

            if (result > 0)
            {
                head = mid;
            }
            else
            {
                tail = mid;
            }
        }
    }
    else
    {
        N = 0;
        head = CUDART_PI_F / 2;
        tail = CUDART_PI_F;
        mid = tail;
        func1(l, mid, w, &result);
        while (abs(result) > LIMIT&&N++ < N_lim)
        {
            mid = (tail + head) / 2;
            func1(l, mid, w, &result);
            if (result > 0)
            {
                head = mid;
            }
            else
            {
                tail = mid;
            }
        }
    }

    *ret = mid;
}
#endif

static const double atan2_p1 = 0.9997878412794807f*(double)(180/CUDART_PI_F);
static const double atan2_p3 = -0.3258083974640975f*(double)(180/CUDART_PI_F);
static const double atan2_p5 = 0.1555786518463281f*(double)(180/CUDART_PI_F);
static const double atan2_p7 = -0.04432655554792128f*(double)(180/CUDART_PI_F);
#define DBL_EPSILON (2.2204460492503131E-16)

__device__ double cvFastArctan(double y, double x)
{
    double ax = abs(x), ay = abs(y);
    double a, c, c2;
    if( ax >= ay )
    {
        c = ay/(ax + DBL_EPSILON);
        c2 = c*c;
        a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    else
    {
        c = ax/(ay + DBL_EPSILON);
        c2 = c*c;
        a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    if( x < 0 )
        a = 180.f - a;
    if( y < 0 )
        a = 360.f - a;

    //*ret = a;
    return a;
}

__global__ void
latitudeCorrectionKernel(void *src, void *dest, int width, int height, double w_latitude, double w_longtitude,  double camerFieldAngle)
{
    char *in = (char *)src;
    char *out = (char *)dest;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

   // int width = max(width, height);
    //int width = 512;
    //int height = width;

    double dx = camerFieldAngle / width;
    double dy = camerFieldAngle / height;


    int center_x = width / 2;
    int center_y = height / 2;

    //coordinate for latitude map
    double latitude;
    double longitude;

    //parameter cooradinate of sphere coordinate
    double Theta_sphere;
    double Phi_sphere;

    double foval = 8.0;
    double radius = 722;
    radius = 700;

    //unity sphere coordinate 
    double x, y, z, r;

    double longitude_offset, latitude_offset;
    longitude_offset = (CUDART_PI_F - camerFieldAngle) / 2;
    latitude_offset = (CUDART_PI_F - camerFieldAngle) / 2;

    //according to the camera type to do the calibration
//    double  limi_latitude = 2 * auxFunc(w_latitude, 0); // 4.0
//    double  limi_longtitude = 2 * auxFunc(w_longtitude, 0); // 4.0


    double l, w, phi;
    double  limi_latitude;
    double  limi_longtitude;
    w_latitude = CUDART_PI_F/2.0;
    w_longtitude = CUDART_PI_F/2.0;
    //auxFunc(CUDART_PI_F/2, 0, &limi_latitude); // 4.0
    //auxFunc(CUDART_PI_F/2, 0, &limi_longtitude); // 4.0
    limi_latitude = 4.0;
    limi_longtitude = 4.0;
    //printf("w_longtitude: %lf  &limi_longtitude: %lf, height: %d, w_latitude,: %lf\n", w_longtitude, limi_longtitude, height, w_latitude);

#if 0
    w = w_latitude;
    phi = 0;
    l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(CUDART_PI_F - w - atan((1 - sin(phi)) / abs(cos(phi))));
    if (phi > CUDART_PI_F / 2)
        l = -l;
    limi_latitude = 2 * l; // 4.0

    w = w_longtitude;
    phi = 0;
    l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(CUDART_PI_F - w - atan((1 - sin(phi)) / abs(cos(phi))));
    if (phi > CUDART_PI_F / 2)
        l = -l;
    limi_longtitude = 2 * l; // 4.0
#endif
   // limi_latitude = 4.0;
   // limi_longtitude = 4.0;
    //polar cooradinate for fish-eye Image
    double p;
    double theta;

    double x_cart, y_cart;

    int u, v;

    //func1(0.2, 0, 0.4, &p);
    //getPhi1((double)2.0*limi_latitude / (double)height, w_latitude, &latitude);
    //printf("======= kernel =====lat: %lf, long: %lf====== w: %d, j: %d,  sin %f\n", limi_latitude , limi_longtitude , w, j,  0);
#if 1
    if (i < width && j < height)
    {

        //latitude = latitude_offset + j*dy;

        getPhi1((double)j*limi_latitude / height, w_latitude, &latitude);
        //longitude = getPhi1((double)i * limi_longtitude / imgSize.width,w_longtitude);
        //latitude = latitude_offset + j*dy;
        longitude = longitude_offset + i*dx;
        //Convert from latitude cooradinate to the sphere cooradinate
        x = -sin(latitude)*cos(longitude);
        y = cos(latitude);
        z = sin(latitude)*sin(longitude);


        //Convert from unit sphere cooradinate to the parameter sphere cooradinate
        Theta_sphere = acos(z);
        Phi_sphere = cvFastArctan(y, x);//return value in Angle
        //Phi_sphere = arctan(y, x);//return value in Angle
        //Phi_sphere = atan2(y, x);//return value in Angle
        //printf("x: %lf, y: %lf, Phi_sphere: %lf\n", x, y, Phi_sphere);
        Phi_sphere = Phi_sphere*CUDART_PI_F / 180;//Convert from Angle to Radian

#if 0
        switch (camProjMode)
        {
            case STEREOGRAPHIC:
                foval = radius / (2 * tan(camerFieldAngle / 4));
                p = 2 * foval*tan(Theta_sphere / 2);
                break;
            case EQUIDISTANCE:
                foval = radius / (camerFieldAngle / 2);
                p = foval*Theta_sphere;
                break;
            case EQUISOLID:
                foval = radius / (2 * sin(camerFieldAngle / 4));
                p = 2 * foval*sin(Theta_sphere / 2);
                break;
            case ORTHOGONAL:
                foval = radius / sin(camerFieldAngle / 2);
                p = foval*sin(Theta_sphere);
                break;
            default:
                break;
                //cout << "The camera mode hasn't been choose!" << endl;
        }
#else
        foval = radius / (camerFieldAngle / 2);
        //printf("foval: %f \n", foval);
        p = foval*Theta_sphere;
#endif
        //Convert from parameter sphere cooradinate to fish-eye polar cooradinate
        //p = sin(Theta_sphere);
        theta = Phi_sphere;

        //Convert from fish-eye polar cooradinate to cartesian cooradinate
        x_cart = p*cos(theta);
        y_cart = p*sin(theta);

        //double R = radius / sin(camerFieldAngle / 2);

        //Convert from cartesian cooradinate to image cooradinate
        u = x_cart + center_x;
        v = -y_cart + center_y;
        //printf(" u: %d, v: %d, j: %d, i: %d\n ", u, v, j, i);

        if ( u >= 0 && u <= height && v >= 0 && v <= width) {
            for (int k = 0; k < 3; k++)
                (out + j * width * 3)[i * 3 + k] = (in + v * width * 3)[u * 3 + k];
        }
    }
#endif

}
#if 1
int latitudeCorrection4(void *src, void *dest,  int radius, double w_longtitude, double w_latitude,  distMapMode distMap, double theta_left, double phi_up, double camerFieldAngle, camMode camProjMode, int w, int h)
{
    if (!(camerFieldAngle > 0 && camerFieldAngle <= CUDART_PI_F))
    {
        printf( "The parameter \"camerFieldAngle\" must be in the interval (0,CUDART_PI_F].");
        return -1;
    }

    char *in = (char *)src;
    char *out = (char *)dest;

    double rateOfWindow = 0.9;

    dim3 threadsPerBlock(16, 32);
    dim3 blocks(w/threadsPerBlock.x, h/threadsPerBlock.y);
    cudaStream_t stream;

    //    convertIntToFloatKernel<<<blocks, threadsPerBlock>>>(pDevPtr, width,
    //                height, cuda_buf, pitch, show_buf);

    //printf("w:%d, h: %d ---- \n", w, h);
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "!!! cannot create stream\n");
        return -1;
    }

    //printf("w:%d, h: %d w_latitude: %lf, w_longtitude: %lf, camerFieldAngle: %lf-===\n", w, h, w_latitude, w_longtitude, camerFieldAngle);
    gettimeofday(&tv0, &tz);

    latitudeCorrectionKernel<<<blocks, threadsPerBlock, 0, stream>>>
        (src, dest, w, h, w_latitude, w_longtitude,  camerFieldAngle);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    gettimeofday(&tv1, &tz);
    //printf("\n latitudeCorrection4 kernel running Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);

    return 0;
}
#endif
__device__  void auxFuncf(float w, float phi, float *ret)
{
    float l;
   l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(CUDART_PI_F - w - atan((1 - sin(phi)) / abs(cos(phi))));
    if (phi > CUDART_PI_F / 2)
        l = -l;
    *ret = l;
}

__device__ void func(float l, float phi, float w, float *ret)
{
    float limit = 0.0;
    auxFuncf(w, 0, &limit);
    float result = 0.0;
    auxFuncf(w, phi, &result);

    result = l - limit + result;

    *ret = result;
}

__device__ void getPhi(float l, float w, float *ret)
//#define GETPHI(l, w)
{
    int N_lim = 100;
    int N = 0;
    //printf("========1 ===============");
    float lim = 0.0;
    auxFuncf(w, 0, &lim);

    float head = 0;
    float tail = 0;
    float mid = 0;
    float result = 0;
    if (l >= 0 && l < lim)
    {
        head = 0;
        tail = CUDART_PI_F / 2;
        mid = head;
        func(l, mid, w, &result);
        while (abs(result)>LIMIT && N++ < N_lim)
        {
            mid = (tail + head) / 2;
            func(l, mid, w, &result);

            if (result > 0)
            {
                head = mid;
            }
            else
            {
                tail = mid;
            }
        }
    }
    else
    {
        N = 0;
        head = CUDART_PI_F / 2;
        tail = CUDART_PI_F;
        mid = tail;
        func(l, mid, w, &result);
        while (abs(result) > LIMIT&&N++ < N_lim)
        {
            mid = (tail + head) / 2;
            func(l, mid, w, &result);
            if (result > 0)
            {
                head = mid;
            }
            else
            {
                tail = mid;
            }
        }
    }

    *ret = mid;
}

static const float atan2_pf1 = 0.9997878412794807f*(float)(180/CUDART_PI_F);
static const float atan2_pf3 = -0.3258083974640975f*(float)(180/CUDART_PI_F);
static const float atan2_pf5 = 0.1555786518463281f*(float)(180/CUDART_PI_F);
static const float atan2_pf7 = -0.04432655554792128f*(float)(180/CUDART_PI_F);
#define DBL_EPSILON_F ((float)(2.2204460492503131E-16))

__device__ float cvFastArctanf(float y, float x)
{
    float ax = abs(x), ay = abs(y);
    float a, c, c2;
    if( ax >= ay )
    {
        c = ay/(ax + DBL_EPSILON_F);
        c2 = c*c;
        a = (((atan2_pf7*c2 + atan2_pf5)*c2 + atan2_pf3)*c2 + atan2_pf1)*c;
    }
    else
    {
        c = ax/(ay + DBL_EPSILON_F);
        c2 = c*c;
        a = 90.f - (((atan2_pf7*c2 + atan2_pf5)*c2 + atan2_pf3)*c2 + atan2_pf1)*c;
    }
    if( x < 0 )
        a = 180.f - a;
    if( y < 0 )
        a = 360.f - a;

    //*ret = a;
    return a;
}

static int *xy = NULL;//[760][760][2];

__global__ void
xytouv(void *src, void *dest, int width, int height, int *xy)
{
    char *in = (char *)src;
    char *out = (char *)dest;
    int u, v;

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width && j < height) {
	u = (xy + j * width * 2)[i * 2 + 0];
	v = (xy + j * width * 2)[i * 2 + 1];
            for (int k = 0; k < 3; k++)
                (out + j * width * 3)[i * 3 + k] = (in + v * width * 3)[u * 3 + k];
    }
}
__global__ void
latitudeCorrectionfKernel(void *src, void *dest, int width, int height, float w_latitude, float w_longtitude,  float camerFieldAngle, int *xy)
{
    char *in = (char *)src;
    char *out = (char *)dest;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

   // int width = max(width, height);
    //int width = 512;
    //int height = width;

    float dx = camerFieldAngle / width;
    float dy = camerFieldAngle / height;


    int center_x = width / 2;
    int center_y = height / 2;

    //coordinate for latitude map
    float latitude;
    float longitude;

    //parameter cooradinate of sphere coordinate
    float Theta_sphere;
    float Phi_sphere;

    float foval = 8.0;
    float radius = 722;
    //radius = 700;
    radius = width / 2;

    //unity sphere coordinate 
    float x, y, z, r;

    float longitude_offset, latitude_offset;
    longitude_offset = (CUDART_PI_F - camerFieldAngle) / 2;
    latitude_offset = (CUDART_PI_F - camerFieldAngle) / 2;

    //according to the camera type to do the calibration
//    float  limi_latitude = 2 * auxFunc(w_latitude, 0); // 4.0
//    float  limi_longtitude = 2 * auxFunc(w_longtitude, 0); // 4.0


    float l, w, phi;
    float  limi_latitude;
    float  limi_longtitude;
    w_latitude = CUDART_PI_F/2.0;
    w_longtitude = CUDART_PI_F/2.0;
    //auxFunc(CUDART_PI_F/2, 0, &limi_latitude); // 4.0
    //auxFunc(CUDART_PI_F/2, 0, &limi_longtitude); // 4.0
    limi_latitude = 4.0;
    limi_longtitude = 4.0;
    //printf("w_longtitude: %lf  &limi_longtitude: %lf, height: %d, w_latitude,: %lf\n", w_longtitude, limi_longtitude, height, w_latitude);

#if 0
    w = w_latitude;
    phi = 0;
    l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(CUDART_PI_F - w - atan((1 - sin(phi)) / abs(cos(phi))));
    if (phi > CUDART_PI_F / 2)
        l = -l;
    limi_latitude = 2 * l; // 4.0

    w = w_longtitude;
    phi = 0;
    l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(CUDART_PI_F - w - atan((1 - sin(phi)) / abs(cos(phi))));
    if (phi > CUDART_PI_F / 2)
        l = -l;
    limi_longtitude = 2 * l; // 4.0
#endif
   // limi_latitude = 4.0;
   // limi_longtitude = 4.0;
    //polar cooradinate for fish-eye Image
    float p;
    float theta;

    float x_cart, y_cart;

    int u, v;

    //func1(0.2, 0, 0.4, &p);
    //getPhi1((float)2.0*limi_latitude / (float)height, w_latitude, &latitude);
    //printf("======= kernel =====lat: %lf, long: %lf====== w: %d, j: %d,  sin %f\n", limi_latitude , limi_longtitude , w, j,  0);
    if (i < width && j < height)
    {

        //latitude = latitude_offset + j*dy;

        getPhi((float)j*limi_latitude / height, w_latitude, &latitude);
        //longitude = getPhi1((float)i * limi_longtitude / imgSize.width,w_longtitude);
        //latitude = latitude_offset + j*dy;
        longitude = longitude_offset + i*dx;
        //Convert from latitude cooradinate to the sphere cooradinate
        x = -sin(latitude)*cos(longitude);
        y = cos(latitude);
        z = sin(latitude)*sin(longitude);


        //Convert from unit sphere cooradinate to the parameter sphere cooradinate
        Theta_sphere = acos(z);
        Phi_sphere = cvFastArctanf(y, x);//return value in Angle
        //Phi_sphere = arctan(y, x);//return value in Angle
        //Phi_sphere = atan2(y, x);//return value in Angle
        //printf("x: %lf, y: %lf, Phi_sphere: %lf\n", x, y, Phi_sphere);
        Phi_sphere = Phi_sphere*CUDART_PI_F / 180;//Convert from Angle to Radian

#if 0
        switch (camProjMode)
        {
            case STEREOGRAPHIC:
                foval = radius / (2 * tan(camerFieldAngle / 4));
                p = 2 * foval*tan(Theta_sphere / 2);
                break;
            case EQUIDISTANCE:
                foval = radius / (camerFieldAngle / 2);
                p = foval*Theta_sphere;
                break;
            case EQUISOLID:
                foval = radius / (2 * sin(camerFieldAngle / 4));
                p = 2 * foval*sin(Theta_sphere / 2);
                break;
            case ORTHOGONAL:
                foval = radius / sin(camerFieldAngle / 2);
                p = foval*sin(Theta_sphere);
                break;
            default:
                break;
                //cout << "The camera mode hasn't been choose!" << endl;
        }
#else
        foval = radius / (camerFieldAngle / 2);
        //printf("foval: %f \n", foval);
        p = foval*Theta_sphere;
#endif
        //Convert from parameter sphere cooradinate to fish-eye polar cooradinate
        //p = sin(Theta_sphere);
        theta = Phi_sphere;

        //Convert from fish-eye polar cooradinate to cartesian cooradinate
        // todo 插值
        x_cart = p*cos(theta);
        y_cart = p*sin(theta);

        //float R = radius / sin(camerFieldAngle / 2);

        //Convert from cartesian cooradinate to image cooradinate
        u = x_cart + center_x;
        v = -y_cart + center_y;
        //printf(" u: %d, v: %d, j: %d, i: %d\n ", u, v, j, i);

        if ( u >= 0 && u <= height && v >= 0 && v <= width) {
#if 1
            (xy + j * width * 2)[i * 2 + 0] = u; 
            (xy + j * width * 2)[i * 2 + 1] = v; 
#endif
            for (int k = 0; k < 3; k++)
                (out + j * width * 3)[i * 3 + k] = (in + v * width * 3)[u * 3 + k];
        }
    }
}

int latitudeCorrectionf(void *src, void *dest,  int radius, float w_longtitude, float w_latitude,  distMapMode distMap, float theta_left, float phi_up, float camerFieldAngle, camMode camProjMode, int w, int h)
{
    if (!(camerFieldAngle > 0 && camerFieldAngle <= CUDART_PI_F))
    {
        printf( "The parameter \"camerFieldAngle\" must be in the interval (0,CUDART_PI_F].");
        return -1;
    }

    char *in = (char *)src;
    char *out = (char *)dest;

    float rateOfWindow = 0.9;

    dim3 threadsPerBlock(16, 32);
    dim3 blocks((w + threadsPerBlock.x - 1)/threadsPerBlock.x, (h + threadsPerBlock.y - 1)/threadsPerBlock.y);
    cudaStream_t stream;

    //    convertIntToFloatKernel<<<blocks, threadsPerBlock>>>(pDevPtr, width,
    //                height, cuda_buf, pitch, show_buf);

//    printf("w:%d, h: %d ---- \n", w, h);
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "!!! cannot create stream\n");
        return -1;
    }

//    printf("w:%d, h: %d w_latitude: %lf, w_longtitude: %lf, camerFieldAngle: %lf-===\n", w, h, w_latitude, w_longtitude, camerFieldAngle);
    gettimeofday(&tv0, &tz);
#if 1
    if (xy == NULL)
    {
        cudaError_t status = cudaMallocHost((void **)&xy, sizeof(int) * w * h * 2);
        if (status != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(status));
#endif
        latitudeCorrectionfKernel<<<blocks, threadsPerBlock, 0, stream>>>
            (src, dest, w, h, w_latitude, w_longtitude,  camerFieldAngle, xy);
#if 1
    } else {
        xytouv<<<blocks, threadsPerBlock, 0, stream>>>
                        (src, dest, w, h, xy);
    }
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    gettimeofday(&tv1, &tz);
    printf("\n latitudeCorrection4 kernel running Cost time :  %lu us\n", tv1.tv_usec - tv0.tv_usec);

    return 0;
}

__global__ void
cuda_remapKernel(char *src, short *dst, float *xmap, float *ymap, int w, int h, int inw, int inh)
{
    char *in = (char *)src;
    short *out = (short *)dst;

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int inwidthstep = getWidth(inw, 3);
    int widthstep = getWidth(w, 3);

	if (i == 0 && j == 0)
		printf("instep: %d, step: %d\n", inwidthstep, widthstep);
#if 1
    if (i < w && j < h) {

        float x = xmap[j * w + i];
        float y = ymap[j * w + i];
        const int x1 = (int)x;
        const int y1 = (int)y;
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;

//	if (x1 > inw - 1 || y1 > inh - 1)
//		printf("x1: %d, y1: %d\t", x1, y1);
#if 1
	//if ((x1 < (inw - 5))  && (y1 < (inh - 5))){
	if ((x1 < 1000)  && (y1 < 1000)){
        char src_b = src[y1 * inwidthstep + x1 * 3 + 0];
        char src_g = src[y1 * inwidthstep + x1 * 3 + 1];
        char src_r = src[y1 * inwidthstep + x1 * 3 + 2];
	
	if (i > 500 && j > 500 && i < 600 && y < 600)
		printf("(%d, %d) -> (%d, %d) : (%d, %d, %d)\n", i, j, x1, y1, src_b, src_g, src_r);

#if 0
        src_r = src_r + src_r * ((x2 - x) * (y2 - y));

        src_reg = src(y1, x2);
        out = out + src_reg * ((x - x1) * (y2 - y));

        src_reg = src(y2, x1);
        out = out + src_reg * ((x2 - x) * (y - y1));

        src_reg = src(y2, x2);
        out = out + src_reg * ((x - x1) * (y - y1));

        (out + (j) * widthstep )[(i + dx) * 3 + 0] += (in + j * inwidthstep)[i * 3 + 0] * (weight + j * w)[i];
        (out + (j) * widthstep )[(i + dx) * 3 + 1] += (in + j * inwidthstep)[i * 3 + 1] * (weight + j * w)[i];
        (out + (j) * widthstep )[(i + dx) * 3 + 2] += (in + j * inwidthstep)[i * 3 + 2] * (weight + j * w)[i];
#endif
        *(out + j * widthstep + i * 3 + 0) = src_b ;
        *(out + j * widthstep + i * 3 + 1) = src_g ;
        *(out + j * widthstep + i * 3 + 2) = src_r ;
	} else
		printf(" out of memory , x1: %d, y1: %d\n", x1, y1);
#endif
    }
#endif
}

int cuda_remap(char *src, short *dst, float *xmap, float *ymap, int w, int h, int inw, int inh)
{
    dim3 threadsPerBlock(16, 8);
    dim3 blocks((w + threadsPerBlock.x - 1)/threadsPerBlock.x, (h + threadsPerBlock.y - 1)/threadsPerBlock.y);
    cudaStream_t stream;

    //    convertIntToFloatKernel<<<blocks, threadsPerBlock>>>(pDevPtr, width,
    //                height, cuda_buf, pitch, show_buf);

    //    printf("w:%d, h: %d ---- \n", w, h);
#if 0
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "!!! cannot create stream\n");
        return -1;
    }
#endif

    //    printf("w:%d, h: %d w_latitude: %lf, w_longtitude: %lf, camerFieldAngle: %lf-===\n", w, h, w_latitude, w_longtitude, camerFieldAngle);
    gettimeofday(&tv0, &tz);
	printf("debug:  %d\n", __LINE__);

    cuda_remapKernel<<<blocks, threadsPerBlock>>>
        (src, dst, xmap, ymap, w, h, inw, inh);

	cudaThreadSynchronize();
	printf("debug:  %d\n", __LINE__);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

	printf("debug:  %d\n", __LINE__);
//    cudaStreamSynchronize(stream);
//    cudaStreamDestroy(stream);
    gettimeofday(&tv1, &tz);
    printf("\n %s kernel running Cost time :  %lu us\n", __func__, tv1.tv_usec - tv0.tv_usec);

    return 0;


}

__global__ void
featherBlender_feed(short *src, short *dest, float *weight, int w, int h, int dw, int dh, int dx, int dy)
{
    short *in = (short *)src;
    short *out = (short *)dest;

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dwidthstep = getWidth(dw, 3);
    int widthstep = getWidth(w, 3);

    if (i < w && j < h) {
            (out + (j + dy) * dwidthstep )[(i + dx) * 3 + 0] += (in + j * widthstep)[i * 3 + 0] * (weight + j * w)[i];
            (out + (j + dy) * dwidthstep )[(i + dx) * 3 + 1] += (in + j * widthstep)[i * 3 + 1] * (weight + j * w)[i];
            (out + (j + dy) * dwidthstep )[(i + dx) * 3 + 2] += (in + j * widthstep)[i * 3 + 2] * (weight + j * w)[i];
    }

}

int feather_blender_feed(short *src, short *dest, float *weight, int w, int h, int dw, int dh, int dx, int dy)
{

    dim3 threadsPerBlock(16,8);
    dim3 blocks((w + threadsPerBlock.x - 1)/threadsPerBlock.x, (h + threadsPerBlock.y - 1)/threadsPerBlock.y);
    cudaStream_t stream;

    //    convertIntToFloatKernel<<<blocks, threadsPerBlock>>>(pDevPtr, width,
    //                height, cuda_buf, pitch, show_buf);

    //    printf("w:%d, h: %d ---- \n", w, h);
#if 0
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "!!! cannot create stream\n");
        return -1;
    }
#endif

    //    printf("w:%d, h: %d w_latitude: %lf, w_longtitude: %lf, camerFieldAngle: %lf-===\n", w, h, w_latitude, w_longtitude, camerFieldAngle);
    gettimeofday(&tv0, &tz);

    featherBlender_feed<<<blocks, threadsPerBlock, 0>>>
        (src, dest, weight, w, h, dw, dh, dx, dy);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

//    cudaStreamSynchronize(stream);
//    cudaStreamDestroy(stream);
    gettimeofday(&tv1, &tz);
    printf("\n %s kernel running Cost time :  %lu us\n", __func__, tv1.tv_usec - tv0.tv_usec);

    return 0;
}

__global__ void
featherBlender_blend(short *src, float *weight, int w, int h)
{
    short *out = (short *)src;

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int widthstep = (w + 3) / 4 * 4 * 3;

    if (i < w && j < h) {
            if (i < 600 && j < 600 && i > 500 && j > 500)
                printf("<%d, %d> ->  <%04d, %08f>\n", i, j, (out + j * w * 3)[i * 3 + 0] , ((weight + j * w)[i] + WEIGHT_EPS) );
            (out + j * widthstep)[i * 3 + 0] = (out + j * widthstep)[i * 3 + 0] / ((weight + j * w)[i] + WEIGHT_EPS);
            (out + j * widthstep)[i * 3 + 1] = (out + j * widthstep)[i * 3 + 1] / ((weight + j * w)[i] + WEIGHT_EPS);
            (out + j * widthstep)[i * 3 + 2] = (out + j * widthstep)[i * 3 + 2] / ((weight + j * w)[i] + WEIGHT_EPS);
    }

}

int feather_blender_blend(short *src, float *weight, int w, int h)
{
    dim3 threadsPerBlock(16, 8);
    dim3 blocks((w + threadsPerBlock.x - 1)/threadsPerBlock.x, (h + threadsPerBlock.y - 1)/threadsPerBlock.y);
    cudaStream_t stream;

    cudaEvent_t start, stop;
    float elapsedTime;

    //    convertIntToFloatKernel<<<blocks, threadsPerBlock>>>(pDevPtr, width,
    //                height, cuda_buf, pitch, show_buf);

//    printf("w:%d, h: %d ---- \n", w, h);
#if 0
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "!!! cannot create stream\n");
        return -1;
    }
#endif
//    printf("w:%d, h: %d w_latitude: %lf, w_longtitude: %lf, camerFieldAngle: %lf-===\n", w, h, w_latitude, w_longtitude, camerFieldAngle);
    gettimeofday(&tv0, &tz);

    (cudaEventCreate(&start));
    (cudaEventCreate(&stop));

    (cudaEventRecord(start, 0));

    featherBlender_blend<<<blocks, threadsPerBlock, 0>>>
	    (src, weight, w, h);

    (cudaEventRecord(stop, 0));
    (cudaEventSynchronize(stop));
    (cudaEventElapsedTime(&elapsedTime, start, stop));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

    //    cudaStreamSynchronize(stream);
    //    cudaStreamDestroy(stream);
    gettimeofday(&tv1, &tz);
    printf("\n %s kernel running Cost time :  %lu us, w: %d, h : %d\n", __func__, tv1.tv_usec - tv0.tv_usec, w, h);

    (cudaEventDestroy(start));
    (cudaEventDestroy(stop));
    return 0;
}
