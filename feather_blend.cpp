#include "feather_blend.h"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/base.hpp"
#include <stdio.h>
#include <iostream>
#include "blender_cuda.cuh"
using namespace std;

static const float WEIGHT_EPS = 1e-5f;
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

int MatAddProcessGpu::MatAddPointProcess(Rect rec, Mat src_laplace, Mat weigth_gauss, Mat & dst_laplace, Mat & dst_weigth)
{
    //malloc gpu memory

    int size = rec.width*rec.height * 3 * sizeof(short);
    int size_w = rec.width*rec.height * sizeof(float);
    printf("FILE(%s)_FUNCTION(%s)__LINE(%d):---width(%d)--height(%d)----\n", __FILE__, __FUNCTION__, __LINE__, rec.width, rec.height);
    cudaError_t status;
    short* src_laplace_d = NULL;
    short* dst_laplace_d = NULL;
    float* src_weight_d = NULL;
    float* dst_weight_d = NULL;
    int err;

    printf("FILE(%s)_FUNCTION(%s)__LINE(%d):---size(%d)--size_w(%d)---\n", __FILE__, __FUNCTION__, __LINE__, size, size_w);
    //host 
    status = cudaMalloc((void**)(&src_laplace_d), size);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "------cudaMalloc failed(%d)!\n", size);
        return -1;
    }

    status = cudaMalloc((void**)(&dst_laplace_d), size);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "------cudaMalloc failed(%d)!\n", size);
        return -1;
    }

    status = cudaMalloc((void**)(&src_weight_d), size_w);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "------cudaMalloc failed(%d)!\n", size_w);
        return -1;
    }

    status = cudaMalloc((void**)(&dst_weight_d), size_w);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "------cudaMalloc failed(%d)!\n", size_w);
        return -1;
    }

    //printf("FILE(%s)_FUNCTION(%s)__LINE(%d):------\n", __FILE__, __FUNCTION__, __LINE__);

    cudaMemcpy(src_laplace_d, src_laplace.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(src_weight_d, weigth_gauss.data, size_w, cudaMemcpyHostToDevice);

#if 0
    for (int i = 0; i < rec.width * rec.height * 3; i += 3)
        printf("-------xy %d: %d %d %d\n", i, ((short*)src_laplace.data)[i], ((short *)src_laplace.data)[i + 1], ((short*)src_laplace.data)[i + 2]);
#endif

    MatAddEx((short)rec.width, (short)rec.height, src_laplace_d, src_weight_d, dst_laplace_d, dst_weight_d);


    //printf("FILE(%s)_FUNCTION(%s)__LINE(%d):------\n", __FILE__, __FUNCTION__, __LINE__);
    //copy device mem to host


    cudaMemcpy(dst_laplace.data, dst_laplace_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_weigth.data, dst_weight_d, size_w, cudaMemcpyDeviceToHost);

    
    cudaDeviceSynchronize();

    cudaFree(src_laplace_d);
    cudaFree(src_weight_d);
    cudaFree(dst_laplace_d);
    cudaFree(dst_weight_d);

    return 0;
}


void BlenderPrepare(const  std::vector < Point > & corners, const std::vector < Size > & sizes, UMat & dst_, UMat & dst_mask_, Rect & dst_roi_)
{
    BlenderPrepareEx(resultRoi(corners, sizes), dst_, dst_mask_, dst_roi_);
}

void BlenderPrepareEx(Rect dst_roi, UMat & dst_, UMat & dst_mask_, Rect & dst_roi_)
{
    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;
}

void BlenderFeed(InputArray _img, InputArray _mask, Point tl, UMat dst_, UMat dst_mask_, Rect dst_roi_)
{
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();
    Mat dst = dst_.getMat(ACCESS_RW);
    Mat dst_mask = dst_mask_.getMat(ACCESS_RW);

    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst.ptr<Point3_<short> >(dy + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask.ptr<uchar>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x])
                dst_row[dx + x] = src_row[x];
            dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}

void BlenderBlend(InputOutputArray dst, InputOutputArray dst_mask, UMat dst_, UMat dst_mask_)
{
    UMat mask;
    compare(dst_mask_, 0, mask, CMP_EQ);
    dst_.setTo(Scalar::all(0), mask);
    dst.assign(dst_);
    dst_mask.assign(dst_mask_);
    dst_.release();
    dst_mask_.release();
}

void FeatherBlanderInit(FeatherInfo_t* info, float sharp_ness /* = 0.02f */)
{
    info->sharp_ness = sharp_ness;
}

void FeatherBlanderPrepare(FeatherInfo_t* info,Rect dst_roi)
{
    BlenderPrepareEx(dst_roi, info->dst_, info->dst_mask_, info->dst_roi_);
    info->dst_weight_map_.create(dst_roi.size(),CV_32F);
    info->dst_weight_map_.setTo(0);
}

void FeatherBlanderFeed(FeatherInfo_t info, InputArray _img, InputArray mask, Point tl)
{
#if 1
    int64 t = getTickCount();
    Mat img = _img.getMat();
    Mat dst = info.dst_.getMat(ACCESS_RW);

    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    createWeightMap(mask, info.sharp_ness, info.weight_map_);
    Mat weight_map = info.weight_map_.getMat(ACCESS_READ);
    Mat dst_weight_map = info.dst_weight_map_.getMat(ACCESS_RW);
    LOGLN("-----------------createWeightMap, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    t = getTickCount();
    int dx = tl.x - info.dst_roi_.x;
    int dy = tl.y - info.dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short>* src_row = img.ptr<Point3_<short> >(y);
        Point3_<short>* dst_row = dst.ptr<Point3_<short> >(dy + y);
        const float* weight_row = weight_map.ptr<float>(y);
        float* dst_weight_row = dst_weight_map.ptr<float>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
            dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
            dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
            dst_weight_row[dx + x] += weight_row[x];
        }
    }
    LOGLN("---------------add weight, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

#elif 0
    //Mat img = _img.getMat();
    //Mat dst = info.dst_.getMat(ACCESS_RW);

    CV_Assert(_img.getMat().type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    createWeightMap(mask, info.sharp_ness, info.weight_map_);
    //Mat weight_map = info.weight_map_.getMat(ACCESS_READ);
    //Mat dst_weight_map = info.dst_weight_map_.getMat(ACCESS_RW);

    int dx = tl.x - info.dst_roi_.x;
    int dy = tl.y - info.dst_roi_.y;

    for (int y = 0; y < _img.getMat().rows; ++y)
    {
        const Point3_<short>* src_row = _img.getMat().ptr<Point3_<short> >(y);
        Point3_<short>* dst_row = info.dst_.getMat(ACCESS_RW).ptr<Point3_<short> >(dy + y);
        const float* weight_row = info.weight_map_.getMat(ACCESS_READ).ptr<float>(y);
        float* dst_weight_row = info.dst_weight_map_.getMat(ACCESS_RW).ptr<float>(dy + y);

        for (int x = 0; x < _img.getMat().cols; ++x)
        {
            dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
            dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
            dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
            dst_weight_row[dx + x] += weight_row[x];
        }
    }
#endif
}

void FeatherBlanderBlend(FeatherInfo_t info, InputOutputArray dst, InputOutputArray dst_mask)
{
    normalizeUsingWeightMap(info.dst_weight_map_, info. dst_);
    compare(info.dst_weight_map_, WEIGHT_EPS, info.dst_mask_, CMP_GT);
    BlenderBlend(dst, dst_mask,info.dst_,info.dst_mask_);
}


void MultiBandBlender_init(MultiBandInfo & info, int try_gpu, int num_bands, int weight_type)
{
    info.actual_num_bands_ = num_bands;

    printf(" %s, %d, weight_type: %d, num_bands: %d\n", __func__, __LINE__, weight_type, num_bands);
#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    info.can_use_gpu_ = try_gpu && cuda::getCudaEnabledDeviceCount();
#else
    (void)try_gpu;
    info.can_use_gpu_ = false;
#endif

    CV_Assert(weight_type == CV_32F || weight_type == CV_16S);
    info.weight_type_ = weight_type;
}

void MultiBandBlender_prepare(MultiBandInfo & info, Rect dst_roi)
{
    info.dst_roi_final_ = dst_roi;


    // Crop unnecessary bands
    double max_len = static_cast<double>(std::max(dst_roi.width, dst_roi.height));
    info.num_bands_ = std::min(info.actual_num_bands_, static_cast<int>(ceil(std::log(max_len) / std::log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << info.num_bands_) - dst_roi.width % (1 << info.num_bands_)) % (1 << info.num_bands_);
    dst_roi.height += ((1 << info.num_bands_) - dst_roi.height % (1 << info.num_bands_)) % (1 << info.num_bands_);

    cout << "dst_roi: " << dst_roi << "num_bands_: " << info.num_bands_ <<  endl;
    BlenderPrepareEx(dst_roi, info.dst_, info.dst_mask_, info.dst_roi_);

    info.dst_pyr_laplace_.resize(info.num_bands_ + 1);
    info.dst_pyr_laplace_[0] = info.dst_;

    info.dst_band_weights_.resize(info.num_bands_ + 1);
    info.dst_band_weights_[0].create(dst_roi.size(), info.weight_type_);
    info.dst_band_weights_[0].setTo(0);

    for (int i = 1; i <= info.num_bands_; ++i)
    {
        info.dst_pyr_laplace_[i].create((info.dst_pyr_laplace_[i - 1].rows + 1) / 2,
                (info.dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
        info.dst_band_weights_[i].create((info.dst_band_weights_[i - 1].rows + 1) / 2,
                (info.dst_band_weights_[i - 1].cols + 1) / 2, info.weight_type_);
        info.dst_pyr_laplace_[i].setTo(Scalar::all(0));
        info.dst_band_weights_[i].setTo(0);
    }
}

void MultiBandBlender_feed(MultiBandInfo info, InputArray _img, InputArray _mask, Point tl)
{
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    UMat img = _img.getUMat();

    cout << "type: " << img.type()  << " CV_16SC3: " << CV_16SC3 << endl;
    CV_Assert(img.type() == CV_16SC3 || img.type() == CV_8UC3);
    CV_Assert(_mask.type() == CV_8U);

    // Keep source image in memory with small border
    int gap = 3 * (1 << info.num_bands_);
    Point tl_new(std::max(info.dst_roi_.x, tl.x - gap),
            std::max(info.dst_roi_.y, tl.y - gap));
    Point br_new(std::min(info.dst_roi_.br().x, tl.x + img.cols + gap),
            std::min(info.dst_roi_.br().y, tl.y + img.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
    // After that scale between layers is exactly 2.
    //
    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
    tl_new.x = info.dst_roi_.x + (((tl_new.x - info.dst_roi_.x) >> info.num_bands_) << info.num_bands_);
    tl_new.y = info.dst_roi_.y + (((tl_new.y - info.dst_roi_.y) >> info.num_bands_) << info.num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << info.num_bands_) - width % (1 << info.num_bands_)) % (1 << info.num_bands_);
    height += ((1 << info.num_bands_) - height % (1 << info.num_bands_)) % (1 << info.num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = std::max(br_new.y - info.dst_roi_.br().y, 0);
    int dx = std::max(br_new.x - info.dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - img.rows;
    int right = br_new.x - tl.x - img.cols;

    // Create the source image Laplacian pyramid
    UMat img_with_border;
    copyMakeBorder(_img, img_with_border, top, bottom, left, right,
            BORDER_REFLECT);
    LOGLN("  Add border to the source image, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#if ENABLE_LOG
    t = getTickCount();
#endif

    std::vector<UMat> src_pyr_laplace;
    if (info.can_use_gpu_ && img_with_border.depth() == CV_16S)
        createLaplacePyrGpu(img_with_border, info.num_bands_, src_pyr_laplace);
    else
        createLaplacePyr(img_with_border, info.num_bands_, src_pyr_laplace);

    LOGLN("  Create the source image Laplacian pyramid, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#if ENABLE_LOG
    t = getTickCount();
#endif

    // Create the weight map Gaussian pyramid
    UMat weight_map;
    std::vector<UMat> weight_pyr_gauss(info.num_bands_ + 1);

    if (info.weight_type_ == CV_32F)
    {
        _mask.getUMat().convertTo(weight_map, CV_32F, 1. / 255.);
    }
    else // weight_type_ == CV_16S
    {
        _mask.getUMat().convertTo(weight_map, CV_16S);
        UMat add_mask;
        compare(_mask, 0, add_mask, CMP_NE);
        add(weight_map, Scalar::all(1), weight_map, add_mask);
    }

    copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, BORDER_CONSTANT);

    for (int i = 0; i < info.num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    LOGLN("  Create the weight map Gaussian pyramid, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#if ENABLE_LOG
    t = getTickCount();
#endif

    int y_tl = tl_new.y - info.dst_roi_.y;
    int y_br = br_new.y - info.dst_roi_.y;
    int x_tl = tl_new.x - info.dst_roi_.x;
    int x_br = br_new.x - info.dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    for (int i = 0; i <= info.num_bands_; ++i)
    {
        Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
#ifdef HAVE_OPENCL
        if (!cv::ocl::useOpenCL() ||
                !ocl_MultiBandBlender_feed(src_pyr_laplace[i], weight_pyr_gauss[i],
                    info.dst_pyr_laplace_[i](rc), info.dst_band_weights_[i](rc)))
#endif
        {
#if 1
            Mat _src_pyr_laplace = src_pyr_laplace[i].getMat(ACCESS_READ);
            Mat _dst_pyr_laplace = info.dst_pyr_laplace_[i](rc).getMat(ACCESS_RW);
            Mat _weight_pyr_gauss = weight_pyr_gauss[i].getMat(ACCESS_READ);
            Mat _dst_band_weights = info.dst_band_weights_[i](rc).getMat(ACCESS_RW);
#else
            Mat _src_pyr_laplace = (src_pyr_laplace[i].getMat(ACCESS_READ)).clone();
            Mat _dst_pyr_laplace = (info.dst_pyr_laplace_[i](rc).getMat(ACCESS_RW));
            Mat _weight_pyr_gauss = (weight_pyr_gauss[i].getMat(ACCESS_READ)).clone();
            Mat _dst_band_weights = (info.dst_band_weights_[i](rc).getMat(ACCESS_RW));
#endif

            if (info.weight_type_ == CV_32F)
            {
                //#ifdef HAVE_CUDA
#if 1
#if 1
                printf("-------_src_pyr_laplace type(%d)-size(%d)---_weight_pyr_gauss type(%d)-size(%d)--\n", _src_pyr_laplace.type(), \
                        _weight_pyr_gauss.type());
                std::cout << "_src_pyr_laplace:" << _src_pyr_laplace.size() <<"is continuous:"<<_src_pyr_laplace.isContinuous()\
                    <<"step[0]"<< _src_pyr_laplace.step1(0) << "step[1]" << _src_pyr_laplace.step1(1) << "step[2]" << _src_pyr_laplace.step1(2) << std::endl;

                std::cout << "_weight_pyr_gauss.size()" << _weight_pyr_gauss.size() << "is continuous:" << _src_pyr_laplace.isContinuous()\
                    << "step[0]" << _src_pyr_laplace.step1(0) << "step[1]" << _src_pyr_laplace.step1(1) << "step[2]" << _src_pyr_laplace.step1(2) << std::endl;



                printf("dst weight ----(%f)\n", *((float*)_dst_band_weights.data));
#endif
#if 1
                for (int y = 0; y < rc.height; ++y)
                {
                    const Point3_<short>* src_row = _src_pyr_laplace.ptr<Point3_<short> >(y);
                    Point3_<short>* dst_row = _dst_pyr_laplace.ptr<Point3_<short> >(y);
                    const float* weight_row = _weight_pyr_gauss.ptr<float>(y);
                    float* dst_weight_row = _dst_band_weights.ptr<float>(y);

                    for (int x = 0; x < rc.width; ++x)
                    {
                        dst_row[x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                        dst_row[x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                        dst_row[x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                        dst_weight_row[x] += weight_row[x];
#if 0
                        if (x < 5 && y < 5) {
                            std::cout << "xy<" << x << "," << y << ">:" << src_row[x].x << " " << src_row[x].y << " " << src_row[x].z << std::endl;
                            std::cout << "weiht[" << x << "]:" << weight_row[x] << std::endl;
                        }
#endif

            
                    }

                }
#if 0
                for (int y = 0; y < rc.height; ++y)
                    for (int x = 0; x < rc.width; ++x)
                        printf("<%02d,%02d> %08f\n",x, y, ((float *)(_dst_band_weights.data))[y * rc.width + x]);

                printf("\n===========================================\n");
#endif
#endif
#if 0

                MatAddProcessGpu::MatAddPointProcess(rc, _src_pyr_laplace, _weight_pyr_gauss, _dst_pyr_laplace, _dst_band_weights);

                for (int y = 0; y < rc.height; ++y)
                    for (int x = 0; x < rc.width; ++x)
                        printf("<%02d,%02d> %08f\n",x, y, ((float *)(_dst_band_weights.data))[y * rc.width + x]);
                        //printf("<%02d,%02d> %08f\t",x, y, _dst_band_weights.data[y * rc.width + x]);
                        //printf("<%02d,%02d> %08f\t",x, y, dst_weight_row[y * rc.width + x]);
#endif
#else

                for (int y = 0; y < rc.height; ++y)
                {
                    const Point3_<short>* src_row = _src_pyr_laplace.ptr<Point3_<short> >(y);
                    Point3_<short>* dst_row = _dst_pyr_laplace.ptr<Point3_<short> >(y);
                    const float* weight_row = _weight_pyr_gauss.ptr<float>(y);
                    float* dst_weight_row = _dst_band_weights.ptr<float>(y);

                    for (int x = 0; x < rc.width; ++x)
                    {
                        dst_row[x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                        dst_row[x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                        dst_row[x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                        dst_weight_row[x] += weight_row[x];
                    }
                }
#endif
            }
            else // weight_type_ == CV_16S
            {
                for (int y = 0; y < y_br - y_tl; ++y)
                {
                    const Point3_<short>* src_row = _src_pyr_laplace.ptr<Point3_<short> >(y);
                    Point3_<short>* dst_row = _dst_pyr_laplace.ptr<Point3_<short> >(y);
                    const short* weight_row = _weight_pyr_gauss.ptr<short>(y);
                    short* dst_weight_row = _dst_band_weights.ptr<short>(y);

                    for (int x = 0; x < x_br - x_tl; ++x)
                    {
                        dst_row[x].x += short((src_row[x].x * weight_row[x]) >> 8);
                        dst_row[x].y += short((src_row[x].y * weight_row[x]) >> 8);
                        dst_row[x].z += short((src_row[x].z * weight_row[x]) >> 8);
                        dst_weight_row[x] += weight_row[x];
                    }
                }
            }
        }
#ifdef HAVE_OPENCL
        else
        {
            CV_IMPL_ADD(CV_IMPL_OCL);
        }
#endif

        x_tl /= 2; y_tl /= 2;
        x_br /= 2; y_br /= 2;
    }

    LOGLN("  Add weighted layer of the source image to the final Laplacian pyramid layer, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

}

void MultiBandBlender_blend(MultiBandInfo info, InputOutputArray dst, InputOutputArray dst_mask)
{
    for (int i = 0; i <= info.num_bands_; ++i)
        normalizeUsingWeightMap(info.dst_band_weights_[i], info.dst_pyr_laplace_[i]);

    if (info.can_use_gpu_)
        restoreImageFromLaplacePyrGpu(info.dst_pyr_laplace_);
    else
        restoreImageFromLaplacePyr(info.dst_pyr_laplace_);

    Rect dst_rc(0, 0, info.dst_roi_final_.width, info.dst_roi_final_.height);
    info.dst_ = info.dst_pyr_laplace_[0](dst_rc);
    UMat _dst_mask;
    compare(info.dst_band_weights_[0](dst_rc), WEIGHT_EPS, info.dst_mask_, CMP_GT);
    info.dst_pyr_laplace_.clear();
    info.dst_band_weights_.clear();

    BlenderBlend(dst, dst_mask, info.dst_, info.dst_mask_);
}
