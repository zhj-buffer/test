#include "image.h"

typedef struct blender{
   int type;
   bool try_gpu;
   img dst_;
   img dst_mask_;
   rect dst_roi_;

   //multi blender
   int actual_num_bands_, num_bands_;
   img *dst_pyr_laplace_;
   img *dst_band_weights_;
   rect dst_roi_final_;
   bool can_use_gpu_;
   int weight_type_; //CV_32F or CV_16S
}blender;


static blender bl;

static rect resultRoi(point *p, int p_num; size *s, int s_num)
{
    if (p_num != s_num) {
        printf("ERROR: %s, %d", __func__, __LINE__);
    }
    rect rect;

    point tl, br;
    tl.x = UINT_MAX;
    tl.y = UINT_MAX;
    br.x = 0;
    br.y = 0;

    for (int i = 0; i < p_num; i++) 
    {
        tl.x = tl.x > p[i].x ? tl.x:p[i].x;
        tl.y = tl.y > p[i].y ? tl.y:p[i].y;

        br.x = br.x < (p[i].x + s[i].width)
        br.y = br.y < (p[i].y + s[i].height)
    }

    rect.x = tl.x;
    rect.y = tl.y;
    rect.width = br.x;
    rect.height = br.y;

    return rect;
} // fixed


static void setNumBands(int num_bands)
{
    bl.actual_num_bands_ = num_bands;
}// fixed


static void create(int try_gpu, int num_bands, int weight_typ)
{

    setNumBands(num_bands);
    if (try_gpu)
        bl.can_use_gpu = true;
    else
        bl.can_use_gpu = false;

    bl.weight_type_ = 4; // CV_32F
}// fixed

static void prepare(rect dst_roi)
{

    cudaError_t status;

    bl.dst_roi_final_.x = dst_roi.x;
    bl.dst_roi_final_.y = dst_roi.y;
    bl.dst_roi_final_.width = dst_roi.width;
    bl.dst_roi_final_.height = dst_roi.height;

    double max_len = (double)(dst_roi.width > dst_roi.height ? dst_roi.width:dst_roi.height);
    bl.num_bands_ = min(bl.actual_num_bands_ , (int)(ceil(log(max_len) / log(2.0))));

	// Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << bl.num_bands_) - dst_roi.width % (1 << bl.num_bands_)) % (1 << bl.num_bands_); 
    dst_roi.height += ((1 << bl.num_bands_) - dst_roi.height % (1 << bl.num_bands_)) % (1 << bl.num_bands_);

    bl.dst_.w = dst_roi.width;
    bl.dst_.h = dst_roi.height;
    bl.dst_.c = 3; // CV_16SC3
    
    status = cudaMallocHost((void **)&bl.dst_.idata, bl.dst_.w * bl.dst_.h * bl.dst_.c * sizeof(int));

    bl.dst_mask_.w = dst_roi.width;
    bl.dst_mask_.h = dst_roi.height;
    bl.dst_mask_.c = 1; // CV_8U
    
    status = cudaMallocHost((void **)&bl.dst_mask_.idata, bl.dst_mask_.w * bl.dst_maks_.h * bl.dst_mask_.c * sizeof(unsigned char));

    bl.dst_roi_.w = dst_roi.w;
    bl.dst_roi_.h = dst_roi.h;
    bl.dst_roi_.width = dst_roi.width;
    bl.dst_roi_.height = dst_roi.height;

    bl.dst_pyr_laplace_ = calloc(bl.num_bands_ + 1, sizeof(img));
    bl.dst_pyr_laplace_[0].w = bl.dst_.w;
    bl.dst_pyr_laplace_[0].h = bl.dst_.h;
    bl.dst_pyr_laplace_[0].c = bl.dst_.c;
    bl.dst_pyr_laplace_[0].idata = bl.dst_.idata;

    bl.dst_band_weights_ = calloc(bl.num_bands_ + 1, sizeof(img));
    bl.dst_band_weights_[0].w = dst_roi.width;
    bl.dst_band_weights_[0].h = dst_roi.height;
    bl.dst_band_weights_[0].c = weight_type_; // CV_32F
    bl.dst_band_weights_[0].idata = bl.dst_.idata;


    for (int i = 1; i <= bl.num_bands_; i++) {
        status = cudaMallocHost((void **)&bl.dst_pyr_laplace_[i]. );
    }
}


