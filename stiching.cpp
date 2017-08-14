

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//
//M*/

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
//#define HAVE_OPENCV_CUDAWARPING 1
//#define HAVE_OPENCV_XFEATURES2D 1
#include "feather_blend.h"
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <cv.h>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <iostream>

#include "image_core.h"

//#define CAMERA_CAL

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace cv::cuda;
//using namespace cv::cuda::device;

static void printUsage()
{
	cout <<
		"Rotation model images stitcher.\n\n"
		"stitching_detailed img1 img2 [...imgN] [flags]\n\n"
		"Flags:\n"
		"  --preview\n"
		"      Run stitching in the preview mode. Works faster than usual mode,\n"
		"      but output image will have lower resolution.\n"
		"  --try_cuda (yes|no)\n"
		"      Try to use CUDA. The default value is 'no'. All default values\n"
		"      are for CPU mode.\n"
		"\nMotion Estimation Flags:\n"
		"  --work_megapix <float>\n"
		"      Resolution for image registration step. The default is 0.6 Mpx.\n"
		"  --features (surf|orb)\n"
		"      Type of features used for images matching. The default is surf.\n"
		"  --matcher (homography|affine)\n"
		"      Matcher used for pairwise image matching.\n"
		"  --estimator (homography|affine)\n"
		"      Type of estimator used for transformation estimation.\n"
		"  --match_conf <float>\n"
		"      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
		"  --conf_thresh <float>\n"
		"      Threshold for two images are from the same panorama confidence.\n"
		"      The default is 1.0.\n"
		"  --ba (no|reproj|ray|affine)\n"
		"      Bundle adjustment cost function. The default is ray.\n"
		"  --ba_refine_mask (mask)\n"
		"      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
		"      where 'x' means refine respective parameter and '_' means don't\n"
		"      refine one, and has the following format:\n"
		"      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
		"      adjustment doesn't support estimation of selected parameter then\n"
		"      the respective flag is ignored.\n"
		"  --wave_correct (no|horiz|vert)\n"
		"      Perform wave effect correction. The default is 'horiz'.\n"
		"  --save_graph <file_name>\n"
		"      Save matches graph represented in DOT language to <file_name> file.\n"
		"      Labels description: Nm is number of matches, Ni is number of inliers,\n"
		"      C is confidence.\n"
		"\nCompositing Flags:\n"
		"  --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
		"      Warp surface type. The default is 'spherical'.\n"
		"  --seam_megapix <float>\n"
		"      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
		"  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
		"      Seam estimation method. The default is 'gc_color'.\n"
		"  --compose_megapix <float>\n"
		"      Resolution for compositing step. Use -1 for original resolution.\n"
		"      The default is -1.\n"
		"  --expos_comp (no|gain|gain_blocks)\n"
		"      Exposure compensation method. The default is 'gain_blocks'.\n"
		"  --blend (no|feather|multiband)\n"
		"      Blending method. The default is 'multiband'.\n"
		"  --blend_strength <float>\n"
		"      Blending strength from [0,100] range. The default is 5.\n"
		"  --output <result_img>\n"
		"      The default is 'result.jpg'.\n"
		"  --timelapse (as_is|crop) \n"
		"      Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
		"  --rangewidth <int>\n"
		"      uses range_width to limit number of images to match with.\n";
}



// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = true;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "surf";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "cylindrical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "gc_color";
//int blend_type = Blender::MULTI_BAND;
int blend_type = Blender::FEATHER;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.png";
bool timelapse = false;
int range_width = -1;


static int parseCmdArgs(int argc, char** argv)
{
	if (argc == 1)
	{
		printUsage();
		return -1;
	}
	for (int i = 1; i < argc; ++i)
	{
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
		{
			printUsage();
			return -1;
		}
		else if (string(argv[i]) == "--preview")
		{
			preview = true;
		}
		else if (string(argv[i]) == "--try_cuda")
		{
			if (string(argv[i + 1]) == "no")
				try_cuda = false;
			else if (string(argv[i + 1]) == "yes")
				try_cuda = true;
			else
			{
				cout << "Bad --try_cuda flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--work_megapix")
		{
			work_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--seam_megapix")
		{
			seam_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--compose_megapix")
		{
			compose_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--result")
		{
			result_name = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--features")
		{
			features_type = argv[i + 1];
			if (features_type == "orb")
				match_conf = 0.3f;
			i++;
		}
		else if (string(argv[i]) == "--matcher")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				matcher_type = argv[i + 1];
			else
			{
				cout << "Bad --matcher flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--estimator")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				estimator_type = argv[i + 1];
			else
			{
				cout << "Bad --estimator flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--match_conf")
		{
			match_conf = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--conf_thresh")
		{
			conf_thresh = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--ba")
		{
			ba_cost_func = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--ba_refine_mask")
		{
			ba_refine_mask = argv[i + 1];
			if (ba_refine_mask.size() != 5)
			{
				cout << "Incorrect refinement mask length.\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--wave_correct")
		{
			if (string(argv[i + 1]) == "no")
				do_wave_correct = false;
			else if (string(argv[i + 1]) == "horiz")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_HORIZ;
			}
			else if (string(argv[i + 1]) == "vert")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_VERT;
			}
			else
			{
				cout << "Bad --wave_correct flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--save_graph")
		{
			save_graph = true;
			save_graph_to = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--warp")
		{
			warp_type = string(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--expos_comp")
		{
			if (string(argv[i + 1]) == "no")
				expos_comp_type = ExposureCompensator::NO;
			else if (string(argv[i + 1]) == "gain")
				expos_comp_type = ExposureCompensator::GAIN;
			else if (string(argv[i + 1]) == "gain_blocks")
				expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
			else
			{
				cout << "Bad exposure compensation method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--seam")
		{
			if (string(argv[i + 1]) == "no" ||
				string(argv[i + 1]) == "voronoi" ||
				string(argv[i + 1]) == "gc_color" ||
				string(argv[i + 1]) == "gc_colorgrad" ||
				string(argv[i + 1]) == "dp_color" ||
				string(argv[i + 1]) == "dp_colorgrad")
				seam_find_type = argv[i + 1];
			else
			{
				cout << "Bad seam finding method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--blend")
		{
			if (string(argv[i + 1]) == "no")
				blend_type = Blender::NO;
			else if (string(argv[i + 1]) == "feather")
				blend_type = Blender::FEATHER;
			else if (string(argv[i + 1]) == "multiband")
				blend_type = Blender::MULTI_BAND;
			else
			{
				cout << "Bad blending method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--timelapse")
		{
			timelapse = true;

			if (string(argv[i + 1]) == "as_is")
				timelapse_type = Timelapser::AS_IS;
			else if (string(argv[i + 1]) == "crop")
				timelapse_type = Timelapser::CROP;
			else
			{
				cout << "Bad timelapse method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--rangewidth")
		{
			range_width = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--blend_strength")
		{
			blend_strength = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--output")
		{
			result_name = argv[i + 1];
			i++;
		}
		else
			img_names.push_back(argv[i]);
	}
	if (preview)
	{
		compose_megapix = 0.6;
	}
	return 0;
}


int main(int argc, char* argv[])
{
#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif

#if 0
	cv::setBreakOnError(true);
#endif

	int retval = parseCmdArgs(argc, argv);
	if (retval)
		return retval;

	// Check if have enough images
	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

	LOGLN("Finding features...");
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

#ifdef CAMERA_CAL

	Ptr<FeaturesFinder> finder;
	if (features_type == "surf")
	{
#ifdef HAVE_OPENCV_XFEATURES2D
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
        {
            finder = makePtr<SurfFeaturesFinderGpu>();   
            printf("@@@@@@@@@################## use GPu feature finder!\n");
        }
		else
#endif
			finder = makePtr<SurfFeaturesFinder>();
	}
	else if (features_type == "orb")
	{
		finder = makePtr<OrbFeaturesFinder>();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images; ++i)
	{
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		if (full_img.empty())
		{
			LOGLN("Can't open image " << img_names[i]);
			return -1;
		}
		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
            cv::resize(full_img, img, Size(), work_scale, work_scale);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(img, features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

        LOGLN("full_img size: " << full_img.size() << "img size: "  << img.size() << "seam_scale: " << seam_scale << "work_scale: " << work_scale);
        cv::resize(full_img, img, Size(), seam_scale, seam_scale);
		images[i] = img.clone();
        LOGLN("full_img size: " << full_img.size() << "img size: " << img.size());
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	LOGLN("###################Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	LOG("Pairwise matching");
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher;
	if (matcher_type == "affine")
		matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
	else if (range_width == -1)
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
	else
		matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Check if we should save matches graph
	if (save_graph)
	{
		LOGLN("Saving matches graph...");
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	Ptr<Estimator> estimator;
	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
		estimator = makePtr<HomographyBasedEstimator>();

	vector<CameraParams> cameras;
	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed.\n";
		return -1;
	}

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
	else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
	else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return -1;
	}
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
		return -1;
	}

	// Find median focal length

	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		//LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}


#if 0
	vector<CameraParams> cameras;
    vector<int> indices;

	for (size_t i = 0; i < cameras.size(); ++i)
	{
        indices[i] = i;

        if (i = 0) {
#if 0
            Mat_<double> K;
            cameras[i].K().convertTo(K, CV_32F);;
            K(0,0) = 7588.419867119918;
            K(0,2) = 387.5;
            K(1,1) = 7588.419867119918;
            K(1,2) = 387.5;
            K(2,2) = 1;
#endif
            cameras[i].focal = 6778.71;
            cameras[i].aspect = 1;
            cameras[i].ppx = 387.5;
            cameras[i].ppy = 387.5;


            Mat_<float> R;
            cameras[i].R.convertTo(R, CV_32F);

            R(0,0) = 0.99948275;
            R(0,1) = -0.013908581;
            R(0,2) = 0.028993731;
            R(1,0) = -3.1985163e-09;
            R(1,1) = 0.90162504;
            R(1,2) = 0.43251848;
            R(2,0) = -0.03215719;
            R(2,1) = -0.43229476;
            R(2,2) = 0.90115869;
            cameras[i].R = R;

            LOGLN("fix: Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R <<"\nfocal: " <<cameras[i].focal);
        }
        if (i = 1) {
#if 0
            Mat_<double> K = cameras[i].K();
            K(0,0) = 8010.314989606631;
            K(0,2) = 387.5;
            K(1,1) = 8010.314989606631;
            K(1,2) = 387.5;
            K(2,2) = 1;
#endif
            cameras[i].focal = 7109.97;
            cameras[i].aspect = 1;
            cameras[i].ppx = 387.5;
            cameras[i].ppy = 387.5;
            Mat_<float> R;
            R.convertTo(R, CV_32F);
            R(0,0) = 0.99948335;
            R(0,1) = 0.013863774;
            R(0,2) = -0.028993733;
            R(1,0) = 0;
            R(1,1) = 0.9021681;
            R(1,2) = 0.43138468;
            R(2,0) = 0.032137856;
            R(2,1) = -0.43116182;
            R(2,2) = 0.90170193;
            cameras[i].R = R;

            LOGLN("fix: Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R <<"\nfocal: " <<cameras[i].focal);
        }
		//focals.push_back(cameras[i].focal);
	}
#endif
	LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif

	vector<Point> corners(num_images);
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<UMat> masks(num_images);

    for (size_t i = 0; i < cameras.size(); ++i)
	{
		LOGLN("fix: Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R <<"\nfocal: " <<cameras[i].focal << " aspect: " << cameras[i].aspect << " ppx:" << cameras[i].ppx <<  " ppy: " << cameras[i].ppy );
		//focals.push_back(cameras[i].focal);
	}

	// Preapre images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks

	Ptr<WarperCreator> warper_creator;
#if 1
#ifdef HAVE_OPENCV_CUDAWARPING
	if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarperGpu>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarperGpu>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarperGpu>();
	}
	else
#endif
#endif
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarper>();
		else if (warp_type == "affine")
			warper_creator = makePtr<cv::AffineWarper>();
		else if (warp_type == "cylindrical")	
			warper_creator = makePtr<cv::CylindricalWarper>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarper>();
		else if (warp_type == "fisheye")
			warper_creator = makePtr<cv::FisheyeWarper>();
		else if (warp_type == "stereographic")
			warper_creator = makePtr<cv::StereographicWarper>();
		else if (warp_type == "compressedPlaneA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlaneA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniA2B1")
			warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniA1.5B1")
			warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniPortraitA2B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniPortraitA1.5B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "mercator")
			warper_creator = makePtr<cv::MercatorWarper>();
		else if (warp_type == "transverseMercator")
			warper_creator = makePtr<cv::TransverseMercatorWarper>();
	}

	if (!warper_creator)
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
	
    LOGLN("fix: Camera #" << "\twarped_image_scale: " << warped_image_scale << "\tseam_work_aspect: " << seam_work_aspect);

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
        LOGLN("fix: Camera #" << "\nK:\n" << K << "\nR:\n" << cameras[i].R << "\ncorners: " << corners[i] << "\tsizes: " << sizes[i]);
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	LOGLN("###############Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = makePtr<detail::NoSeamFinder>();
	else if (seam_find_type == "voronoi")
		seam_finder = makePtr<detail::VoronoiSeamFinder>();
	else if (seam_find_type == "gc_color")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
            //seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
	if (!seam_finder)
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
		return 1;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);

#if 0
    int z = 0;
    for (z = 0; z < 2; z++) {
        LOGLN(" masks_warped type: " << masks_warped[z].type()  << "CV_8U: " << CV_8U  << " size: " <<  masks_warped[z].size());
        Mat a;
        a = masks_warped[z].getMat(ACCESS_READ);
        for (int i = 0; i < a.cols * a.rows; i++) {
            printf("%d,", a.data[i]);
        }
    }
#endif

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	double compose_work_aspect = 1;

#if 1
    if (!is_compose_scale_set)
    {
        if (compose_megapix > 0)
            compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
        is_compose_scale_set = true;

        // Compute relative scales
        //compose_seam_aspect = compose_scale / seam_scale;
        compose_work_aspect = compose_scale / work_scale;

        printf("%d\n", __LINE__);
        // Update warped image scale
        warped_image_scale *= static_cast<float>(compose_work_aspect);
        printf("fix: warped_image_scale: %f\n", warped_image_scale);
        warper = warper_creator->create(warped_image_scale);

            printf("%s, %s, %d\n", __FILE__, __func__, __LINE__);
        // Update corners and sizes
        for (int i = 0; i < num_images; ++i)
        {
            // Update intrinsics
            cameras[i].focal *= compose_work_aspect;
            cameras[i].ppx *= compose_work_aspect;
            cameras[i].ppy *= compose_work_aspect;

            printf("%s, %s, %d\n", __FILE__, __func__, __LINE__);
            // Update corner and size
            Size sz = full_img_sizes[i];
#if 0
            if (std::abs(compose_scale - 1) > 1e-1)
            {
                sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                printf("fix: compose_scale:: %f, sz.w : %d, sz.h: %d\n", compose_scale, sz.width, sz.height);
            }
#endif

            Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            Rect roi = warper->warpRoi(sz, K, cameras[i].R); // todo
            corners[i] = roi.tl();
            sizes[i] = roi.size();
            LOGLN("fix: Camera #" << i <<"\nK:\n" << K << "\nR:\n" << cameras[i].R << "\ncorners: " << corners[i] << "\tsizes: " << sizes[i]);
        }
    }
#endif
            printf("%s, %s, %d\n", __FILE__, __func__, __LINE__);
#if 0
    for (int i = 0; i < num_images; ++i)
    {
        // Update intrinsics
        cameras[i].focal *= compose_work_aspect;
        cameras[i].ppx *= compose_work_aspect;
        cameras[i].ppy *= compose_work_aspect;

        // Update corner and size
        Size sz = full_img_sizes[i];
#if 1
        if (std::abs(compose_scale - 1) > 1e-1)
        {
            sz.width = cvRound(full_img_sizes[i].width * compose_scale);
            sz.height = cvRound(full_img_sizes[i].height * compose_scale);
            printf("fix: compose_scale:: %f, sz.w : %d, sz.h: %d\n", compose_scale, sz.width, sz.height);
        }
#endif

        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper->warpRoi(sz, K, cameras[i].R); // todo
        corners[i] = roi.tl();
        sizes[i] = roi.size();
        LOGLN("fix: Camera #" << i <<"\nK:\n" << K << "\nR:\n" << cameras[i].R << "\ncorners: " << corners[i] << "\tsizes: " << sizes[i]);
    }
#endif
    FileStorage fs( "camera.yml", FileStorage::WRITE );
    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );
    fs << "calibration_time" << buf;

    for (int i = 0; i < num_images; i++) {
        Mat a;
        a = masks_warped[i].getMat(ACCESS_READ);

        fs << "Camera" + to_string(i) << i;
        fs << "R" + to_string(i) <<  cameras[i].R;
        fs << "focal" + to_string(i) << cameras[i].focal;
        fs << "aspect" + to_string(i) << cameras[i].aspect;
        fs << "ppx" + to_string(i) <<  cameras[i].ppx;
        fs << "ppy" + to_string(i) <<  cameras[i].ppy;
        fs << "corners" + to_string(i) <<  corners[i];
        fs << "masks_warped" + to_string(i) << a;
        fs << "warped_image_scale" + to_string(i) << warped_image_scale;
        fs << "seam_work_aspect" + to_string(i) << seam_work_aspect;
        fs << "full_img_sizes" + to_string(i) << full_img_sizes[i];
        fs << "work_scale" + to_string(i) << work_scale;
        fs << "compose_scale" + to_string(i) << compose_scale;
        fs << "indices" + to_string(i) << indices[i];
        fs << "sizes" + to_string(i) << sizes[i];
    }

    fs.release();
#else

    vector<Mat> masks_warped(num_images);
	vector<Point> corners(num_images);
	vector<CameraParams> cameras(num_images);
    vector<int> indices(num_images);
	Mat full_img, img;

    double seam_work_aspect;
	float warped_image_scale;
	Ptr<WarperCreator> warper_creator;
    //warper_creator = makePtr<cv::CylindricalWarper>();
    warper_creator = makePtr<cv::CylindricalWarperGpu>();
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
	vector<Size> full_img_sizes(num_images);
	vector<Size> sizes(num_images);
	double compose_work_aspect = 1;

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    FileStorage fs( "camera.yml", FileStorage::READ);
    string buf;
    fs["calibration_time"] >> buf;
    cout << "calibration_time: " << buf << endl;

    cout << "cameras.size　"<< cameras.size() << "corner.size()" << corners.size()<< endl;
    for (int i = 0; i < num_images; i++) {
        //Mat R = Mat(cameras[i].R.size(), cameras[i].R.type());
        fs["R" + to_string(i)] >>  cameras[i].R;
        //cameras[i].R = R;
        cout << "R type: " << cameras[i].R.type() << " size: " << cameras[i].R.size()<<  " R: " << cameras[i].R << endl;
        fs["focal" + to_string(i)] >> cameras[i].focal;
        fs["aspect" + to_string(i)] >> cameras[i].aspect;
        fs["ppx" + to_string(i)] >>  cameras[i].ppx;
        fs["ppy" + to_string(i)] >>  cameras[i].ppy;
        fs["corners" + to_string(i)] >>  corners[i];

        //Mat a = Mat(masks_warped[i].size(), masks_warped[i].type());
        fs["masks_warped" + to_string(i)] >> masks_warped[i];
        fs["warped_image_scale" + to_string(i)] >> warped_image_scale;
        fs["seam_work_aspect" + to_string(i)] >> seam_work_aspect;
        fs["full_img_sizes" + to_string(i)] >> full_img_sizes[i];
        fs["compose_scale" + to_string(i)] >> compose_scale;
        fs["work_scale" + to_string(i)] >> work_scale;
        fs["indices" + to_string(i)] >> indices[i];
        fs["sizes" + to_string(i)] >> sizes[i];
        //a.getUMat(ACCESS_READ).copyto(masks_warped);
        //a.copyTo(masks_warped);
        //masks_warped_r[i] = a.getUMat(ACCESS_READ);
    }

    fs.release();


//    compose_work_aspect = compose_scale / work_scale;
    warper = warper_creator->create(warped_image_scale);
	//compensator->feed(corners, images_warped, masks_warped);
    //
    //
#endif

	vector<Mat> img_warped(num_images), img_warped_s(num_images);
    vector<Mat> mask(num_images), mask_warped(num_images);
#ifndef CAMERA_CAL
    vector<cuda::GpuMat> d_xmap(num_images), d_ymap(num_images), d_xmask(num_images), d_ymask(num_images), d_src(num_images), d_dst(num_images), d_mask(num_images), d_mask_s(num_images);
    vector<Mat> xmap(num_images), ymap(num_images), xmask(num_images), ymask(num_images);//  mask_s(num_images);
    vector<Size> xs(num_images), ys(num_images), xms(num_images), yms(num_images);
    vector<Rect> dst_roi(num_images), mask_roi(num_images);
    
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {

        char name[10];
        int i = img_idx;

        //img
        sprintf(name, "map%d.yml", i);
        FileStorage fs( name, FileStorage::READ);
        string buf;
        fs["calibration_time"] >> buf;
        cout << "calibration_time: " << buf << endl;

        sprintf(name, "dst_roi%d", i);
        fs[name] >>  dst_roi[img_idx];

        sprintf(name, "xmap%d", i);
        fs[name] >>  xmap[img_idx];
        sprintf(name, "ymap%d", i);
        fs[name] >>  ymap[img_idx];
#if 0
        sprintf(name, "xsize%d", i);
        fs[name] >>  xs[img_idx];
        sprintf(name, "ysize%d", i);
        fs[name] >>  ys[img_idx];
        xmap[img_idx].create(xs[img_idx], CV_32FC1);
        ymap[img_idx].create(ys[img_idx], CV_32FC1);
#endif


#if 0
        // mask
        i += 1;
        sprintf(name, "map%d.yml", i);
        FileStorage fsm( name, FileStorage::READ);
        string buf;
        fsm["calibration_time"] >> buf;
        cout << "calibration_time: " << buf << endl;

        sprintf(name, "dst_roi%d", i);
        fsm[name] >>  mask_roi[img_idx];

        sprintf(name, "xmap%d", i);
        fsm[name] >>  xmask[img_idx];
        sprintf(name, "ymap%d", i);
        fsm[name] >>  ymask[img_idx];
        sprintf(name, "xsize%d", i);
        fsm[name] >>  xms[img_idx];
        sprintf(name, "ysize%d", i);
        fsm[name] >>  yms[img_idx];

        xmask[img_idx].create(xms[img_idx], CV_32FC1);
        ymask[img_idx].create(yms[img_idx], CV_32FC1);

        mask[img_idx].create(Size(1444, 1444), CV_8U);
        mask[img_idx].setTo(Scalar::all(255));
        mask_s[img_idx].create(mask_roi[img_idx].height + 1, mask_roi[img_idx].width + 1, mask[img_idx].type()); 

        d_mask[img_idx].upload(mask[img_idx]);
        //mask_roi = warper->buildMaps(mask.size(), K, cameras[img_idx].R, d_xmask, d_ymask);
        cuda::remap(d_mask, d_mask_s, d_xmask, d_ymask, INTER_NEAREST, BORDER_CONSTANT);
        d_mask_s.download(mask_warped);

        d_xmap.release();
        d_ymap.release();
        d_xmask.release();
        d_ymask.release();
#endif
        d_xmap[img_idx].upload(xmap[img_idx]);
        d_ymap[img_idx].upload(ymap[img_idx]);
        //dst_roi = warper->buildMaps(img.size(), K, cameras[img_idx].R, d_xmap, d_ymap);
        
        img_warped[img_idx].create(dst_roi[img_idx].height + 1, dst_roi[img_idx].width + 1, CV_8UC3); 
        fs.release();

        {
            char name[10];
            sprintf(name, "mask_warped%d.yml", img_idx);
            FileStorage fs( name, FileStorage::READ );
            string buf;
            fs["calibration_time"] >> buf;

            sprintf(name, "mask_warped%d", img_idx);
            fs[name] >> mask_warped[img_idx];

            fs.release();
        }


    }

#endif // endof CAMERA_CAL
	LOGLN("Compositing...");



#ifdef CAMERA_CAL
//	Mat img_warped, img_warped_s;
#endif // endof CAMERA_CAL
	Mat dilated_mask, seam_mask;// mask, mask_warped;
	Ptr<Blender> blender;
	Ptr<Timelapser> timelapser;
	//double compose_seam_aspect = 1;
	//double compose_work_aspect = 1;
	MultiBandInfo  band_info;
	FeatherInfo_t  feather_info;


  
    Size dst_sz = resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
    int band_num =static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.);
    static bool is_inited = false;
    t = getTickCount();
//    if (blend_type == Blender::MULTI_BAND)
//    {
        if(try_cuda  && !is_inited)
        {
#if 0
            MultiBandBlender_init(band_info,try_cuda,band_num);
            MultiBandBlender_prepare(band_info,resultRoi(corners, sizes));
#else
			blender = Blender::createDefault(blend_type, try_cuda);           
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_cuda);      
			else if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
				fb->setSharpness(1.f/blend_width);
				LOGLN("Feather blender, sharpness: " << fb->sharpness());
			}
			blender->prepare(corners, sizes);

#endif
            is_inited = true;

            LOGLN("##########multi blend prepare time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
            t = getTickCount();
        }

#if 0
        if(is_inited)
        {
            MultiBandBlender_feed(band_info,img_warped_s[img_idx],mask_warped,corners[img_idx]);
        }
#endif

#if 0
    }else if(blend_type == Blender::FEATHER){

        if(try_cuda  && !is_inited)
        {
            FeatherBlanderInit(&feather_info,1.f / blend_width);
            FeatherBlanderPrepare(&feather_info,resultRoi(corners, sizes));
            is_inited = true;
            LOGLN("##########feather blend prepare time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
            t = getTickCount();
        }
#endif
#if 0
        if(is_inited)
        {
            FeatherBlanderFeed(feather_info,img_warped_s[img_idx],mask_warped,corners[img_idx]);
        }
#endif
    //}


	//InitCUDA();


    
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        mask[img_idx].create(Size(1444,1444), CV_8U);
        mask[img_idx].setTo(Scalar::all(255));
    }
    

    int maped = 0;

    std::vector<Mat> bimg;
    bimg.resize(num_images);


#if 1
#ifndef CAMERA_CAL

    IplImage *r[num_images];

    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        r[img_idx] = cvLoadImage(img_names[img_idx].c_str(), 1);
    }

    float *fw[num_images];
    float *dfw[num_images];
    char *cbuf[num_images];
#if 0
    short *feedbuf[num_images];
    short *feedbuf_d[num_images];
#endif
    char *dcbuf[num_images];
    char *cbuf_d[num_images];
    float *dstw;
    float *ddstw;
    IplImage *showImg;
    char *sbuf;
    short *ssbuf;
    short *ssbuf_d;
    char *dsbuf;
    IplImage *srcImg[num_images];// = cvCreateImage(cvSize(gtx->vic_w, gtx->vic_h), 8, 3);
    IplImage *mapImg[num_images];// = cvCreateImage(cvSize(gtx->vic_w, gtx->vic_h), 8, 3);
    float *xmapbuf[num_images], *ymapbuf[num_images];
    float *xmapbuf_d[num_images], *ymapbuf_d[num_images];
    short *warpbuf[num_images], *warpbuf_d[num_images];
    Rect dst_roi_;
    cudaError_t status;
    cudaError_t err;


    vector<Mat> weights(num_images);
    Mat dst_weight_map_;
    FileStorage wfs("weights.yml", FileStorage::READ);
    //string buf;
    wfs["calibration_time"] >> buf;
    cout << "calibration_time: " << buf << endl;

    wfs["dst_roi"] >> dst_roi_;
    wfs["dst_weight_map"] >> dst_weight_map_;

    Mat_<float> W;
    dst_weight_map_.convertTo(W, CV_32F);
    int dwsize = dst_weight_map_.cols * dst_weight_map_.rows;

    printf("========= W.isContinuous: %d============%d\n",W.isContinuous(),  __LINE__);
    status = cudaHostAlloc((void **)&dstw, dwsize* sizeof(float), cudaHostAllocMapped);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));
    printf("================dst_weight_map_.rows: %d === cols: %d ==%d\n", dst_weight_map_.rows, dst_weight_map_.cols,__LINE__);
    for (int j = 0; j < dst_weight_map_.rows; j++)
        for (int z = 0; z < dst_weight_map_.cols; z++) {
            //dstw[j * dst_weight_map_.cols + z] = W[j][z];
            dstw[j * dst_weight_map_.cols + z] = dst_weight_map_.at<float>(j, z);
		//if (j < 200 && z < 200)
//		if (dstw[j * dst_weight_map_.cols + z] > 0)
//		printf(" %f, %f\t, ", dstw[j * dst_weight_map_.cols + z], W[j][z]);
	}
    cudaHostGetDevicePointer((void **)&ddstw, dstw, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error: %s\n", cudaGetErrorString(err));


    showImg = cvCreateImageHeader(Size(dst_roi_.width, dst_roi_.height), 8, 3);

    printf("================dst_roi_ ros. %d, cols %d =====%d\n", dst_roi_.width, dst_roi_.height ,__LINE__);
    //status = cudaHostAlloc((void **)&sbuf, dst_roi_.width * dst_roi_.height * sizeof(char) * 3);
    cudaHostAlloc((void **)&sbuf, dst_roi_.height * showImg->widthStep *  sizeof(char), cudaHostAllocMapped);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error: %s\n", cudaGetErrorString(err));

    status = cudaHostAlloc((void **)&ssbuf, dst_roi_.height * showImg->widthStep *  sizeof(short), cudaHostAllocMapped);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error: %s\n", cudaGetErrorString(err));
    cvSetData(showImg, sbuf, showImg->widthStep);
    printf("showImg  widtshSetp: %d\n", showImg->widthStep);



    cudaHostGetDevicePointer((void **)&dsbuf, sbuf, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error: %s\n", cudaGetErrorString(err));

    cudaHostGetDevicePointer((void **)&ssbuf_d, ssbuf, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error: %s\n", cudaGetErrorString(err));
    printf("=====================%d, num_images: %d\n", __LINE__, num_images);
    for (int img_idy = 0; img_idy < num_images; img_idy++)
    {

        printf("=====================%d, i: %d\n", __LINE__, img_idy);
        char name[20];
        sprintf(name, "weight%d", img_idy);
        wfs[name] >> weights[img_idy];
        printf(" cols: %d, rows: %d \n", weights[img_idy].cols, weights[img_idy].rows);
        srcImg[img_idy] = cvCreateImageHeader(Size(1444, 1444), 8, 3);
        status = cudaHostAlloc((void **)&cbuf[img_idy], 1444 * srcImg[img_idy]->widthStep  * sizeof(char), cudaHostAllocMapped);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error: %s\n", cudaGetErrorString(err));
        //status = cudaHostAlloc((void **)&feedbuf[img_idy], 1444 * 1444 * sizeof(short) * 3, cudaHostAllocMapped);
        cvSetData(srcImg[img_idy], cbuf[img_idy], srcImg[img_idy]->widthStep);
	printf(" r. widthstep: %d, w : %d, h: %d\n", r[img_idy]->widthStep, r[img_idy]->width, r[img_idy]->height);
        //r[img_idy]->copyTo(srcImg[img_idy]);
        for (int i = 0; i < 1444; i++)
            for (int j = 0; j < 1444; j++)
            for (int k = 0; k < 3; k++) {
                *(cbuf[img_idy] + i * srcImg[img_idy]->widthStep + j * 3 + k) = *(r[img_idy]->imageData + i * r[img_idy]->widthStep + j * 3 + k);
                //*(feedbuf[img_idy] + i * 1444 * 3 + j * 3 + k) = *(r[img_idy]->imageData + i * 1444 * 3 + j * 3 + k);
            }
    cvShowImage("result1", srcImg[img_idy]); 
    waitKey(0);

        printf("=====================%d\n", __LINE__);
        Mat_<float> K;
        int size = weights[img_idy].cols * weights[img_idy].rows;

        weights[img_idy].convertTo(K, CV_32F);

        printf("===========size: %d === K.isContinuous: %d =======%d\n",size, K.isContinuous(),  __LINE__);
     //   fw[img_idy] = (float*)malloc(size* sizeof(float));
#if 1
        status = cudaHostAlloc((void **)&fw[img_idy], size* sizeof(float), cudaHostAllocMapped);
    err = cudaGetLastError();
    if (err != cudaSuccess)
	printf("Error: %s\n", cudaGetErrorString(err));

#endif
        printf("===========weights: rows: %d ,cols: %d==========%d\n", weights[img_idy].rows, weights[img_idy].cols,   __LINE__);
        for (int x = 0; x < weights[img_idy].rows; x++)
        for (int y = 0; y < weights[img_idy].cols; y++)
        {
            //printf("=====================%d\n", __LINE__);
            fw[img_idy][x *  weights[img_idy].cols + y] =  weights[img_idy].at<float>(x,y);
        }

        cudaHostGetDevicePointer((void **)&dfw[img_idy], fw[img_idy], 0);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

        cudaHostGetDevicePointer((void **)&dcbuf[img_idy], cbuf[img_idy], 0);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
        //cudaHostGetDevicePointer((void **)&feedbuf_d[img_idy], feedbuf[img_idy], 0);
    }
    
    for (int img_idx = 0; img_idx < num_images; img_idx++) {
        Mat_<float> M;
        int w, h;
        w = xmap[img_idx].cols;
        h = xmap[img_idx].rows;

        xmap[img_idx].convertTo(M, CV_32F);
        status = cudaHostAlloc((void **)&xmapbuf[img_idx], w * h * sizeof(float), cudaHostAllocMapped);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
        cudaHostGetDevicePointer((void **)&xmapbuf_d[img_idx], xmapbuf[img_idx], 0);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));


        Mat_<float> N;
        ymap[img_idx].convertTo(N, CV_32F);
        status = cudaHostAlloc((void **)&ymapbuf[img_idx], w * h * sizeof(float), cudaHostAllocMapped);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
        cudaHostGetDevicePointer((void **)&ymapbuf_d[img_idx], ymapbuf[img_idx], 0);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

        status = cudaHostAlloc((void **)&warpbuf[img_idx], getWidth(w, 3) * h * sizeof(short), cudaHostAllocMapped);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
        cudaHostGetDevicePointer((void **)&warpbuf_d[img_idx], (void *)warpbuf[img_idx], 0);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	for (int i = 0; i < h; i++)
	for (int j = 0; j < w; j++) {
	//	if (xmap[img_idx].at<float>(i, j) > 500.)
			//printf("==%f, %f\t",  xmapbuf[img_idx][i * w + j],  ymapbuf[img_idx][i * w + j]);
	//		printf("xmap: %f\t", xmap[img_idx].at<float>(i, j) );
		xmapbuf[img_idx][i * w + j] = xmap[img_idx].at<float>(i, j);
		ymapbuf[img_idx][i * w + j] = ymap[img_idx].at<float>(i, j);
	}
        mapImg[img_idx] = cvCreateImage(cvSize(w, h), 8, 3);// = cvCreateImage(cvSize(gtx->vic_w, gtx->vic_h), 8, 3);
//                cvShowImage("warp", mapImg[img_idx]);
//                cvWaitKey(0);
    printf(" img_idx: %d ===========w: %d, h: %d====== widtstep : %d = depth: %d=, get width: %d ==%d\n", img_idx, w, h, mapImg[img_idx]->widthStep, mapImg[img_idx]->depth, getWidth(w, 3), __LINE__);
    }

        wfs.release();
    printf("=====================%d\n", __LINE__);


//   while (1) 
    {
        for (int j = 0; j < num_images; j++) {
            //feather_blender_feed(dcbuf[j] , dsbuf, dfw[j], 1444, 1444, corners[j].x - dst_roi_.x, corners[j].y - dst_roi_.y);
            printf(" j : %d mapImg[j]->widthStep : %d ==? %d, rows: %d cols: %d\n", j,  mapImg[j]->widthStep, ((mapImg[j]->width + 3) / 4) * 4  * 3, xmap[j].rows, xmap[j].cols);
		printf("srcImg->widthStep: %d\n",srcImg[j]->widthStep);
//		cudaThreadSynchronize();
            cuda_remap(dcbuf[j], warpbuf_d[j], xmapbuf_d[j], ymapbuf_d[j], xmap[j].cols, xmap[j].rows, srcImg[j]->width, srcImg[j]->height);
            for (int i = 0; i < xmap[j].rows; i++)
            for (int k = 0; k < xmap[j].cols; k++) {
                *(mapImg[j]->imageData + i * mapImg[j]->widthStep + k * 3 + 0) = *(warpbuf[j] + i * mapImg[j]->widthStep + k * 3 + 0);
                *(mapImg[j]->imageData + i * mapImg[j]->widthStep + k * 3 + 1) = *(warpbuf[j] + i * mapImg[j]->widthStep + k * 3 + 1);
                *(mapImg[j]->imageData + i * mapImg[j]->widthStep + k * 3 + 2) = *(warpbuf[j] + i * mapImg[j]->widthStep + k * 3 + 2);
            }
                cvShowImage("warp", mapImg[j]);
                cvWaitKey(0);

            std::cout << "corners: " << corners[j] << std::endl;
            feather_blender_feed(warpbuf_d[j] , ssbuf_d, dfw[j], xmap[j].cols, xmap[j].rows, dst_roi_.width,  dst_roi_.height, corners[j].x - dst_roi_.x, corners[j].y - dst_roi_.y);
            printf("=======corners[j].x: %d, corners[j].y: %d==============%d\n", corners[j].x, corners[j].y, __LINE__);
        }

        feather_blender_blend(ssbuf_d, dstw, dst_roi_.width, dst_roi_.height);

        int widthstep = getWidth(dst_weight_map_.cols, 3);
        printf("=====================%d\n", __LINE__);
        for (int j = 0; j < dst_weight_map_.rows; j++)
        for (int z = 0; z < dst_weight_map_.cols; z++) {
            sbuf[j * widthstep + z * 3 + 0] = ssbuf[j * widthstep + z * 3 + 0];
            sbuf[j * widthstep + z * 3 + 1] = ssbuf[j * widthstep + z * 3 + 1];
            sbuf[j * widthstep + z * 3 + 2] = ssbuf[j * widthstep + z * 3 + 2];
        }
        cvShowImage("result", showImg);
        waitKey(0);

	return 0;
    }

#endif
#endif

#ifndef CAMERA_CAL
//    while (1) 
#endif
    {
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
        bimg[img_idx] = imread(img_names[img_idx]);

        t = getTickCount();
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		LOGLN("Compositing image #" << indices[img_idx] + 1);

		// Read image and resize it if necessary
		//full_img = imread(img_names[img_idx]);
#if 0
		if (!is_compose_scale_set)
		{
			if (compose_megapix > 0)
				compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
			is_compose_scale_set = true;

			// Compute relative scales
			//compose_seam_aspect = compose_scale / seam_scale;
			compose_work_aspect = compose_scale / work_scale;

    printf("%d\n", __LINE__);
			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
            printf("fix: warped_image_scale: %f\n", warped_image_scale);
			warper = warper_creator->create(warped_image_scale);

			// Update corners and sizes
			for (int i = 0; i < num_images; ++i)
			{
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				// Update corner and size
				Size sz = full_img_sizes[i];
#if 0
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                    printf("fix: compose_scale:: %f, sz.w : %d, sz.h: %d\n", compose_scale, sz.width, sz.height);
				}
#endif

				Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[i].R); // todo
				corners[i] = roi.tl();
				sizes[i] = roi.size();
                LOGLN("fix: Camera #" << i <<"\nK:\n" << K << "\nR:\n" << cameras[i].R << "\ncorners: " << corners[i] << "\tsizes: " << sizes[i]);
			}
		}
#endif
#if 0
		if (abs(compose_scale - 1) > 1e-1) {
            cv::resize(full_img, img, Size(), compose_scale, compose_scale);
            printf("check :   reszie ?");
        }
		else
#endif
		//img = full_img;
		//full_img.release();
		//Size img_size = img.size();

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);
#if ENABLE_LOG
#endif


#ifndef CAMERA_CAL
        //img_warped[img_idx].create(dst_roi[img_idx].height + 1, dst_roi[img_idx].width + 1, img.type()); 
        //t = getTickCount();
#if 1
#if 0
        for (int x = 0; x < 50; x++)
        for (int y = 0; y < 50; y++)
        {
            printf("<%d, %d> -> (%08f, %08f)\t", x, y, xmap[img_idx].at<float>(x, y), ymap[img_idx].at<float>(x, y) );
        }
#endif
        d_src[img_idx].upload(bimg[img_idx]);
        cuda::remap(d_src[img_idx], d_dst[img_idx], d_xmap[img_idx], d_ymap[img_idx], INTER_LINEAR, BORDER_REFLECT);
        //cuda::remap(d_src[img_idx], d_dst[img_idx], d_xmap[img_idx], d_ymap[img_idx], INTER_LINEAR, BORDER_CONSTANT);
        d_dst[img_idx].download(img_warped[img_idx]);
		img_warped[img_idx].convertTo(img_warped_s[img_idx], CV_16S);
        //LOGLN("##########remap cost time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#else
        cv::remap(bimg[img_idx], img_warped[img_idx], xmap[img_idx], ymap[img_idx], INTER_LINEAR, BORDER_REFLECT);   
        img_warped[img_idx].convertTo(img_warped_s[img_idx], CV_16S);
#endif

#else


        // Warp the current image
        printf("================== >> begin to save warp img i: %d %d\n",img_idx,  __LINE__);
        //warper->warp(bimg[img_idx], K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped[img_idx]); // todo
        warper->warp(bimg[img_idx], K, cameras[img_idx].R, INTER_LINEAR, BORDER_CONSTANT, img_warped[img_idx]); // todo
        printf("================== >> begin to save warp mask i: %d %d\n",img_idx,  __LINE__);
        // Warp the current image mask
        warper->warp(mask[img_idx], K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped[img_idx]); // todo

        //
        // Compensate exposure
        LOGLN("fix img_idx: " << img_idx << " corners: " << corners[img_idx]);
        LOGLN("##########Warp time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

#if 0
        Point a(-418, 6568), b(-1210,6588);
        if (img_idx == 0)
		compensator->apply(img_idx, a, img_warped, mask_warped); // todo
        else
		compensator->apply(img_idx, b, img_warped, mask_warped); // todo
        
#endif
		//compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped); // todo

		img_warped[img_idx].convertTo(img_warped_s[img_idx], CV_16S);
//		img_warped.release();
		//img.release();
		//mask.release();
        printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);



        printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);
		dilate(masks_warped[img_idx], dilated_mask, Mat());
        cv::resize(dilated_mask, seam_mask, mask_warped[img_idx].size());
		mask_warped[img_idx] = seam_mask & mask_warped[img_idx];


        printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);
        {
            char name[10];
            sprintf(name, "mask_warped%d.yml", img_idx);
            FileStorage fs( name, FileStorage::WRITE );
            time_t tt;
            time( &tt );
            struct tm *t2 = localtime( &tt ); 
            char buf[1024];
            strftime( buf, sizeof(buf)-1, "%c", t2 );
            fs << "calibration_time" << buf;

            sprintf(name, "mask_warped%d", img_idx);
            fs << name << mask_warped[img_idx];

            fs.release();
        }

        //printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);
#endif //endof CAMERA_CAL

#if 0
        {
            char name[10];
            sprintf(name, "%d.png", img_idx);
            imwrite(name, img_warped_s[img_idx]);
        }
#endif

#if 0
		Size dst_sz = resultRoi(corners, sizes).size();
		float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
		int band_num =static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.);
		static bool is_inited = false;
		if (blend_type == Blender::MULTI_BAND)
		{
			if(try_cuda  && !is_inited)
			{
				MultiBandBlender_init(band_info,try_cuda,band_num);
				MultiBandBlender_prepare(band_info,resultRoi(corners, sizes));
				is_inited = true;
				
				LOGLN("##########multi blend prepare time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
				t = getTickCount();
			}
			
			if(is_inited)
			{
				MultiBandBlender_feed(band_info,img_warped_s[img_idx],mask_warped,corners[img_idx]);
			}

		}else if(blend_type == Blender::FEATHER){

			if(try_cuda  && !is_inited)
			{
					FeatherBlanderInit(&feather_info,1.f / blend_width);
					FeatherBlanderPrepare(&feather_info,resultRoi(corners, sizes));
					is_inited = true;
					LOGLN("##########feather blend prepare time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
					t = getTickCount();
			}

			if(is_inited)
			{
				FeatherBlanderFeed(feather_info,img_warped_s[img_idx],mask_warped,corners[img_idx]);
			}
		}else{

		}
#else
	//t = getTickCount();
#if 0
        if (blend_type == Blender::MULTI_BAND)
        {
            MultiBandBlender_feed(band_info,img_warped_s[img_idx],mask_warped[img_idx],corners[img_idx]);

        } else if (blend_type == Blender::FEATHER) {
            FeatherBlanderFeed(feather_info,img_warped_s[img_idx],mask_warped[img_idx],corners[img_idx]);
        }
#else

        if (maped == 0) {
            //printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);
            std::cout << "corners: " << corners << std::endl;

            FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get()); 
            //printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);
            //printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);
            fb->map(mask_warped[img_idx], corners[img_idx], bimg[img_idx].rows, bimg[img_idx].cols, img_idx);
            //printf(" debug %s, %s, %d\n", __FILE__, __func__, __LINE__);
        }
        
        FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get()); 
        fb->feed(img_warped_s[img_idx], mask_warped[img_idx], corners[img_idx], img_idx);
#endif
#endif
        bimg[img_idx].release();
		
	}

    if (maped == 0)
        maped = 1;
    
	LOGLN("##########blend feed time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	t = getTickCount();
	
	if (!timelapse)
	{
		Mat result, result_mask;
		
		if(!try_cuda) {
			blender->blend(result, result_mask);
		} else {
#if 0
			if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender_blend(band_info,result, result_mask);
			}
			else if(blend_type == Blender::FEATHER)
			{
				FeatherBlanderBlend(feather_info,result,result_mask);
			}
			else
			{

			}
#else
		blender->blend(result, result_mask);
#endif
		}
		
		LOGLN("##########blender blend time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		Mat d;
		d.create(result.size(), CV_8UC3);

		result.convertTo(d, CV_8U);

		imshow("result", d);
		waitKey(0);
		//imwrite(result_name, result);
		while (0) {
			imshow("result", result);
			waitKey(33);
		}

		//result.release();
		//result_mask.release();
	}
    }

	LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

	return 0;
}
