#include "Soft.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

SoftBOWImgDescriptorExtractor::SoftBOWImgDescriptorExtractor(const Ptr<DescriptorExtractor>& _dextractor, const Ptr<DescriptorMatcher>& _dmatcher) : BOWImgDescriptorExtractor(_dextractor, _dmatcher)
{

}

void SoftBOWImgDescriptorExtractor::compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& imgDescriptor, int k)
{
	imgDescriptor.release();

	if ( keypoints.empty() )
		return;

	int clusterCount = BOWImgDescriptorExtractor::descriptorSize(); // = vocabulary.rows

	// Compute descriptors for the image.
	Mat descriptors = Mat();
	dextractor->compute( image, keypoints, descriptors );


	// coding
	vector<vector<DMatch>> matches;
	dmatcher->knnMatch(descriptors, matches, k);

	//pooling
	imgDescriptor = Mat( 1, clusterCount, CV_32FC1, Scalar::all(0.0) );
	float *dptr = (float*)imgDescriptor.data;

	for (size_t i = 0; i < matches.size(); i++) {

		for (size_t j = 0; j < matches[i].size(); j++) {
			if (dptr[matches[i][j].trainIdx] == 0 || dptr[matches[i][j].trainIdx] > matches[i][j].distance)
				dptr[matches[i][j].trainIdx] = matches[i][j].distance;
		}
	}
}