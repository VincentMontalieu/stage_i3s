#include "ImageData.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

ImageData::ImageData(string filenamep, Mat bowDescriptorsp, vector<KeyPoint> imageKeypointsp, Mat imageDescriptorsp)
{
	filename = filenamep;
	bowDescriptors = bowDescriptorsp;
	imageKeypoints = imageKeypointsp;
	imageDescriptors = imageDescriptorsp;
}