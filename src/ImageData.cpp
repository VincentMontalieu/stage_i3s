#include "ImageData.hpp"

using namespace std;
using namespace cv;

ImageData::ImageData(string filenamep, Mat bowDescriptorsp, vector<KeyPoint> imageKeypointsp, Mat imageDescriptorsp)
{
	filename = filenamep;
	bowDescriptors = bowDescriptorsp;
	imageKeypoints = imageKeypointsp;
	imageDescriptors = imageDescriptorsp;
}