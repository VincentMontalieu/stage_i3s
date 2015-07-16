#include "soft.hpp"

using namespace std;
using namespace cv;


void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& imgDescriptor, int k )
{
	imgDescriptor.release();

	if ( keypoints.empty() )
		return;

	int clusterCount = descriptorSize(); // = vocabulary.rows

	// Compute descriptors for the image.
	Mat descriptors = Mat();
	dextractor->compute( image, keypoints, descriptors );


	// coding
	vector<vector<DMatch>> matches;
	dmatcher->knnMatch(descriptors, matches, k);

	//pooling
	imgDescriptor = Mat( 1, clusterCount, CV_32FC1, Scalar::all(0.0) );
	float *dptr = (float*)imgDescriptor.data;

	for (int i = 0; i < matches.size(); i++) {

		for (int j = 0; j < matches[i].size(); j++) {
			if (dptr[matches[i][j].trainIdx] == 0 || dptr[matches[i][j].trainIdx] > matches[i][j].distance)
				dptr[matches[i][j].trainIdx] = matches[i][j].distance;
		}
	}
}