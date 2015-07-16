#ifndef IMAGE_DATA_HPP
#define IMAGE_DATA_HPP

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"

#include <vector>
#include <string>

/*
* Classe contenant les informations utiles pour un fichier image (descripteurs SIFT, nom de fichier...)
*/
class ImageData
{
public:
	std::string filename;
	cv::Mat bowDescriptors;
	std::vector<cv::KeyPoint> imageKeypoints;
	cv::Mat imageDescriptors;

	ImageData(std::string filenamep, cv::Mat bowDescriptorsp, std::vector<cv::KeyPoint> imageKeypointsp, cv::Mat imageDescriptorsp);
};

#endif