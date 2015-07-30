#ifndef IMAGE_DATA_HPP
#define IMAGE_DATA_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include <vector>
#include <string>

/*
* Classe contenant les informations utiles pour un fichier image
*/
class ImageData
{
public:
	// Le nom du fichier avec chemin relatif
	std::string filename;

	// La matrice contenant l'histogramme d'une image (pas utilisée pour le moment)
	cv::Mat bowDescriptors;

	// Le vecteur contenant les points d'intérêt d'une image
	std::vector<cv::KeyPoint> imageKeypoints;

	// La matrice contenant les descripteurs SIFT d'une image
	cv::Mat imageDescriptors;

	ImageData(std::string filenamep, cv::Mat bowDescriptorsp, std::vector<cv::KeyPoint> imageKeypointsp, cv::Mat imageDescriptorsp);
};

#endif