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