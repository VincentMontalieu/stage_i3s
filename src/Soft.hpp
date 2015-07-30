#ifndef SOFT_HPP
#define SOFT_HPP

#include "Tools.hpp"

/*
* Classe qui modélise un extracteur de BOW en soft assignment, en respectant l'implémentation OpenCV de la version hard assignment
*/
class SoftBOWImgDescriptorExtractor : public cv::BOWImgDescriptorExtractor
{
public:
	SoftBOWImgDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& _dextractor, const cv::Ptr<cv::DescriptorMatcher>& _dmatcher);
	
	// Surcharge de la méthode compute présente dans l'implémentation OpenCV de BOWImgDescriptorExtractor
	void compute(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, cv::Mat & imgDescriptor, int k);
};

#endif