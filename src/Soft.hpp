#ifndef SOFT_HPP
#define SOFT_HPP

#include "Tools.hpp"

class SoftBOWImgDescriptorExtractor : public cv::BOWImgDescriptorExtractor
{
public:
	SoftBOWImgDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& _dextractor, const cv::Ptr<cv::DescriptorMatcher>& _dmatcher);
	void compute(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, cv::Mat & imgDescriptor, int k);
};

#endif