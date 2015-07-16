#ifndef SOFT_HPP
#define SOFT_HPP

#include "tools.hpp"

class SoftBOWImgDescriptorExtractor : public BOWImgDescriptorExtractor
{
public:
	SoftBOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& _dextractor, const Ptr<DescriptorMatcher>& _dmatcher )
		: BOWImgDescriptorExtractor(_dextractor, _dmatcher) {}

	void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& imgDescriptor, int k );
};

#endif