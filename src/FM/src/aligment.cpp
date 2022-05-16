

#include "aligment.hpp"
/*exceptional case: SIFT/ORB does not work well because the sift does not provide an octave with a feature fix in older code*/
void descriptorsAqusition(cv::Mat img,std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,bool upright,
                          cv::Ptr<cv::FeatureDetector> detector,
                          cv::Ptr<cv::DescriptorExtractor> descriptor,
                          cv::GriefDescriptorExtractor *griefDescriptor ){
  /*detection*/
  detector->detect(img, keypoints);
  sort(keypoints.begin(),keypoints.end(),compare_response);
  /*extraction*/
  if (upright) for (unsigned int j = 0;j<keypoints.size();j++) keypoints[j].angle = -1;
  if(griefDescriptor != NULL) griefDescriptor->computeImpl(img,keypoints,descriptors);
  else
    descriptor->compute(img,keypoints,descriptors);
}
