
#include "aligment.hpp"
#include "filter.hpp"
#include "histogram.hpp"
/*exceptional case: SIFT/ORB does not work well because the sift does not provide an octave with a feature fix in older code*/
void descriptorsAqusition(cv::Mat img,std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors,
                          struct settings settings,
                          cv::Ptr<cv::FeatureDetector> detector,
                          cv::Ptr<cv::DescriptorExtractor> descriptor,
                          cv::GriefDescriptorExtractor *griefDescriptor ){
  /*detection*/
  detector->detect(img, keypoints);
  sort(keypoints.begin(),keypoints.end(),compare_response);
  /*extraction*/
  if (settings.upright) for (unsigned int j = 0;j<keypoints.size();j++) keypoints[j].angle = -1;
  if(griefDescriptor != NULL) griefDescriptor->computeImpl(img,keypoints,descriptors);
  else
    descriptor->compute(img,keypoints,descriptors);
  resizeFeatures(descriptors,keypoints,settings.featureMaximum);
}
// inlier mathces are require  for this to work i guess ...

float imageDisplacement( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2,  float groundTruth,  int &fails,settings settings , std::ofstream& hist_file_out,std::vector<cv::DMatch> &inliers_matches,
                         std::vector<std::vector <double> > *sortedHistogram  ){
					//use all features when numFeatures is 0
					std::vector<cv::DMatch> matches;
					// matching descriptors
					if (descriptors1.rows*descriptors2.rows > 0){
						if (sortedHistogram == nullptr) 	   distinctiveMatch(descriptors1, descriptors2, matches, norm2,settings.crossCheck, settings.distance_factor);
            else pre_filter(descriptors1,descriptors2,keypoints1,keypoints2,matches,settings.crossCheck,settings.distance_factor,*sortedHistogram);
					}
          int numBins = 100;
          int histogram[100];
          int bestHistogram[100];
          float difference = 0;
          float displacement = 999999;
					if (matches.size() > 0){
            inliers_matches = internalHistogram(keypoints1,keypoints2, displacement, numBins, histogram, bestHistogram , matches,   settings.granularity,settings.verticaLimit);
					}
#pragma omp ordered
          difference = displacement + groundTruth;//((offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]));
					if (fabs(difference) > 35) fails++;
					/*if the heading estimation error is bigger than 35 pixels, it's considered as false, otherwise it's considered correct*/
          resultsOut(displacement,inliers_matches.size(),difference,bestHistogram, hist_file_out,fails);
          return displacement;
}

void resultsOut(float displacement, int matchesSize, float difference, int (&bestHistogram)[100],  std::ofstream& hist_file_out, int fails){

  if (displacement < 9999) hist_file_out << displacement << "," << matchesSize << "," << difference << "," << fails << ",";
  else hist_file_out << displacement << "," << 0 << "," << difference << "," << fails << ",";
  int numBins = sizeof(bestHistogram)/sizeof(bestHistogram[0]);
  for (int i = 0; i < numBins; i++){
    hist_file_out << bestHistogram[i];
    if (i != numBins-1) hist_file_out << ",";
  }
  hist_file_out << "\n";
}
