#include <sys/time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <stdio.h>

#include "../src/grief/grief.h"
#include <fstream>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include "helper_functions.hpp"
using std::vector;
using cv::Scalar;

struct settings{
  bool crossCheck;
  int verticaLimit;
  int granularity;
  float distance_factor;
  bool upright;
  int featureMaximum;
  char descriptorName[50];
  char detectorName[50];
} ;


void descriptorsAqusition(cv::Mat img,std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,struct settings settings,
                          cv::Ptr<cv::FeatureDetector> detector,
                          cv::Ptr<cv::DescriptorExtractor> descriptor,
                          cv::GriefDescriptorExtractor *griefDescriptor );
float imageDisplacement( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2,  float groundTruth,  int &fails,struct settings settings,  std::ofstream& hist_file_out,std::vector<cv::DMatch> &inliers_matches,std::vector<std::vector <double> > *sortedHistogram = nullptr);

void resultsOut(float displacement, int matchesSize, float difference, int (&bestHistogram)[100], std::ofstream& hist_file_out, int fails);
