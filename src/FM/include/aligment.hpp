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

//used itnerlaly detects keypoints and crops them to size
void descriptorsAqusition(cv::Mat img,std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,struct settings settings,
                          cv::Ptr<cv::FeatureDetector> detector,
                          cv::Ptr<cv::DescriptorExtractor> descriptor,
                          cv::GriefDescriptorExtractor *griefDescriptor );

//used for writing to standart file
void resultsOut(float displacement, int matchesSize, float difference, int (&bestHistogram)[100], std::ofstream& hist_file_out, int fails);

//computes dispalcement on two images and saves it if it knows where
float imageDisplacement( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2,  float groundTruth,  int &fails,struct settings settings,  std::ofstream& hist_file_out,std::vector<cv::DMatch> &inliers_matches,std::vector<std::vector <double> > *sortedHistogram = nullptr);
//used to compute diff on two images without saving
float imageDisplacementUnsave( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2, settings settings ,std::vector<cv::DMatch> &inliers_matches,int (&bestHistogram)[100],std::vector<std::vector <double> > *sortedHistogram=nullptr);


// one serves as a dummy if no output file is given 
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features, int  &fails, float GT = 0.0,std::vector< vector <double> > *sortedHistogram = nullptr);
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features, int  &fails, std::ofstream& hist_file_out, float GT = 0.0,std::vector< vector <double> > *sortedHistogram = nullptr);

// runs all on two passed iamges and thyer histogram. expecets 63 bin probabilty distribution
float twoImagesAndHistogram(cv::Mat img1, cv::Mat img2, vector<double> histogram_input );
bool compare_response(cv::KeyPoint first, cv::KeyPoint second);

//default config for twoImagesAndHistogram
struct settings default_config ();

float pBindTst(int a);
void teachOnFile(const char* f1, const char* f2, float &displacemnet, int &feature_count, int &fails);


extern "C"{
  void teachOnFiles(const char ** filesFrom, const char ** filesTo, float *displacement, int *feature_count, int files );


}
