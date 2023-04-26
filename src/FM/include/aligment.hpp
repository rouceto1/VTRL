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
void resultsOut(float displacement, int matchesSize, float difference, int (&bestHistogram)[63], std::ofstream& hist_file_out, int fails);

//computes dispalcement on two images and saves it if it knows where
float imageDisplacement( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2,
                         float groundTruth,  int &fails,int &inliers_matches_count, settings settings , std::ofstream& hist_file_out,
                         std::vector<int> &hist_out, std::vector<std::vector <double> > *sortedHistogram,std::vector<cv::DMatch> &inliers_matches);
//used to compute diff on two images without saving

float imageDisplacementUnsave( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2,
                               settings settings ,int &inliers_matches_count,std::vector<int> &bestHistogram, std::vector<cv::DMatch> &inliers_matches,
                               std::vector<std::vector <double> > *sortedHistogram=nullptr);



// one serves as a dummy if no output file or internal information is given. or both
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features_l,int &features_r, int  &fails,int &inliers_matches_count,
                         float GT = 0.0,std::vector< vector <double> > *sortedHistogram = nullptr);
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features_l,int &features_r, int  &fails,int &inliers_matches_count,
                         std::ofstream& hist_file_out, std::vector<int> &hist_out,
                         float GT = 0.0,std::vector< vector <double> > *sortedHistogram = nullptr);
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features_l,int &features_r, int  &fails,int &inliers_matches_count,
                         std::vector<int> &hist_out,
                         float GT =0.0, std::vector< vector <double> > *sortedHistogram = nullptr);
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features_l,int &features_r, int  &fails,int &inliers_matches_count,
                         std::ofstream& hist_file_out,
                         float GT = 0.0,std::vector< vector <double> > *sortedHistogram= nullptr);

// runs all on two passed iamges and thyer histogram. expecets 63 bin probabilty distribution
float twoImagesAndHistogram(cv::Mat img1, cv::Mat img2, vector<double> histogram_input );
bool compare_response(cv::KeyPoint first, cv::KeyPoint second);

//default config for twoImagesAndHistogram
struct settings default_config ();

float pBindTst(int a);
void teachOnFile(const char* f1, const char* f2, float &displacemnet, int &feature_count_l,int &feature_count_r, int &fails,int &inliers_matches_count);
void evalOnFile(const char* f1, const char* f2, float &displacemnet, int &feature_count_l,int &feature_count_r, int &fails,
                int &inliers_matches_count, vector<double> histogram, float GT, vector<int> &histogram_out);

extern "C"{
  void teachOnFiles(const char ** filesFrom, const char ** filesTo, float *displacement, int *feature_count_l,int *feature_count_r,int *inliers_matches_count, int files );

  void evalOnFiles(const char ** filesFrom, const char ** filesTo,double ** histogram_in, double ** hist_out, double * gt, float *displacement, int *feature_count_l,int *feature_count_r, int *inliers_matches_count, int hist_width,int files);
}
