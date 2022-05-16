#ifndef HELPER_FUNCITONS
#define HELPER_FUNCITONS


#include <sys/time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
using std::vector;
using cv::Scalar;


bool norm2 =true;
bool upright =true;

int maxFeatures=1600;

char fileInfo[1000];
bool normalizeSift=false;
enum matching_method {ORIGINAL,PRE_FILTER,POST_FILTER,HIST_ONLY} ; //selector of filtering

enum hist_handling_methods {hist_max,hist_sorted, hist_enthropy, hist_zero} ; //selector of specific hisgram loading
void distinctiveMatch(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, bool norm2, bool crossCheck,float distance_factor);



float * initializeDateset(int &seasons,char (&season)[50][50],char (&dataset)[50], int &numLocations);
int getElapsedTime();


cv::Mat plotGraph(std::vector<double>& vals, int YRange[2], int gt, int estimate,int hpeak,int inter);
void progress_bar(int var, int max,int fails);
bool compare_response(cv::KeyPoint first, cv::KeyPoint second);
const char* matching_method_enum2str(enum matching_method e);
const char* hist_method_enum2str(enum hist_handling_methods e);


//initializeDateset
cv::Ptr<cv::FeatureDetector> initializeDetector(char *nameI);

void rootSift(cv::Mat *m);

cv::Ptr<cv::DescriptorExtractor> initializeDescriptor(char *nameI);

std::vector<size_t> sort_indexes(const std::vector<double> &v);
float interploation(vector <vector <vector <double> > > vec_sorted, int ims, vector<vector< double > > vec_temp , int a , int b, int image_width, vector < vector <double > > vec_bin_s, float *offsetX, int numLocations, int nn_fails, std::ofstream& nn_file_out);
void resizeFeatures(cv::Mat &descriptors1,cv::Mat &descriptors2,std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, int numFeatures);
//reads the detections from a file- in our case, used to read Super-pixel grid bounding boxes
namespace cv{

	class CV_EXPORTS FakeFeatureDetector : public FeatureDetector
	{
  public:
    FakeFeatureDetector(){}
  protected:
    //<aybe rmove VIRTUAL 
    virtual void detectImpl( const cv::Mat& image, std::vector<KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;
	};
}
void resultsOut(float displacement, int matchesSize, float difference, int (&bestHistogram)[100], std::ofstream& hist_file_out, int fails);
cv::Mat loadImage(char (&filename)[1000]);
void visualisation(	std::vector<cv::KeyPoint> keypoints1,	std::vector<cv::KeyPoint> keypoints2,double image_width, int ims,int sumDev, int histMax,std::vector<cv::DMatch> inliers_matches,float difference,float groundTruth,  float pp, cv::Mat  imga, cv::Mat imgb, char (&filename)[1000],vector <vector<double> > vec_temp, bool save, int totalTests, vector <vector <vector <double> > > vec_sorted, char (&dataset)[50], int fails);
//void FakeFeatureDetector::detectImpl( const cv::Mat& image, vector<cv::KeyPoint>& keypoints, const cv::Mat& mask ) const;
#endif
