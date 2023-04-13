
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
#include "helper_functions.hpp"
using std::vector;
using cv::Scalar;


std::tuple<vector <vector <vector <double> > >,
           vector <vector <double> > >
readHistogram_sort( vector <vector<double> > vec_temp, double input_bin_count,double image_width);

vector <vector <vector<double> > >   readHistogram_enthr( vector <vector<double> > vec_temp, double input_bin_count,double image_width);
void readHistogram_max(const char* name);


std::vector<vector <double> > readHistogram(const char* name,int size_of_batch, int bin_count);

int internalHistogram(std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2, float &displacement ,  int numBins, int (&histogram)[63], std::vector<int> &bestHistogram , std::vector<cv::DMatch> matches, std::vector<cv::DMatch> &inliers_matches, int granularity, int verticaLimit);
std::tuple<vector  <vector <double>  >,  vector <double> > histogram_single_sort(vector<double> vec_temp_l,double input_bin_count,double image_width);
