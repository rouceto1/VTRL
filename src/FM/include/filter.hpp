
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


void post_filter(int ims, const cv::Mat& descriptors1,const cv::Mat& descriptors2, std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches, bool norm, bool crossCheck,float range[2],float distance_factor);


void pre_filter(int ims, const cv::Mat& descriptors1,const cv::Mat& descriptors2, std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches,enum hist_handling_methods hist_method, bool crossCheck, float distance_factor ,vector <int> vec_num,vector <vector <double > > vec,vector <vector <vector <double> > > vec_enthropy,vector <vector <vector <double> > > vec_sorted);
