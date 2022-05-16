
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
  vector <int>,
           vector <vector <double> > >
readHistogram_sort( vector <vector<double> > vec_temp, int batch_size,double input_bin_count,double image_width);

vector <vector <vector<double> > >   readHistogram_enthr( vector <vector<double> > vec_temp, int batch_size,double input_bin_count,double image_width);
void readHistogram_max(const char* name);


std::vector<vector <double> > readHistogram(const char* name,int size_of_batch, int bin_count);

std::vector<cv::DMatch> internalHistogram(std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2, int &sumDev, int &auxMax, int &histMax, int numBins, int (&histogram)[100], int (&bestHistogram)[100] , std::vector<cv::DMatch> matches, bool hist2D, int width, int height, int granularity, int verticaLimit);
