#ifndef EXPLORATION
#define EXPLORATION


#include <sys/time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "../src/grief/grief.h"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include "ros/ros.h"
#include "histogram.hpp"
#include "helper_functions.hpp"
#include "filter.hpp"
#include "aligment.hpp"
using std::vector;
using cv::Scalar;
int nn_fails = 0;
bool update=false;
bool save=false;
bool draw=false;
bool drawAll=false;
unsigned int n;
float distance_factor = 0.95;
const int granularity = 20;
const int width = 500;
const int height = 100;
int histogram2D[width*2+granularity][height*2+granularity];
bool hist2D = false;
FILE *output = NULL;
int minFeatures=1599;
int numFeatures=maxFeatures;

int numFails[1600/100+1];
int numFeats[1600/100+1];
char season[50][50];
int seasons = 0;
char dataset[50];
char descriptorName[50];
char detectorName[50];

cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorExtractor> descriptor;
cv::GriefDescriptorExtractor *griefDescriptor = NULL;

#define VERTICAL_LIMIT 50
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
char hist_file[5000] = "../stromovka_nn_1.csv"; //input histogram name


/*benchmarking time*/
float timeDetection = 0;
float timeDescription = 0;
float timeMatching = 0;
int totalExtracted = 0;
int totalMatched = 0;

int *seq1;
int *seq2;
float *offsetX;
float *offsetY;
int numLocations = 0;
int wait_a=0;

char print_flag=0;
vector <vector <double > > vec;
double histEst[500];
vector <int> vec_num;
vector <vector <double> > vec_hist;
vector <vector <vector <double> > > vec_hist_sorted;
vector <vector <double>  > vec_bin_s;
int totalTests = 0;
int numPictures = 0;
std::ofstream hist_file_out;
std::ofstream nn_file_out;
char hist_file_o[1000];
char nn_file_o[1000];
vector <vector<double> > vec_temp;
const char * form = NULL;

#endif
