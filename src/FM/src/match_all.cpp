#include <sys/time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "grief/grief.h"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
 #define CROSSCHECK false
//#define CROSSCHECK false
#define INPUT_BIN_COUNT 63.0
double BATCH_SIZE = 500.0; //14123.0 //500 for whole stromovka 14123 for nordland

enum matching_method {ORIGINAL,PRE_FILTER,POST_FILTER,HIST_ONLY} ; //selector of filtering
//const enum matching_method METHOD = PRE_FILTER;
enum matching_method METHOD = ORIGINAL;


enum hist_handling_methods {hist_max,hist_sorted, hist_enthropy, hist_zero} ; //selector of specific hisgram loading
enum hist_handling_methods hist_method = hist_enthropy;

double image_width = 512.0 ; // nordland 512 stromovka 1024
double image_height = 400;
//DO NOT FORGET TO CHANGE HIST BIN ESTIAMTION IF SWITHCING DATASETS
//normalise histogram to 1
//flip histogram



#define VERTICAL_LIMIT 50
bool norm2 =true;
bool upright =true;
int nn_fails = 0;
bool update=false;
bool save=false;
bool draw=false;
bool drawAll=false;
bool normalizeSift=false;
using namespace std;
using namespace cv;
unsigned int n;
float distance_factor = 0.95;
const int granularity = 20;
const int width = 500;
const int height = 100;
int histogram2D[width*2+granularity][height*2+granularity];
bool hist2D = false;
FILE *output = NULL;
int maxFeatures=1600;
int minFeatures=1599;
int numFeatures=maxFeatures;

int numFails[1600/100+1];
int numFeats[1600/100+1];
char season[50][50];
int seasons = 0;
char dataset[50];
char descriptorName[50];
char detectorName[50];
char fileInfo[1000];

Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> descriptor;
GriefDescriptorExtractor *griefDescriptor = NULL;

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

float range[2]={-100000,100000};
char print_flag=0;
vector <vector <double > > vec;
double histEst[500];
vector <int> vec_num;
vector <vector <vector <double> > > vec_enthropy;
vector <vector <vector <double> > > vec_sorted;
vector <vector <double>  > vec_bin_s;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
int totalTests = 0;
int numPictures = 0;
std::ofstream hist_file_out;
std::ofstream nn_file_out;
char hist_file_o[1000];
char nn_file_o[1000];
vector <vector<double> > vec_temp;
char hist_file[3000] = "../stromovka_nn_1.csv"; //input histogram name
const char * form = NULL;
vector<float> differences;
//reads the detections from a file- in our case, used to read Super-pixel grid bounding boxes
namespace cv{

	class CV_EXPORTS FakeFeatureDetector : public FeatureDetector
	{
		public:
			FakeFeatureDetector(){}
		protected:
			virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
	};
}


void FakeFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{

	FILE *file = fopen(fileInfo,"r");
	keypoints.clear();
	float a,b,c,d;
	KeyPoint kp;
	while (feof(file) == 0)
	{
		fscanf(file,"%f,%f,%f,%f\n",&a,&b,&c,&d);
		kp.pt.x = (a+c)/2.0;
		kp.pt.y = (b+d)/2.0;
		kp.angle = -1;
		kp.octave = 1;
		kp.response = 1;
		kp.size = sqrt((d-b)*(c-a));
		keypoints.push_back(kp);
	}
	fclose(file);
}

/*for benchmarking purposes*/
int getElapsedTime()
{
  static int lastTime;
  struct  timeval currentTime;
  gettimeofday(&currentTime, NULL);
  int timeNow = currentTime.tv_sec*1000 + currentTime.tv_usec/1000;
  int result = timeNow - lastTime;
  lastTime = timeNow; 
  return result; 
}

/*to select most responsive features*/
bool compare_response(KeyPoint first, KeyPoint second)
{
	  if (first.response > second.response) return true; else return false;
}

template <typename T>
cv::Mat plotGraph(std::vector<T>& vals, int YRange[2], int gt, int estimate,int hpeak,int inter)
{

    auto it = minmax_element(vals.begin(), vals.end());
    float scale = 1./ceil(*it.second - *it.first); 
    float bias = *it.first;
    int rows = YRange[1] - YRange[0] + 1;
    cv::Mat image = Mat::zeros( rows, vals.size(), CV_8UC3 );
    image.setTo(0);
    for (int i = 0; i < (int)vals.size()-1; i++)
    {
        cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]), cv::Point(i+1, rows - 1 - (vals[i+1] - bias)*scale*YRange[1]), Scalar(255, 0, 0), 1);
    }

    //cv::line(image, cv::Point(gt,YRange[0]), cv::Point(gt,YRange[1]), Scalar(0, 255, 0), 1);
 
    cv::line(image, cv::Point(gt+35,YRange[0]), cv::Point(gt+35,YRange[1]), Scalar(255, 255, 0), 1);
    cv::line(image, cv::Point(gt-35,YRange[0]), cv::Point(gt-35,YRange[1]), Scalar(255, 255, 0), 1);

    //cv::line(image, cv::Point(estimate, YRange[0]), cv::Point(estimate,YRange[1]), Scalar(0, 0, 255), 1);
    cv::line(image, cv::Point(inter, YRange[0]), cv::Point(inter,YRange[1]), Scalar(255, 255, 255), 1);
    //cv::line(image, cv::Point(hpeak, YRange[0]), cv::Point(hpeak,YRange[1]), Scalar(0, 255, 255), 1);
    return image;
}

/*matching scheme*/
void distinctiveMatch(const Mat& descriptors1, const Mat& descriptors2, vector<DMatch>& matches, bool norm2= true, bool crossCheck=false)
{
   DescriptorMatcher *descriptorMatcher;
   vector<vector<DMatch> > allMatches1to2, allMatches2to1;
   if (norm2)
     descriptorMatcher = new BFMatcher(cv::NORM_L2,  false);
   else
      descriptorMatcher = new BFMatcher(cv::NORM_HAMMING, false);
   descriptorMatcher->knnMatch(descriptors1, descriptors2, allMatches1to2, 2);

   if (!crossCheck){
      for(unsigned int i=0; i < allMatches1to2.size(); i++){
        if (allMatches1to2[i].size() == 2){//check if the matches have two possible matches
          if (allMatches1to2[i][0].distance < allMatches1to2[i][1].distance * distance_factor){ // calcualte if the best distance of match is better then scond
            DMatch match = DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
            matches.push_back(match);
          }
        }else if (allMatches1to2[i].size() == 1){ // check if tehre is at least one possible match
          DMatch match = DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
          matches.push_back(match);
          //cout << "ERROR 1" << endl;
        }
      }
   }else{
     descriptorMatcher->knnMatch(descriptors2, descriptors1, allMatches2to1, 2);
     for(unsigned int i=0; i < allMatches1to2.size(); i++){
       if (allMatches1to2[i].size() == 2){
         if (allMatches2to1[allMatches1to2[i][0].trainIdx].size() == 2){
           if (allMatches1to2[i][0].distance < allMatches1to2[i][1].distance * distance_factor && allMatches2to1[allMatches1to2[i][0].trainIdx][0].distance < allMatches2to1[allMatches1to2[i][0].trainIdx][1].distance * distance_factor && allMatches1to2[i][0].trainIdx == allMatches2to1[allMatches1to2[i][0].trainIdx][0].queryIdx){
             DMatch match = DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
             matches.push_back(match);
           }
         }else if (allMatches2to1[allMatches1to2[i][0].trainIdx].size() == 1)
           if (allMatches1to2[i][0].distance  < allMatches1to2[i][1].distance * distance_factor && allMatches1to2[i][0].trainIdx == allMatches2to1[allMatches1to2[i][0].trainIdx][0].queryIdx){
             DMatch match = DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
             matches.push_back(match);
             //cout << "ERROR 2" << endl;
           }
       }else if (allMatches2to1[allMatches1to2[i][0].trainIdx].size() == 2){
         if (allMatches2to1[allMatches1to2[i][0].trainIdx][0].distance < allMatches2to1[allMatches1to2[i][0].trainIdx][1].distance * distance_factor && allMatches1to2[i][0].trainIdx == allMatches2to1[allMatches1to2[i][0].trainIdx][0].queryIdx){
           DMatch match = DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
           matches.push_back(match);
           //cout << "ERROR 3" << endl;
         }
       }else if (allMatches1to2[i][0].trainIdx == allMatches2to1[allMatches1to2[i][0].trainIdx][0].queryIdx){
         DMatch match = DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
         matches.push_back(match);
         //cout << "ERROR 4" << endl;
       }
     }
   }
   delete descriptorMatcher;
}

void rootSift(Mat *m)
{
	for (int r = 0;r<m->rows;r++)
	{
		float n1 = 0;	
		for (int c = 0;c<m->cols;c++) n1+=fabs(m->at<float>(r,c));
		for (int c = 0;c<m->cols;c++) m->at<float>(r,c)=sqrt(fabs(m->at<float>(r,c)/n1));
	}
}

/*initialize the dataset parameters*/
int initializeDateset()
{
	DIR *dpdf;
	struct dirent *epdf;
	FILE* file; 
	float dum1,dum2;
	int dum3,dum4; //for reading from a file
	char filename[1000];

	/*how many seasons are in the dataset?*/
	dpdf = opendir(dataset);
	if (dpdf != NULL)
	{
		while ((epdf = readdir(dpdf)))
		{
			if (strncmp(epdf->d_name,"season_",7)==0 && (epdf->d_type == 4 || epdf->d_type == DT_LNK))
			{
				//printf("Season directory %i %s \n",seasons,epdf->d_name);
				strcpy(season[seasons++],epdf->d_name);
			}
		}
	}else{
		fprintf(stderr,"Dataset directory not found\n");
		return -1;
	}
	for (int i=0;i<seasons;i++)sprintf(season[i],"season_%02i",i);
	//for (int i=0;i<seasons;i++)printf("Season ordered %i %s \n",i,season[i]);
	/*are there at least two?*/
	if (seasons < 2)
	{
		fprintf(stderr,"At least two directories with the prefix season_ have to be present in the datasets directory.\n");
		return -1;	
	}
	

	
	/*check the files*/
	dum4 = 0;
	for (int i = 0;i<seasons;i++){
		sprintf(filename,"%s/%s/displacements.txt",dataset,season[i]);
		file = fopen(filename,"r");
		if (file==NULL)
		{
			fprintf(stderr,"File %s not found - look at the examples on how to fill the datasets.\n",filename);
			return -1;
		}
		dum3 = 0;
		while (feof(file)==0)
		{
			if (fscanf(file,"%f %f\n",&dum1,&dum2)!=2)
			{
				fprintf(stderr,"File %s corrupt at line %i!\n",filename,dum3);
				return -1;
			}
			dum3++;
		}
		if (i==0) dum4 = dum3;
		if (dum3 != dum4)
		{
				fprintf(stderr,"Files from %s and %s dataset do not have the same number of lines %i %i!\n",season[0],season[1],dum4,dum3);
				return -1;
		}
		fclose(file);
	}
	numLocations = dum4;
	//printf("Dataset seems to be OK: %i seasons and %i locations\n",seasons,numLocations);
	/*allocate variables*/
	offsetX = (float*)malloc(sizeof(float)*numLocations*seasons);
	offsetY = (float*)malloc(sizeof(float)*numLocations*seasons);
  
	/*read offsets*/
	for (int i = 0;i<seasons;i++){
		sprintf(filename,"%s/%s/displacements.txt",dataset,season[i]);
		file = fopen(filename,"r");
		dum3 = 0;
		while (feof(file) == 0)
		{
			dum4 = fscanf(file,"%f\t%f\n",&dum1,&dum2);
			offsetX[i*numLocations+dum3] = -dum1;
			offsetY[i*numLocations+dum3] = -dum2;
			dum3++;
		}
		fclose(file);
	}
	return 0;
}

/*initialize detector*/
void initializeDetector(char *nameI)
{
	char name[100];
	strcpy(name,nameI);
	/*modifiers*/
	if (strncmp("up-",name,3)==0)	{upright = true;strcpy(name,&nameI[3]);}
	if (strncmp("norm-",name,5)==0)	{upright = false;strcpy(name,&nameI[5]);}

	/*detectors*/
	//if (strcmp("sift",  name)==0)  	detector = SIFT::create(0,3,0.0,10,1.6);   //this can be used with opecv 4.4+
	//if (strcmp("surf",  name)==0)  	detector = xfeatures2d::SurfFeatureDetector::create(400);	
	//if (strcmp("star",  name)==0)  	detector = xfeatures2d::StarDetector::create(45,0,10,8,5);
	if (strcmp("brisk", name)==0) 	detector = BRISK::create(0,4);
	if (strcmp("orb",   name)==0) 	detector = ORB::create(maxFeatures,1.2f,8,31,0,2); 
	if (strcmp("fast",  name)==0)	detector = FastFeatureDetector::create(0,true); 

	/*new ones*/
	if (strcmp("mser",  name)==0)	detector = MSER::create(2);
	if (strcmp("gftt",  name)==0)	detector = GFTTDetector::create(1600,0.01,1,3,false,0.04);
	if (strcmp("fake",  name)==0)	detector = new FakeFeatureDetector();
}

/*initialize detector*/
void initializeDescriptor(char *nameI)
{
	char name[100];
	strcpy(name,nameI);
	/*modifiers*/
	if (strncmp("root-",name,5)==0){normalizeSift= true;strcpy(name,&nameI[5]);}
	/*descriptors*/
	//if (strcmp("sift",  name)==0)  	{norm2=true;descriptor = cv::SIFT::create(0,3,0.0,10,1.6);}   //this can be used with opecv 4.4+
	//if (strcmp("surf",  name)==0)   {norm2=true;descriptor = xfeatures2d::SurfDescriptorExtractor::create(0);}
	if (strcmp("brisk", name)==0)   {norm2=false;descriptor = BRISK::create(0,4);}
	//if (strcmp("brief", name)==0)   {norm2=false;descriptor = xfeatures2d::BriefDescriptorExtractor::create(32);}
	if (strcmp("grief", name)==0)   {norm2=false;griefDescriptor = new GriefDescriptorExtractor(32);}
	if (strcmp("orb",   name)==0)   {norm2=false;descriptor = ORB::create(maxFeatures,1.2f,8,31,0,2);} 
	//if (strcmp("freak",  name)==0)	{norm2=false;descriptor = xfeatures2d::FREAK::create();}
}

const char* matching_method_enum2str(enum matching_method e)
{
	switch(e)
	{
  case ORIGINAL: return "ORIGINAL";
  case PRE_FILTER: return "PRE_FILTER";
  case POST_FILTER: return "POST_FILTER";
  case HIST_ONLY: return "HIST_ONLY";
  default: return "UNKNOWN_MATCHING_METHOD";
	}
}

const char* hist_method_enum2str(enum hist_handling_methods e)
{
	switch(e)
    {
		case hist_max: return "MAX";
		case hist_sorted: return "SORT";
		case hist_enthropy: return "ENTR";
    case hist_zero: return "ZERO";
		default: return "UNKNOWN_HISTO_METHOD";
    }
}

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  return idx;
}

vector<vector <double> > readHistogram(const char* name){
  int size_of_batch = BATCH_SIZE;
  int bin_count = INPUT_BIN_COUNT;

  FILE *file = fopen(name,"r");
  vector<vector<double > > vec_rec;
  for (int l = 0;l<size_of_batch;l++){
    vector<double> vec_temp;
    for (int i = 0;i<bin_count;i++){
      double var;
      fscanf(file,"%lf,",&var);
      vec_temp.push_back(var);
    }
    vec_rec.push_back(vec_temp);
  }
  fclose(file);

  return(vec_rec);
}
int readHistogram_max(const char* name) {
	FILE *file = fopen(name,"r");

  double var = 0;
  for (int l = 0;l<500;l++){
    double maxV = 0;
    int index = 0;
    for (int i = 0;i<63;i++){
      fscanf(file,"%lf,",&var);
      //printf("%lf,",var);
      if (maxV < var){
        maxV = var;
        index = i;
      }
    }
    fscanf(file,"%lf\n",&var);
    if (maxV < var){
      maxV = var;
      index = 63;
    }
    vector<double> range;

    range.push_back((32-index)*16 - 16);
    range.push_back ((32-index)*16 + 16);
    //printf("Range is %f %f\n",range[0],range[1]);
    vec.push_back(range);
  }
  std::cout<< "dataset loaded" << std::endl;
	fclose(file);
	return -1;

}
void readHistogram_sort(const char* name){
  vec_temp = readHistogram(name);
  vector<double>  vec_temp_abs;
  for(size_t l = 0; l < vec_temp.size(); l++){
    int count = 0;
    int threshold_count = 0;
    for (auto i : vec_temp[l]){
      //cout << i << " ";
    }
    //cout << endl;
    double histmax = *max_element(vec_temp[l].begin(), vec_temp[l].end());
    //cout <<histmax << endl;

    for (size_t i = 0; i < vec_temp[l].size();i++){
      if (histmax * 0.5 < vec_temp[l][i]) {
        threshold_count ++;
      }
    }
    vec_num.push_back(threshold_count);
    vector<vector <double> > vec_bins_e;
    vector<double> bns;
    //cout<< threshold_count << endl;
    for (int indx: sort_indexes(vec_temp[l])){
      double lower = -image_width*((indx/INPUT_BIN_COUNT)-(1.0/2.0));//1 should be in bracket for 63 bins 0 for 64
      double upper = -image_width*((indx+1)/INPUT_BIN_COUNT-1.0/2.0);
      vector<double> range;
      lower = -((31.0-indx)*8.0+4.0)*image_width/512.0;
      upper = -((31.0-indx)*8.0-4.0)*image_width/512.0;
	
      //cout << indx << "   first " << lower << " " << upper << endl ;
			range.push_back(lower);
		  range.push_back (upper);
      vec_bins_e.push_back(range);
      bns.push_back(vec_temp[l][indx]);
      count++ ;
    }
    vec_sorted.push_back(vec_bins_e);
    vec_bin_s.push_back(bns);
    if (l < 0){
      for (int i = 0 ; i < vec_num[l]; i++){
        //cout << "pos "   << vec_sorted[l][i][0] << "  "<<vec_sorted[l][i][1]  << endl;
      }
    }
    //cout << endl;}
  }
  // vec_temp.clear();
  //vec_temp_abs.clear();
}
void readHistogram_enthr(const char* name){
  vector<vector<double> > vec_temp = readHistogram(name);
  cout << vec_temp.size();
  for(size_t l = 0; l < vec_temp.size(); l++){ // for each line ... TODO: make dinamic for actual file reading
    double enthropy = 0;
    for(size_t bin = 0; bin <vec_temp[l].size(); bin ++){ //calucalte the entropy of one histogram
      float val = vec_temp[l][bin];
      if (val != 0){
        enthropy -= val * log2(val);
      } 
    } 
    double Bc = vec_temp[l].size();
    vector<double> range;
    vector<vector <double> > vec_bins_e;
    int possibly_valid_bins = int (pow(2,enthropy)); //ehtropy corespons to bits that are usable
    int reject = 0;
    //possibly_valid_bins = 3;
    for (int indx: sort_indexes(vec_temp[l])) {// sort the original histogram by size
      if (reject < possibly_valid_bins){ //ditch all bins that are under the maximum availiable enthropy
        //set the bounderies in images coordiantes
        // (Bi -Bc/2)*Iw/Bc =  (Bi -Bc/2)*Bw = Iw(Bi/Bc - 1/2)  // this is the first pixel in X coordantes in the image of given bin
        //the ending is Bw shifted
        double lower = -image_width*((indx/INPUT_BIN_COUNT)-(1.0/2.0));//1 should be in bracket for 63 bins 0 for 64
        double upper = -image_width*((indx+1)/INPUT_BIN_COUNT-1.0/2.0);
        lower = -((31.0-indx)*8.0+4.0)*image_width/512.0;
        upper = -((31.0-indx)*8.0-4.0)*image_width/512.0;

        range.push_back(lower);
        range.push_back(upper);
        vec_bins_e.push_back(range);
        range.clear();
      }
      reject ++;
    }
    vec_enthropy.push_back(vec_bins_e);
  }
}

void pre_filter(int ims, const Mat& descriptors1,const Mat& descriptors2, vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, vector<DMatch>& matches){
  // TODO BUG this does not work properly. mutliple featrus from map image are being matched to one feature in quearry image 
  vector<int> counter_2_bank;
	Mat descriptors2_chosens;
	size_t matches_previous_size=0; //for checking if size of mathces changed or not
	for (int counter_1=0 ; counter_1<descriptors1.rows ; counter_1++){
		descriptors2_chosens.release();
		counter_2_bank.clear();
		Mat descriptors2_chosens;
		vector<int> counter_2_bank;
		for (int counter_2=0 ; counter_2<descriptors2.rows ; counter_2++){
			if (hist_method == hist_max){
				if ((keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x )> vec[ims][0] && (keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x )< vec[ims][1]){
					descriptors2_chosens.push_back(descriptors2.row(counter_2));
					counter_2_bank.push_back(counter_2);
				}
			}else if (hist_method == hist_sorted){
        for(int i=0; i < vec_num[ims]; i++){
					//cout << keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x << " " << vec_sorted[ims][i][0] <<" " << vec_sorted[ims][i][1] << endl;
					if ((keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x ) >= vec_sorted[ims][i][0] && (keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x ) <= vec_sorted[ims][i][1]){
						descriptors2_chosens.push_back(descriptors2.row(counter_2));
						counter_2_bank.push_back(counter_2);
						break;
					}
				}
			}else if (hist_method == hist_enthropy){
        for(size_t i = 0; vec_enthropy[ims].size() >  i; i++){
            float origx = keypoints1[counter_1].pt.x;
            float newx= keypoints2[counter_2].pt.x;
            float shift = origx - newx;
            // cout << keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x << " " << vec_enthropy[ims][i][0] <<" " << vec_enthropy[ims][i][1] << endl;
            if (shift > vec_enthropy[ims][i][0] && shift < vec_enthropy[ims][i][1])              {
              descriptors2_chosens.push_back(descriptors2.row(counter_2));
              counter_2_bank.push_back(counter_2);
              break;
            }
        }
      }
			else
				cout<<"[-] no prefilter method was selected"<< endl;
		}
		if (descriptors2_chosens.rows > 0){
			distinctiveMatch(descriptors1.row(counter_1), descriptors2_chosens, matches, norm2, CROSSCHECK);
			if (matches.size() > matches_previous_size){
				matches_previous_size = matches.size();
				matches.back().queryIdx=counter_1;
				matches.back().trainIdx=counter_2_bank[matches.back().trainIdx];
			}
		}
	}
}

void post_filter(int ims, const Mat& descriptors1,const Mat& descriptors2, vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, vector<DMatch>& matches)
{
	vector<DMatch> matches_post_filtered_temp;
	distinctiveMatch(descriptors1, descriptors2, matches_post_filtered_temp, norm2, CROSSCHECK);
	for (size_t counter=0; counter<matches_post_filtered_temp.size() ; counter++){
		Point2f p1= keypoints1[matches_post_filtered_temp[counter].queryIdx].pt;
		Point2f p2= keypoints2[matches_post_filtered_temp[counter].trainIdx].pt;
		if ( abs(p1.x - p2.x )> range[0] && abs(p1.x - p2.x )> range[1]){
			matches.push_back(matches_post_filtered_temp[counter]);
		}
	}
}

void progress_bar(int var, int max,int fails){
  if (var%50 ==0){
    float progress = float(var)/float(max);
    int barWidth = 30;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos) std::cout << "=";
      else if (i == pos) std::cout << ">";
      else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) <<"% " << var << "/" << max <<  "  fails:"<< fails <<  "\r";
    std::cout.flush();
  }
}

void init(int argc, char ** argv){
	if (argc > 4 && strcmp(argv[4],"draw")==0) drawAll = true;
	if (argc > 4 && strcmp(argv[4],"save")==0) save = true;
	if (argc > 4 && strcmp(argv[4],"update")==0) update = true;
	if (update) output = fopen("newcon.txt","w+");
	srand (time(NULL));
	strcpy(dataset,argv[3]);
	strcpy(descriptorName,argv[2]);
	strcpy(detectorName,argv[1]);
  if (argc > 5) {
    strcpy(hist_file,dataset);
    strcat(hist_file, argv[5]);
  }

  if (argc > 6){
    switch(atoi(argv[6])){
    case 0:
      METHOD = ORIGINAL;
      break;
    case 1:
      METHOD = PRE_FILTER;
    }
  }

    if(argc > 7){
      switch(atoi(argv[7])){
      case 0:
        hist_method = hist_max;
        break;
      case 1:
        hist_method = hist_sorted;
        break;
      case 2:
        hist_method = hist_enthropy;
      }
    }
    if (argc > 8) BATCH_SIZE = stod(argv[8]);
    if(argc > 9) image_width = stod(argv[9]);// - stod(argv[9])/(INPUT_BIN_COUNT+1);
    cout << "Feature matcher : " << descriptorName << " "<< detectorName << " "<< hist_file << " " << hist_method_enum2str(hist_method) << " " << matching_method_enum2str(METHOD) << " " << BATCH_SIZE << " " << image_width <<  endl;
  if (METHOD != ORIGINAL){
    if (hist_method == hist_max){
      readHistogram_max(hist_file);
    }else if(hist_method == hist_sorted){
      readHistogram_sort(hist_file);
    }else{
      readHistogram_enthr(hist_file);
    }
  }

  initializeDateset();
	memset(numFails,0,(maxFeatures/100+1)*sizeof(int));
	memset(numFeats,0,(maxFeatures/100+1)*sizeof(int));
	sprintf(hist_file_o,"%s/%s_%s_%s_%s.csv",dataset,detectorName,descriptorName, matching_method_enum2str(METHOD),hist_method_enum2str(hist_method));
  sprintf(nn_file_o,"%s/nn_%s_%s_%s_%s.csv",dataset,detectorName,descriptorName, matching_method_enum2str(METHOD),hist_method_enum2str(hist_method));
  if (argc > 10) strcpy(hist_file_o,argv[10]);
  if (argc > 11) strcpy(nn_file_o,argv[11]);
  //cout << nn_file_o << endl;
  //cout <<hist_file_o << endl;
  hist_file_out.open (hist_file_o);
  nn_file_out.open(nn_file_o);
	//char detailFileName[1000];
	//sprintf(detailFileName,"%s/results/%s_%s.details",dataset,detectorName,descriptorName);
	//FILE* detailFile = fopen(detailFileName,"w+");
  if ( strstr (dataset,"strom") != NULL){
    form = "%s/%s/%09i.bmp";
  }else if(strstr (dataset,"nordland") != NULL){
    form = "%s/%s/%06i.png";
  }else if(strstr (dataset,"carle") != NULL){
    form = "%s/%s/%09i.bmp";
  }
  cout << "saving to"<< hist_file_o << endl;
 
}

int compute(int ims){

		char filename[1000];
		Mat img[seasons];
		Mat descriptors[seasons];
		vector<KeyPoint> keypoints[seasons];
		for (int s = 0;s<seasons;s++) {
      sprintf(filename,form,dataset,season[s],ims); //for stromoka 0.9 bmp // for nrodland 0.6 png
			img[s] = imread(filename, IMREAD_GRAYSCALE);
			if (img[s].empty()){
				printf("Can't read image %s\n",filename);
				return -1;
			}
		}
    KeyPoint kp;
		Mat dp;
		for (int s = 0;s<seasons;s++)
		{
			/*detection*/
			getElapsedTime();
			detector->detect(img[s], keypoints[s]);
			timeDetection += getElapsedTime();
			sort(keypoints[s].begin(),keypoints[s].end(),compare_response);
			/*extraction*/
			getElapsedTime();
			if (upright) for (unsigned int j = 0;j<keypoints[s].size();j++) keypoints[s][j].angle = -1;

			/*exceptional case: SIFT/ORB does not work well because the sift does not provide an octave with a feature*/
			//if (strstr(detectorName,"sift")!=NULL && strcmp(descriptorName,"orb")==0)
			//{
			//	/*providing a fake octave*/
			//	for (unsigned int j = 0;j<keypoints[s].size();j++) keypoints[s][j].octave = 1;
			//}
      // sprintf(fileInfo,"%s/%s/spgrid_regions_%09i_740.txt",dataset,season[s],ims);
      if(griefDescriptor != NULL) griefDescriptor->computeImpl(img[s],keypoints[s],descriptors[s]);
			else
				descriptor->compute(img[s],keypoints[s],descriptors[s]);
			//if (normalizeSift) rootSift(&descriptors[s]);
			timeDescription += getElapsedTime();
			totalExtracted += descriptors[s].rows;
			numPictures++;
		}
		for (int nFeatures=maxFeatures;nFeatures>=minFeatures;nFeatures-=100)
		{
			for (int a=0;a<seasons;a++){
				for (int b=a+1;b<seasons;b++){

					Mat descriptors1,descriptors2;
					vector<KeyPoint> keypoints1,keypoints2;
					descriptors1 = descriptors[a];
					descriptors2 = descriptors[b];
					keypoints1 = keypoints[a];
					keypoints2 = keypoints[b];
					numFeatures = nFeatures;
					//use all features when numFeatures is 0
					if (numFeatures > 0){
						int numRemove = max(0,descriptors1.rows-numFeatures);
						keypoints1.resize(numFeatures);
						descriptors1.pop_back(numRemove);
						numRemove = max(0,descriptors2.rows-numFeatures);
						keypoints2.resize(numFeatures);
						descriptors2.pop_back(numRemove);
					}
					vector<DMatch> matches, inliers_matches;
					int sumDev,auxMax,histMax;
					sumDev = auxMax = histMax = 0;
					numFeats[numFeatures/100]+=(descriptors1.rows+descriptors2.rows)/2;

					// matching descriptors
					matches.clear();
					inliers_matches.clear();
					getElapsedTime();
					if (descriptors1.rows*descriptors2.rows > 0){
						if (METHOD==PRE_FILTER)
              pre_filter(ims,descriptors1,descriptors2,keypoints1,keypoints2,matches);
						else if (METHOD==POST_FILTER)
							post_filter(ims,descriptors1,descriptors2,keypoints1,keypoints2,matches);
						else if (METHOD==ORIGINAL)
							distinctiveMatch(descriptors1, descriptors2, matches, norm2, CROSSCHECK);
					}
          
					//benchmark unrestricted detector sets only
					timeMatching += getElapsedTime();
					totalMatched += descriptors1.rows*descriptors2.rows;
          int numBins = 100;
          int histogram[numBins];
          int bestHistogram[numBins];

          float difference = 0;
          #pragma omp ordered
					if (matches.size() > 0){
						//histogram assembly
						if (hist2D){
							int iX = 0;
							int iY = 0;
							for(int i = 0; i < width*2+granularity; i++ ){
								for(int j = 0; j < height*2+granularity; j++ ) histogram2D[i][j]=0;
							}
							//create the histogram
							for(size_t i = 0; i < matches.size(); i++ ) {
								int i1 = matches[i].queryIdx;
								int i2 = matches[i].trainIdx;
								int iXO = (int)(keypoints1[i1].pt.x-keypoints2[i2].pt.x + width);
								int iYO = (int)(keypoints1[i1].pt.y-keypoints2[i2].pt.y + height);
								for(int ii = 0; ii < granularity; ii++ ){
									for(int ij = 0; ij < granularity; ij++ ){
										iX = iXO+ii;
										iY = iYO+ij;
										if (iX > -1 && iX < width*2 && iY > -1 && iY < height*2) histogram2D[iX][iY]++;
									}
								}
							}
							//find histogram maximum
							histMax = 0;
							for (int i = 0;i<width*2;i++){
								for (int j = 0;j<height*2;j++){
									if (histMax < histogram2D[i][j]){
										histMax = histogram2D[i][j];
										iX = i;
										iY = j;
									}
								}
							}
							sumDev = iX-width-granularity/2;
							sumDev = sumDev*histMax;
							auxMax = 0;
						}else{
							//histogram assembly
							memset(bestHistogram,0,sizeof(int)*numBins);
							histMax = 0;
							int maxS,domDir;
							maxS = domDir = 0;
							for (int s = 0;s<granularity;s++){
								memset(histogram,0,sizeof(int)*numBins);
								for( size_t i = 0; i < matches.size(); i++ )
								{
									int i1 = matches[i].queryIdx;
									int i2 = matches[i].trainIdx;
									if ((fabs(keypoints1[i1].pt.y-keypoints2[i2].pt.y))<VERTICAL_LIMIT){
										int devx = (int)(keypoints1[i1].pt.x-keypoints2[i2].pt.x + numBins/2*granularity);
										int index = (devx+s)/granularity;
										if (index > -1 && index < numBins) histogram[index]++;
									}
								}
								for (int i = 0;i<numBins;i++){
									if (histMax < histogram[i]){
										histMax = histogram[i];
										maxS = s;
										domDir = i;
										memcpy(bestHistogram,histogram,sizeof(int)*numBins);
									}
								}
							}
							auxMax=0;
							for (int i =0;i<numBins;i++){
								if (auxMax < bestHistogram[i] && bestHistogram[i] != histMax){
									auxMax = bestHistogram[i];
								}
							}
							sumDev = 0;
							for( size_t i = 0; i < matches.size(); i++ ){
								int i1 = matches[i].queryIdx;
								int i2 = matches[i].trainIdx;

								if ((int)((keypoints1[i1].pt.x-keypoints2[i2].pt.x + numBins/2*granularity+maxS)/granularity) == domDir && fabs(keypoints1[i1].pt.y-keypoints2[i2].pt.y)<VERTICAL_LIMIT)
								{
									sumDev += keypoints1[i1].pt.x-keypoints2[i2].pt.x;
									inliers_matches.push_back(matches[i]);
								}
							}
						}
						//sumDev = histEst[ims]; 
						//histMax = 1;
						if (histMax > 0) difference = ((float)sumDev/histMax)+((offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b])); else difference = 1000;
            //cout << (float)sumDev/histMax << "  " <<  offsetX[ims+numLocations*a] << "  " << offsetX[ims+numLocations*b] <<  "   "<<difference<<endl;
              //printf("[+] difference is %f \n", difference );
						differences.push_back(difference);
         						// if (histMax > 0) difference = ((float)((-vec[ims][0]-vec[ims][1])/2))+(offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]); else difference = 1000;
						// printf("bin is %f", ((float)((-vec[ims][0]-vec[ims][1])/2)));
						//if (histMax > 0) printf("\nDirection histogram %i %i %i\n",-(sumDev/histMax),histMax,auxMax); else printf("\nDirection histogram 1000 0 0\n");
						//if (histMax > 0) printf("%05i %05i %i %i %i %i %i\n",imNum1,imNum2,difference,-(sumDev/histMax),-offsetX[ims],histMax,auxMax); else printf("%05i %05i 1000 1000 %i 0 0\n",imNum1,imNum2,offsetX[ims]);
						if (drawAll==false && update) draw = (abs(difference) > 35); else draw = drawAll;
						//if (update && draw == false) fprintf(output,"Offset %i %i %i %i\n",imNum1,imNum2,offsetX[ims],offsetY[ims]);
						//if (histMax > 0) fprintf(output,"%05i %05i %i %i %i %i %i\n",a,b,-(sumDev/histMax)+offsetX[ims],-(sumDev/histMax),-offsetX[ims],histMax,auxMax); else fprintf(output,"%05i %05i 1000 1000 %i 0 0\n",a,b,offsetX[ims]);
					}else {
						difference = 1000;
						// printf("%05i %05i 1000 1000 %.1f 0 0\n",ims,ims,offsetX[ims]);
						draw = update;
					}

					/*Arash>*/
					// if(sumDev) printf("[+] sumDev/histMax is %d and range is between %d , %d\n",sumDev/histMax,vec[ims][0],vec[ims][1]);
					// else printf("[!+!] sumDev is %d and range is between %d , %d\n",sumDev,vec[ims][0],vec[ims][1]);
					/*Arash<*/
					/*if the heading estimation error is bigger than 35 pixels, it's considered as false, otherwise it's considered correct*/
					if (fabs(difference) > 35) numFails[numFeatures/100]++;


          if (histMax > 0){
            hist_file_out << (float) sumDev/histMax << "," << inliers_matches.size() << "," << difference << "," << numFails[numFeatures/100] << ",";
          }else{
            hist_file_out << 1000 << "," << 0 << "," << difference << "," << numFails[numFeatures/100] << ",";
          }
          for (int i = 0; i < numBins; i++){
            hist_file_out << bestHistogram[i];
            if (i != numBins-1) hist_file_out << ",";
          }
          hist_file_out << "\n";

          float pp=0;
          if (METHOD !=ORIGINAL && hist_method == hist_sorted){
            //cout << "sthi";
            float x0,x1,x2,x3;
            float y0,y1,y2,y3;
            float bin1= (vec_sorted[ims][0][0]+vec_sorted[ims][0][1])/2.0;
            float bin2 = 0;
            y1= (vec_sorted[ims][0][0]+vec_sorted[ims][0][1])/2.0;
            x1 = vec_temp[ims][0];
            for (int b = 1; b < vec_sorted[ims].size();b++){
              bin2= (vec_sorted[ims][b][0]+vec_sorted[ims][b][1])/2.0;
              if (abs(bin1-bin2) == 8*image_width/512){
                float sum = vec_temp[ims][0]+vec_temp[ims][b];
                float w1 =vec_temp[ims][0]/sum;
                float w2 = vec_temp[ims][0]/sum;
                }
              if (bin2-bin1 == 8*image_width/512) {
                y0=bin2;
                x0=vec_bin_s[ims][b];
              }else if(bin1-bin2 == 8*image_width/512){
                y1=bin2;
                x1=vec_bin_s[ims][b];
              }else if(bin1-bin2 == 2*8*image_width/512){
                y2=bin2;
                x2=vec_bin_s[ims][b];
              }else if(bin2-bin1 == 2*8*image_width/512){
                y3=bin2;
                x3=vec_bin_s[ims][b];
              }
            }
            float sum = x0+x1+x2+x3;
            pp = (y0*x0+y1*x1+y2*x2+y3*x3)/sum;
            //cout<<pp <<endl; 
            float nn_difference = pp +((offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]));
            //cout << y0 << " " << y2 << " b:" << pp<< " " << bin1<<" "<< x0 << " " << x2<< endl;
            if (fabs(nn_difference) > 35) nn_fails++;
            nn_file_out << pp << "," << 0 << "," << nn_difference<< "," << nn_fails << ",";
            for (int i = 0; i < 63; i++){
              nn_file_out << vec_temp[ims][i];
              if (i != 63-1) nn_file_out << ",";
              //cout << "writin" << endl;
            }
            nn_file_out << "\n";
            //cout << nn_difference<< endl;
          }
          /*Arash>*/
					//printf("Report: %s %s %09i : %i : %i : %.3f - \n",season[a],season[b],totalTests,numFails[numFeatures/100],fabs(difference) > 35,difference);
					/*Arash<*/
          progress_bar(ims,numLocations,numFails[numFeatures/100]);
					//fprintf(detailFile,"Report: %s %s %09i : %i : %i : %.3f \n",season[a],season[b],totalTests,numFails[numFeatures/100],fabs(difference) > 35,fabs(difference));
					if (drawAll || save || (draw&&(fabs(difference) > 35)))
					{
						//if (fabs(difference) > 35) cout << "DIFF: " << (sumDev/histMax) << " " << -(offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]) << std::endl;
						Mat imA,imB,img_matches,img_matches_transposed;
						vector<KeyPoint> kpA,kpB;
						KeyPoint kp;
						kpA.clear();
						kpB.clear();
						for (unsigned int s=0;s<keypoints1.size();s++){
							kp = keypoints1[s];
							kp.pt.x = keypoints1[s].pt.y;
							kp.pt.y = keypoints1[s].pt.x;
							kpA.push_back(kp);
						}
						for (unsigned int s=0;s<keypoints2.size();s++){
							kp = keypoints2[s];
							kp.pt.x = keypoints2[s].pt.y;
							kp.pt.y = keypoints2[s].pt.x;
							kpB.push_back(kp);
						}

            vector<double> hgr;
            for (int pad = 0 ; pad < 4*image_width/512.0; pad++){
              hgr.push_back(0);
            }
            int widold = 0;
            //cout << difference << endl;
            for (size_t bn  = 0; bn < vec_temp[ims].size(); bn++){
              double lower = -((31.0-bn)*8.0+4.0)*image_width/512.0;
              double upper = -((31.0-bn)*8.0-4.0)*image_width/512.0;
              //cout << endl<<lower << " " <<upper  << endl;
              for (int wid = lower; wid < upper; wid ++){
                hgr.push_back(vec_temp[ims][bn]*10);
                //cout << wid-widold << " "; 
                widold = wid;
              }
            }
            for (int pad = 0 ; pad < 4*image_width/512.0; pad++){
              hgr.push_back(0);
            }
            //cout <<endl << hgr.size() << endl;
            int rng [2]={0,int(500)};
            cout << (float)sumDev/histMax << "  gt: "<<offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b] <<  endl;
            cv::Mat gr = plotGraph(hgr,rng,-offsetX[ims+numLocations*a]+offsetX[ims+numLocations*b]+image_width/2.0,(float)sumDev/histMax+image_width/2.0,(vec_sorted[ims][0][0]+vec_sorted[ims][0][1])/2+image_width/2.0,pp+image_width/2.0);
            imshow("graph", gr);
            
						cv::transpose(img[a], imA);
						cv::transpose(img[b], imB);
						namedWindow("matches", 1);
						Scalar color(0,0,255);
						if (kpA.size() >0 && kpB.size()>0 && inliers_matches.size() >0)
						{
							if (fabs(difference) <= 35) color = Scalar(0,255,0);
						}else{
							kpA.push_back(kp);
							kpB.push_back(kp);
						}
						drawMatches(imA, kpA, imB, kpB, inliers_matches, img_matches, color, color, vector<char>());
						cv::transpose(img_matches,img_matches_transposed);
						if (save){
							sprintf(filename,"%s/%09i-%02i-%02i-A.bmp",dataset,totalTests,a,b);
							char description[1000];
							sprintf(description,"Successes: %03i Failures: %03i",totalTests+1-numFails[numFeatures/100],numFails[numFeatures/100]);
							line(img_matches_transposed,Point(15,20),Point(358,20),Scalar(0,0,0),32,0);
							putText(img_matches_transposed,description,Point(10,28), FONT_HERSHEY_SIMPLEX, 0.75,color,2.5);

							imwrite(filename,img_matches_transposed);
							imA = 0*imA;
							imB = 0*imB;
							drawMatches(imA, kpA, imB, kpB, inliers_matches, img_matches, color, color, vector<char>());
							cv::transpose(img_matches,img_matches_transposed);
							sprintf(filename,"%s/%09i-%02i-%02i-B.bmp",dataset,totalTests,a,b);
							imwrite(filename,img_matches_transposed);
							//imshow("matches", img_matches_transposed);
							//waitKey(0);
						}else{
							imshow("matches", img_matches_transposed);
							waitKey(0);
						}
					}
					// printf("Season %02i vs %02i, features %04i of %04i, location %03i of %03i \n",a,b,nFeatures,maxFeatures,ims+1,numLocations);
					totalTests++;
				}
			}
		}
    return 0;
}
int main(int argc, char ** argv){

  auto t1 = high_resolution_clock::now();
  init(argc,argv);
  auto t2 = high_resolution_clock::now();
  initializeDetector(detectorName);
  initializeDescriptor(descriptorName);
  #pragma omp parallel for ordered schedule(dynamic)
  for (int ims=0;ims<numLocations;ims++) {
    compute(ims);
  }

  auto t3 = high_resolution_clock::now();
  hist_file_out.close();
  nn_file_out.close();

	std::cout << "Tests have finished.\n";
	if (update) fclose(output);
	//fclose(detailFile);
	int numPairs = numLocations*seasons*(seasons-1)/2;
	printf("%i %i\n",totalTests,numLocations*seasons*(seasons-1)/2);
	numFails[0] = numPairs;
	char report[1000];
	sprintf(report,"%s/%s_%s_%s_%s.histogram",dataset,detectorName,descriptorName, matching_method_enum2str(METHOD),hist_method_enum2str(hist_method));
	FILE* summary = fopen(report,"w+");
	/*ARASH>*/
	for (int n=0;n<=maxFeatures/100;n++){
		fprintf(summary,"%02i %.4f Detections: %i Times: %.3f %.3f %.3f Extracted: %i %i \n",n,100.0*numFails[n]/numPairs,numFeats[n]/numPairs,timeDetection/numPictures,timeDescription/totalExtracted*1000,timeMatching/totalMatched*1000000,totalExtracted/numPictures,totalMatched/totalTests);
		//printf("%02i %.4f Detections: %i Times: %.3f %.3f %.3f Extracted: %i %i \n",n,100.0*numFails[n]/numPairs,numFeats[n]/numPairs,timeDetection/numPictures,timeDescription/totalExtracted*1000,timeMatching/totalMatched*1000000,totalExtracted/numPictures,totalMatched/totalTests);
	}

	n=maxFeatures/100;
	printf("%02i %.4f Detections: %i Times: %.3f %.3f %.3f Extracted: %i %i \n",n,100.0*numFails[n]/numPairs,numFeats[n]/numPairs,timeDetection/numPictures,timeDescription/totalExtracted*1000,timeMatching/totalMatched*1000000,totalExtracted/numPictures,totalMatched/totalTests);
	/*ARASH<*/
  cout << "Timing: dataLoad: "<< duration_cast<milliseconds>(t2 - t1).count() << " computation time: "<< duration_cast<milliseconds>(t3 - t2).count() << " c time per picture: " <<  duration_cast<milliseconds>(t3 - t2).count()/numLocations << endl;
	fclose(summary);
	return 0;
}

