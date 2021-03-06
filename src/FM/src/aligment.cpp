
#include "aligment.hpp"
#include "filter.hpp"
#include "histogram.hpp"
#include <unistd.h>

/*exceptional case: SIFT/ORB does not work well because the sift does not provide an octave with a feature fix in older code*/
/*to select most responsive features*/
bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
{
  if (first.response > second.response) return true; else return false;
}


void resultsOut(float displacement, int matchesSize, float difference, int (&bestHistogram)[63],  std::ofstream& hist_file_out, int fails){

  if (displacement < 9999) hist_file_out << displacement << "," << matchesSize << "," << difference << "," << fails << ",";
  else hist_file_out << displacement << "," << 0 << "," << difference << "," << fails << ",";
  int numBins = sizeof(bestHistogram)/sizeof(bestHistogram[0]);
  for (int i = 0; i < numBins; i++){
    hist_file_out << bestHistogram[i];
    if (i != numBins-1) hist_file_out << ",";
  }
  hist_file_out << "\n";
}
struct settings default_config (){
  struct settings settings;
  settings.crossCheck = false;
  settings.verticaLimit = 50;
  settings.granularity = 20;
  settings.distance_factor = 0.95;
  settings.upright = upright;
  settings.featureMaximum = 1600;
  strcpy(settings.detectorName,"fast");
  strcpy(settings.descriptorName,"grief");
  return settings;
}
float twoImagesAndHistogram(cv::Mat img1, cv::Mat img2, vector<double> histogram_input ){
  int fails= 0;
  int features = 0;
  float displacement = 0.0;
  struct settings config = default_config();
  if (histogram_input.size() == 63){
    auto [sortedHistogram, bns] = histogram_single_sort(histogram_input,histogram_input.size(),img1.cols);
    displacement = computeOnTwoImages(img1, img2, config, features, fails, 0 , &sortedHistogram );
  }else {
    displacement = computeOnTwoImages(img1, img2, config, features, fails);
  }
  return displacement;
}
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
float imageDisplacementUnsave( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2, settings settings ,int &inliers_matches_count,std::vector<int> &bestHistogram, std::vector<std::vector <double> > *sortedHistogram){
  std::vector<cv::DMatch> matches;
  // matching descriptors
  if (descriptors1.rows*descriptors2.rows > 0){
    if (sortedHistogram == nullptr) 	   distinctiveMatch(descriptors1, descriptors2, matches, norm2,settings.crossCheck, settings.distance_factor);
    else pre_filter(descriptors1,descriptors2,keypoints1,keypoints2,matches,settings.crossCheck,settings.distance_factor,*sortedHistogram);
  }
  int numBins = 63;
  int histogram[63];
  float displacement = 999999;
  if (matches.size() > 0){
    inliers_matches_count = internalHistogram(keypoints1,keypoints2, displacement, numBins, histogram, bestHistogram , matches,   settings.granularity,settings.verticaLimit);
  }
  return displacement;
}


float imageDisplacement( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2,  float groundTruth,  int &fails,settings settings , std::ofstream& hist_file_out,int &inliers_matches_count, std::vector<int> &hist_out, std::vector<std::vector <double> > *sortedHistogram){
  //use all features when numFeatures is 0
  float difference = 0;
  float displ= 999999;
  int bestHistogram[63];
  if (sortedHistogram == nullptr){
    displ = imageDisplacementUnsave(descriptors1, descriptors2, keypoints1, keypoints2, settings, inliers_matches_count, hist_out);
    }else{
    displ = imageDisplacementUnsave(descriptors1, descriptors2, keypoints1, keypoints2, settings, inliers_matches_count, hist_out, sortedHistogram);
  }
  //#pragma omp ordered
      difference = displ + groundTruth;//((offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]));
  if (fabs(difference) > 35) fails++;

  if (hist_file_out.is_open())
    /*if the heading estimation error is bigger than 35 pixels, it's considered as false, otherwise it's considered correct*/
    //TODO: fix best histogram, shuld be hist_out but is now a different datatype
    resultsOut(displ,inliers_matches_count,difference, bestHistogram, hist_file_out,fails);
  return displ;
}


float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features, int  &fails,
                         std::ofstream& hist_file_out,std::vector<int> &hist_out,
                         float GT,std::vector< vector <double> > *sortedHistogram){
  //detect
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor;
  cv::GriefDescriptorExtractor *griefDescriptor = NULL;

  detector = initializeDetector(settings.detectorName);
  descriptor = initializeDescriptor(settings.descriptorName);
  if (descriptor == NULL){griefDescriptor = new cv::GriefDescriptorExtractor(32);}

  cv::Mat descriptor1,descriptor2;
  std::vector<cv::KeyPoint> kp1, kp2;
  float displacement = 999999;
  descriptorsAqusition(img1,kp1, descriptor1,settings, detector,descriptor,griefDescriptor);
  descriptorsAqusition(img2,kp2, descriptor2,settings, detector,descriptor,griefDescriptor);

  features = (descriptor1.rows+descriptor2.rows)/2;
  std::vector<cv::DMatch> inliers_matches;
  int inliers_matches_count;
  displacement = imageDisplacement( descriptor1, descriptor2,kp1, kp2, GT, fails ,settings, hist_file_out, inliers_matches_count, hist_out, sortedHistogram);

  return displacement;
}

float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features, int  &fails,
                         std::ofstream& hist_file_out,
                         float GT,std::vector< vector <double> > *sortedHistogram){
  float displacement;
  std::vector<int> hist_out;
  displacement = computeOnTwoImages(img1,img2, settings,features, fails, hist_file_out,hist_out,GT, sortedHistogram);
  return displacement;
}
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features, int  &fails,
                         float GT,std::vector< vector <double> > *sortedHistogram){
  std::ofstream hist_file_out;
  float displacement;
  std::vector<int> hist_out;
  displacement = computeOnTwoImages(img1,img2, settings,features, fails, hist_file_out,hist_out,GT, sortedHistogram);
  return displacement;
}
float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features, int  &fails,
                         std::vector<int> &hist_out,
                         float GT, std::vector< vector <double> > *sortedHistogram){
  std::ofstream hist_file_out;
  float displacement;
  displacement = computeOnTwoImages(img1,img2, settings,features, fails, hist_file_out,hist_out,GT, sortedHistogram);
  return displacement;
}

void evalOnFile(const char* f1, const char* f2, float &displacemnet, int &feature_count, int &fails, vector<double> histogram, float GT, vector<int> &histogram_out){
  int features;
  settings settings = default_config();
  cv::Mat img1 = loadImage(f1);
  auto [sortedHistogram, bns] = histogram_single_sort(histogram,histogram.size(),img1.cols);
  displacemnet = computeOnTwoImages(img1,loadImage(f2), settings,features, fails,histogram_out, GT, &sortedHistogram);
  feature_count = features;

}

void evalOnFiles(const char ** filesFrom, const char ** filesTo, double ** histogram_in, double ** hist_out,double * gt, float *displacement, int *feature_count, int hist_width,int files){
  int fails = 0;
  vector<int> histogram_out;
  //#pragma omp parallel for ordered schedule(dynamic)
  for (int i = 0; i < files; i++){
    float dsp = 0;
    int fcount = 0;
    vector<double> hist_vector (histogram_in[i], histogram_in[i] + hist_width);
    evalOnFile(filesFrom[i], filesTo[i],dsp,fcount, fails, hist_vector,gt[i], histogram_out);

    for (int w = 0; w < hist_width; w ++){
      hist_out[i][w] = histogram_out[w];
      //std::cout << histogram_out[w] << " ";
    } 
    feature_count[i]=fcount;
    displacement[i] = dsp;
    progress_bar(i,files,fails);
    //std::cout << histogram_out.data();
    //hist_out[i] = histogram_out.data();
    for (size_t w = 0; w < 5; w ++){
      //std::cout << hist_out[i];
    }
    //std::cout << std::endl;
    std::cout << "ev: " << i  << " "<<  fcount << " "  <<dsp << std::endl;
  }
}




void teachOnFiles(const char ** filesFrom, const char ** filesTo, float *displacement, int *feature_count, int files ){
  int fails = 0;
#pragma omp parallel for ordered schedule(dynamic)
  for (int i = 0; i < files; i++){
    float dsp = 0;
    int fcount = 0;
    teachOnFile(filesFrom[i], filesTo[i],dsp,fcount, fails );
    feature_count[i]=fcount;
    displacement[i] = dsp;
    progress_bar(i,files,fails);
  }
}

void teachOnFile(const char* f1, const char* f2, float &displacemnet, int &feature_count, int &fails){
  int features;
  if (f1 == nullptr) std::cerr << "shit1";
  if (f2 == nullptr) std::cerr << "shit2";
  const settings settings = default_config();
  displacemnet = computeOnTwoImages(loadImage(f1),loadImage(f2), settings,features, fails);
  feature_count = features;
}
