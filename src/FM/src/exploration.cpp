
#include "exploration.hpp"
//#define CROSSCHECK false
#define INPUT_BIN_COUNT 63.0
double BATCH_SIZE = 500.0; //14123.0 //500 for whole stromovka 14123 for nordland
bool  crossCheck= false;
//const enum matching_method METHOD = PRE_FILTER;
struct settings settings;
//enum matching_method {ORIGINAL,PRE_FILTER,POST_FILTER,HIST_ONLY} ; //selector of filtering
enum matching_method METHOD = ORIGINAL;


//enum hist_handling_methods {hist_max,hist_sorted, hist_enthropy, hist_zero} ; //selector of specific hisgram loading
enum hist_handling_methods hist_method = hist_enthropy;

double image_width = 512.0 ; // nordland 512 stromovka 1024
double image_height = 400;
//DO NOT FORGET TO CHANGE HIST BIN ESTIAMTION IF SWITHCING DATASETS
//normalise histogram to 1
//flip histogram


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
    if (argc > 8) BATCH_SIZE = std::stod(argv[8]);
    if(argc > 9) image_width = std::stod(argv[9]);// - stod(argv[9])/(INPUT_BIN_COUNT+1);
    std::cout << "Feature matcher : " << descriptorName << " "<< detectorName << " "<< hist_file << " " << hist_method_enum2str(hist_method) << " " << matching_method_enum2str(METHOD) << " " << BATCH_SIZE << " " << image_width <<  std::endl;
  if (METHOD != ORIGINAL){
    vec_hist = readHistogram(hist_file,BATCH_SIZE,INPUT_BIN_COUNT);
    std::cerr << vec_hist[0].size() << std::endl;
    if (hist_method == hist_max){
      readHistogram_max(hist_file);
    }else if(hist_method == hist_sorted){
      auto [vec_s,vec_b] = readHistogram_sort(vec_hist,INPUT_BIN_COUNT,image_width);
      std::cout<<"loading done";
      vec_hist_sorted = vec_s;
      vec_bin_s = vec_b;
      std::cout << " fully" << std::endl;
    }else{

      vec_hist_sorted = readHistogram_enthr(vec_hist,INPUT_BIN_COUNT,image_width);
    }
  }
	offsetX = initializeDateset(seasons, season, dataset, numLocations);
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
  std::cout << "saving to"<< hist_file_o << std::endl;
  settings.crossCheck = crossCheck;
  settings.verticaLimit = VERTICAL_LIMIT;
  settings.granularity = granularity;
  settings.distance_factor = distance_factor;
  settings.upright = upright;
  settings.featureMaximum = maxFeatures;
  strcpy(settings.detectorName,detectorName);
  strcpy(settings.descriptorName,descriptorName);
}


//int computeOnTwoSavedImages(char (&f1)[1000],char (&f2)[1000], ){
  //computeOnTwoImages(loadImage(f1),loadImage(f2));
//  return 0;
//}
struct settings default_config (){

  settings.crossCheck = false;
  settings.verticaLimit = 50;
  settings.granularity = 20;
  settings.distance_factor = 0.95;
  settings.upright = upright;
  settings.featureMaximum = 1600;
  strcpy(settings.detectorName,"fast");
  strcpy(settings.descriptorName,"grief");
}


float computeOnTwoImages(cv::Mat img1, cv::Mat img2 , struct settings settings,int &features, int  &fails, float GT = 0.0,std::vector< vector <double> > *sortedHistogram = nullptr){
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
  displacement = imageDisplacement( descriptor1, descriptor2,kp1, kp2, GT, fails ,settings, hist_file_out, inliers_matches, sortedHistogram);
  return displacement;
}



int computeOnDataset2(int ims){
		char f1[1000];
    char f2[1000];
    std::vector < cv::Mat > img;
    cv::Mat descriptors[seasons];
		std::vector<cv::KeyPoint> keypoints[seasons];
    float GT = (offsetX[ims+numLocations*0]-offsetX[ims+numLocations*1]);
    sprintf(f1,form,dataset,season[0],ims); //for stromoka 0.9 bmp // for nrodland 0.6 png
    sprintf(f2,form,dataset,season[1],ims); //for stromoka 0.9 bmp // for nrodland 0.6 png
    cv::KeyPoint kp;
    cv::Mat dp;
    timeDetection += getElapsedTime();
    //descriptorsAqusition(img[s],keypoints[s], descriptors[s],settings, detector, descriptor,griefDescriptor);
    timeDescription += getElapsedTime();
    //totalExtracted += descriptors[s].rows;
    numPictures++;
    int displacement = 99999;
    int features;
    if (METHOD !=ORIGINAL)
      displacement = computeOnTwoImages(loadImage(f1),loadImage(f2), settings,features, numFails[numFeatures/100],GT, &vec_hist_sorted[ims]);
    else
      displacement = computeOnTwoImages(loadImage(f1),loadImage(f2), settings,features, numFails[numFeatures/100],GT);
    totalTests++;
    numFeats[numFeatures/100]+=features;
    progress_bar(ims,numLocations,numFails[numFeatures/100]);
    return 0;

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


int computeOnDataset(int ims){

		char filename[1000];
    std::vector < cv::Mat > img;
    cv::Mat descriptors[seasons];
		std::vector<cv::KeyPoint> keypoints[seasons];
		for (int s = 0;s<seasons;s++) {
      sprintf(filename,form,dataset,season[s],ims); //for stromoka 0.9 bmp // for nrodland 0.6 png
     img.push_back(loadImage(filename));
    }
    cv::KeyPoint kp;
    cv::Mat dp;
		for (int s = 0; s < seasons;s++)
		{
			timeDetection += getElapsedTime();
      descriptorsAqusition(img[s],keypoints[s], descriptors[s],settings, detector, descriptor,griefDescriptor);
			timeDescription += getElapsedTime();
			totalExtracted += descriptors[s].rows;
			numPictures++;
		}
    int displacement;
		for (int numFeatures=maxFeatures;numFeatures>=minFeatures;numFeatures-=100)
		{
			for (int a=0;a<seasons;a++){
				for (int b=a+1;b<seasons;b++){
          float GT = (offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]);
          //std::cout << GT-(offsetX[ims+numLocations*0]-offsetX[ims+numLocations*1]) << std::endl;
          //benchmark unrestricted detector sets only
					timeMatching += getElapsedTime();
					totalMatched += descriptors[1].rows*descriptors[2].rows;
          std::vector<cv::DMatch> inliers_matches;
          if (METHOD !=ORIGINAL) 
            displacement = imageDisplacement( descriptors[a], descriptors[b],keypoints[a], keypoints[b], GT, numFails[numFeatures/100] , settings, hist_file_out, inliers_matches, &vec_hist_sorted[ims]);
          else
            displacement = imageDisplacement( descriptors[a], descriptors[b],keypoints[a], keypoints[b],  GT, numFails[numFeatures/100], settings,  hist_file_out, inliers_matches);


          totalTests++;
          numFeats[numFeatures/100]+=(descriptors[a].rows+descriptors[b].rows)/2;
          if (drawAll || save) {
            float pp = 0; //TODO : fx interpolation
            if (METHOD !=ORIGINAL && hist_method == hist_sorted)
              pp =  interploation(vec_hist_sorted[ims],  vec_hist[ims] , image_width, vec_bin_s[ims],  numLocations, nn_fails, nn_file_out, GT);
            visualisation(keypoints[a],	 keypoints[b], image_width, displacement, inliers_matches,  GT,  pp,  img[a], img[b],  filename,vec_hist[ims], save,vec_hist_sorted[ims],dataset, numFails[numFeatures/100]);
          }
        }
      }
		}
    progress_bar(ims,numLocations,numFails[numFeatures/100]);
    return 0;
}


int main(int argc, char ** argv){

  //ros::init(argc, argv, "features");
  //ros::NodeHandle n;


  auto t1 = high_resolution_clock::now();
  init(argc,argv);
  auto t2 = high_resolution_clock::now();
  detector = initializeDetector(detectorName);
  descriptor = initializeDescriptor(descriptorName);
  if (descriptor == NULL){griefDescriptor = new cv::GriefDescriptorExtractor(32);}
  std::cout << "computing" << std::endl;
#pragma omp parallel for ordered schedule(dynamic)
  for (int ims=0;ims<numLocations;ims++) {
    computeOnDataset2(ims);
  }

  auto t3 = high_resolution_clock::now();
  hist_file_out.close();
  nn_file_out.close();

	std::cout << "Tests have finished.\n";
	if (update) fclose(output);
	//fclose(detailFile);
	int numPairs = numLocations*seasons*(seasons-1)/2;
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
  std::cout << "Timing [ms]: data Loading: "<< duration_cast<milliseconds>(t2 - t1).count() << " computation time: "<< duration_cast<milliseconds>(t3 - t2).count() << " compute time per picture: " <<  duration_cast<milliseconds>(t3 - t2).count()/numLocations << std::endl;
	fclose(summary);
	return 0;
}

