
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
//cut down version of for computing the original dataset
int computeOnFiles(int ims, char (&f1)[1000],char (&f2)[1000] ,  float GT = 0.0, std::vector< vector <double> > *sortedHistogram = nullptr){
  //numPictures++;
  int displacement = 99999;
  int features;
  if (METHOD !=ORIGINAL)
    displacement = computeOnTwoImages(loadImage(f1),loadImage(f2), settings,features, numFails[numFeatures/100], hist_file_out,GT, sortedHistogram);
  else
    displacement = computeOnTwoImages(loadImage(f1),loadImage(f2), settings,features, numFails[numFeatures/100], hist_file_out,GT);
  totalTests++;
  numFeats[numFeatures/100]+=features;
  progress_bar(ims,numLocations,numFails[numFeatures/100]);
  return 0;
}





//cut down version of for computing the original dataset
int computeOnDataset2(int ims){
  //numPictures++;
  char f1[1000];
  char f2[1000];
  sprintf(f1,form,dataset,season[0],ims);
  sprintf(f2,form,dataset,season[1],ims);
  float GT = (offsetX[ims+numLocations*0]-offsetX[ims+numLocations*1]);
  int displacement = 99999;
  int features;
  computeOnFiles(ims, f1,f2,GT,&vec_hist_sorted[ims]);
  totalTests++;
  numFeats[numFeatures/100]+=features;
  progress_bar(ims,numLocations,numFails[numFeatures/100]);
  return 0;

}

int main(int argc, char ** argv){

  //ros::init(argc, argv, "features");
  //ros::NodeHandle n;


  auto t1 = high_resolution_clock::now();
  init(argc,argv);
  auto t2 = high_resolution_clock::now();
  std::cout << "computing" << std::endl;


#pragma omp parallel for ordered schedule(dynamic)
  for (int ims=0;ims<numLocations;ims++) {
    computeOnDataset2(ims);
  }

  auto t3 = high_resolution_clock::now();
  hist_file_out.close();
  nn_file_out.close();

	std::cout << "Tests have finished.\n";
	int numPairs = numLocations*seasons*(seasons-1)/2;
	numFails[0] = numPairs;
	char report[1000];
	//sprintf(report,"%s/%s_%s_%s_%s.histogram",dataset,detectorName,descriptorName, matching_method_enum2str(METHOD),hist_method_enum2str(hist_method));

	n=maxFeatures/100;
	//printf("%02i %.4f Detections: %i Times: %.3f %.3f %.3f Extracted: %i %i \n",n,100.0*numFails[n]/numPairs,numFeats[n]/numPairs,timeDetection/numPictures,timeDescription/totalExtracted*1000,timeMatching/totalMatched*1000000,totalExtracted/numPictures,totalMatched/totalTests);
  std::cout << "Timing [ms]: data Loading: "<< duration_cast<milliseconds>(t2 - t1).count() << " computation time: "<< duration_cast<milliseconds>(t3 - t2).count() << " compute time per picture: " <<  duration_cast<milliseconds>(t3 - t2).count()/numLocations << std::endl;
	return 0;
}

