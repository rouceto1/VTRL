
#include "exploration.hpp"
#define CROSSCHECK false
//#define CROSSCHECK false
#define INPUT_BIN_COUNT 63.0
double BATCH_SIZE = 500.0; //14123.0 //500 for whole stromovka 14123 for nordland

//const enum matching_method METHOD = PRE_FILTER;

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
    vec_temp = readHistogram(hist_file,BATCH_SIZE,INPUT_BIN_COUNT);
    std::cerr << vec_temp[0].size() << std::endl;
    if (hist_method == hist_max){
      readHistogram_max(hist_file);
    }else if(hist_method == hist_sorted){
      auto [vec_s,vec_n,vec_b] = readHistogram_sort(vec_temp,BATCH_SIZE,INPUT_BIN_COUNT,image_width);
      std::cout<<"loading done";
      vec_sorted = vec_s;
      vec_num = vec_n;
      vec_bin_s = vec_b;
      std::cout << " fully" << std::endl;
    }else{

      vec_enthropy = readHistogram_enthr(vec_temp,BATCH_SIZE,INPUT_BIN_COUNT,image_width);
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
 
}


void imageDisplacement( cv::Mat descriptors1, cv::Mat descriptors2,std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2, int ims, int a, int b,  cv::Mat imga,cv::Mat imgb, char (&filename)[1000], float groundTruth){
					//use all features when numFeatures is 0
					if (numFeatures > 0){
            resizeFeatures(descriptors1,descriptors2,keypoints1, keypoints2, numFeatures);
					}
					std::vector<cv::DMatch> matches, inliers_matches;
					int sumDev,auxMax,histMax;
					sumDev = auxMax = histMax = 0;
					numFeats[numFeatures/100]+=(descriptors1.rows+descriptors2.rows)/2;

					// matching descriptors
					if (descriptors1.rows*descriptors2.rows > 0){
						if (METHOD==PRE_FILTER) pre_filter(ims,descriptors1,descriptors2,keypoints1,keypoints2,matches,hist_method,CROSSCHECK,distance_factor,vec_num,vec,vec_enthropy,vec_sorted);
						else if (METHOD==POST_FILTER) post_filter(ims,descriptors1,descriptors2,keypoints1,keypoints2,matches,norm2,CROSSCHECK,range,distance_factor);
						else if (METHOD==ORIGINAL) distinctiveMatch(descriptors1, descriptors2, matches, norm2, CROSSCHECK,distance_factor);
					}
          //benchmark unrestricted detector sets only
					timeMatching += getElapsedTime();
					totalMatched += descriptors1.rows*descriptors2.rows;
          int numBins = 100;
          int histogram[100];
          int bestHistogram[100];
          float difference = 0;
          float displacement = 999999;
          #pragma omp ordered
					if (matches.size() > 0){
            inliers_matches = internalHistogram(keypoints1,keypoints2, sumDev, auxMax, histMax, numBins, histogram, bestHistogram , matches,  hist2D, width, height, granularity, VERTICAL_LIMIT);
            if (histMax > 0) displacement = (float)sumDev/histMax;
					if (drawAll==false && update) draw = (abs(difference) > 35); else draw = drawAll;
					}
          difference = displacement+ groundTruth;//((offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]));
          differences.push_back(difference);
					if (fabs(difference) > 35) numFails[numFeatures/100]++;
					if (drawAll || save || (draw&&(fabs(difference) > 35))) {
            float pp = 0; //interpolation
            if (METHOD !=ORIGINAL && hist_method == hist_sorted) pp =  interploation(vec_sorted, ims, vec_temp ,a,b, image_width, vec_bin_s, offsetX, numLocations, nn_fails, nn_file_out);
            visualisation(	 keypoints1,	 keypoints2, image_width,  ims, sumDev,  histMax, inliers_matches, difference, groundTruth,  pp,  imga, imgb,  filename,vec_temp, save,totalTests,vec_sorted,dataset,numFails[numFeatures/100]);
             }
          progress_bar(ims,numLocations,numFails[numFeatures/100]);
					/*if the heading estimation error is bigger than 35 pixels, it's considered as false, otherwise it's considered correct*/
          resultsOut(displacement,inliers_matches.size(),difference,bestHistogram, hist_file_out,numFails[numFeatures/100]);
					totalTests++;
}



int computeOnTwoSavedImages(char (&f1)[1000],char (&f2)[1000]){
  //computeOnTwoImages(loadImage(f1),loadImage(f2));
}

int computeOnTwoImages(cv::Mat img1, cv::Mat img2){
  //detect
  cv::Mat descriptor1,descriptor2;
  std::vector<cv::KeyPoint> kp1, kp2;
  int ims = 1;
  //descriptorsAqusition(img1,kp1, descriptor1,upright);
  //descriptorsAqusition(img2,kp2, descriptor2,upright);
  //imageDisplacement( descriptor1, descriptor2,kp1,kp2,  ims, img1, img2, filename,((offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b])));

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
		for (int s = 0;s<seasons;s++)
		{
			timeDetection += getElapsedTime();
      descriptorsAqusition(img[s],keypoints[s], descriptors[s],upright, detector, descriptor,griefDescriptor);
			timeDescription += getElapsedTime();
			totalExtracted += descriptors[s].rows;
			numPictures++;
		}
		for (int numFeatures=maxFeatures;numFeatures>=minFeatures;numFeatures-=100)
		{
			for (int a=0;a<seasons;a++){
				for (int b=a+1;b<seasons;b++){
          imageDisplacement( descriptors[a], descriptors[b],keypoints[a], keypoints[b],  ims,  a,b, img[a], img[b], filename,(offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]));
				}
			}
		}
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
    computeOnDataset(ims);
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
  std::cout << "Timing [ms]: data Loading: "<< duration_cast<milliseconds>(t2 - t1).count() << " computation time: "<< duration_cast<milliseconds>(t3 - t2).count() << " compute time per picture: " <<  duration_cast<milliseconds>(t3 - t2).count()/numLocations << std::endl;
	fclose(summary);
	return 0;
}

