
#include "helper_functions.hpp"

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


cv::Mat loadImage(char (&filename)[1000]){
  cv::Mat img;
  img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (img.empty()){
    std::cerr << "Could not read image " << filename << std::endl;
  }
  return img;
}

void resultsOut(float displacement, int matchesSize, float difference, int (&bestHistogram)[100],  std::ofstream& hist_file_out, int fails){

  if (displacement < 9999) hist_file_out << displacement << "," << matchesSize << "," << difference << "," << fails << ",";
  else hist_file_out << displacement << "," << 0 << "," << difference << "," << fails << ",";
  int numBins = sizeof(bestHistogram)/sizeof(bestHistogram[0]);
  for (int i = 0; i < numBins; i++){
    hist_file_out << bestHistogram[i];
    if (i != numBins-1) hist_file_out << ",";
  }
  hist_file_out << "\n";
}

/*to select most responsive features*/
bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
{
  if (first.response > second.response) return true; else return false;
}

void visualisation(	std::vector<cv::KeyPoint> keypoints1,	std::vector<cv::KeyPoint> keypoints2,double image_width, int ims,int sumDev, int histMax,std::vector<cv::DMatch> inliers_matches,float difference, float groundTruth , float pp, cv::Mat  imga, cv::Mat imgb, char (&filename)[1000],vector <vector<double> > vec_temp, bool save, int totalTests, vector <vector <vector <double> > > vec_sorted, char (&dataset)[50], int fails){
						//if (fabs(difference) > 35) cout << "DIFF: " << (sumDev/histMax) << " " << -(offsetX[ims+numLocations*a]-offsetX[ims+numLocations*b]) << std::endl;
            cv::Mat imA,imB,img_matches,img_matches_transposed;
						std::vector<cv::KeyPoint> kpA,kpB;
						cv::KeyPoint kp;
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

            std::vector<double> hgr;
            for (int pad = 0 ; pad < 4*image_width/512.0; pad++){
              hgr.push_back(0);
            }
            int widold = 0;
            for (size_t bn  = 0; bn < vec_temp[ims].size(); bn++){
              
              double lower = -((31.0-bn)*8.0+4.0)*image_width/512.0;
              double upper = -((31.0-bn)*8.0-4.0)*image_width/512.0;
              for (int wid = lower; wid < upper; wid ++){
                hgr.push_back(vec_temp[ims][bn]*10);
                widold = wid;
              }
            }
            for (int pad = 0 ; pad < 4*image_width/512.0; pad++){
              hgr.push_back(0);
            }
            int rng [2]={0,int(500)};
            std::cout << (float)sumDev/histMax << "  gt: "<< groundTruth <<  std::endl;
            cv::Mat gr = plotGraph(hgr,rng,-groundTruth+image_width/2.0,(float)sumDev/histMax+image_width/2.0,(vec_sorted[ims][0][0]+vec_sorted[ims][0][1])/2+image_width/2.0,pp+image_width/2.0);
            cv::imshow("graph", gr);
						cv::transpose(imga, imA);
						cv::transpose(imgb, imB);
            cv::namedWindow("matches", 1);
						Scalar color(0,0,255);
						if (kpA.size() >0 && kpB.size()>0 && inliers_matches.size() >0)
						{
							if (fabs(difference) <= 35) color = Scalar(0,255,0);
						}else{
							kpA.push_back(kp);
							kpB.push_back(kp);
						}
						drawMatches(imA, kpA, imB, kpB, inliers_matches, img_matches, color, color, std::vector<char>());
						cv::transpose(img_matches,img_matches_transposed);
						if (save){
							sprintf(filename,"%s/%09i-A.bmp",dataset,totalTests);
							char description[1000];
							sprintf(description,"Successes: %03i Failures: %03i",totalTests+1-fails,fails);
							line(img_matches_transposed,cv::Point(15,20),cv::Point(358,20),Scalar(0,0,0),32,0);
							putText(img_matches_transposed,description,cv::Point(10,28), cv::FONT_HERSHEY_SIMPLEX, 0.75,color,2.5);

							cv::imwrite(filename,img_matches_transposed);
							imA = 0*imA;
							imB = 0*imB;
							drawMatches(imA, kpA, imB, kpB, inliers_matches, img_matches, color, color, std::vector<char>());
							cv::transpose(img_matches,img_matches_transposed);
							sprintf(filename,"%s/%09i-B.bmp",dataset,totalTests);
							cv::imwrite(filename,img_matches_transposed);
							//cv::imshow("matches", img_matches_transposed);
							//cv::waitKey(0);
						}else{
							cv::imshow("matches", img_matches_transposed);
              cv::waitKey(0);
						}

}


/*matching scheme*/
void distinctiveMatch(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, bool norm2= true, bool crossCheck=false, float distance_factor = 0.95)
{
  cv::DescriptorMatcher *descriptorMatcher;
   std::vector<std::vector<cv::DMatch> > allMatches1to2, allMatches2to1;
   if (norm2)
     descriptorMatcher = new cv::BFMatcher(cv::NORM_L2,  false);
   else
     descriptorMatcher = new cv::BFMatcher(cv::NORM_HAMMING, false);
   descriptorMatcher->knnMatch(descriptors1, descriptors2, allMatches1to2, 2);

   if (!crossCheck){
      for(unsigned int i=0; i < allMatches1to2.size(); i++){
        if (allMatches1to2[i].size() == 2){//check if the matches have two possible matches
          if (allMatches1to2[i][0].distance < allMatches1to2[i][1].distance * distance_factor){ // calcualte if the best distance of match is better then scond
            cv::DMatch match = cv::DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
            matches.push_back(match);
          }
        }else if (allMatches1to2[i].size() == 1){ // check if tehre is at least one possible match
          cv::DMatch match = cv::DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
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
             cv::DMatch match = cv::DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
             matches.push_back(match);
           }
         }else if (allMatches2to1[allMatches1to2[i][0].trainIdx].size() == 1)
           if (allMatches1to2[i][0].distance  < allMatches1to2[i][1].distance * distance_factor && allMatches1to2[i][0].trainIdx == allMatches2to1[allMatches1to2[i][0].trainIdx][0].queryIdx){
             cv::DMatch match = cv::DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
             matches.push_back(match);
             //cout << "ERROR 2" << endl;
           }
       }else if (allMatches2to1[allMatches1to2[i][0].trainIdx].size() == 2){
         if (allMatches2to1[allMatches1to2[i][0].trainIdx][0].distance < allMatches2to1[allMatches1to2[i][0].trainIdx][1].distance * distance_factor && allMatches1to2[i][0].trainIdx == allMatches2to1[allMatches1to2[i][0].trainIdx][0].queryIdx){
           cv::DMatch match = cv::DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
           matches.push_back(match);
           //cout << "ERROR 3" << endl;
         }
       }else if (allMatches1to2[i][0].trainIdx == allMatches2to1[allMatches1to2[i][0].trainIdx][0].queryIdx){
         cv::DMatch match = cv::DMatch(allMatches1to2[i][0].queryIdx, allMatches1to2[i][0].trainIdx, allMatches1to2[i][0].distance);
         matches.push_back(match);
         //cout << "ERROR 4" << endl;
       }
     }
   }
   delete descriptorMatcher;
}



cv::Mat plotGraph(std::vector<double> & vals, int YRange[2], int gt, int estimate,int hpeak,int inter)
{

    auto it = minmax_element(vals.begin(), vals.end());
    float scale = 1./ceil(*it.second - *it.first); 
    float bias = *it.first;
    int rows = YRange[1] - YRange[0] + 1;
    cv::Mat image = cv::Mat::zeros( rows, vals.size(), CV_8UC3 );
    image.setTo(0);
    for (int i = 0; i < (int)vals.size()-1; i++)
    {
        cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]), cv::Point(i+1, rows - 1 - (vals[i+1] - bias)*scale*YRange[1]), Scalar(255, 0, 0), 1);
    }
    //cv::line(image, cv::Point(gt,YRange[0]), cv::Point(gt,YRange[1]), Scalar(0, 255, 0), 1);
    cv::line(image, cv::Point(gt+35,YRange[0]), cv::Point(gt+35,YRange[1]), Scalar(255, 255, 0), 1);
    cv::line(image, cv::Point(gt-35,YRange[0]), cv::Point(gt-35,YRange[1]), Scalar(255, 255, 0), 1);

    cv::line(image, cv::Point(estimate, YRange[0]), cv::Point(estimate,YRange[1]), Scalar(0, 0, 255), 1);
    cv::line(image, cv::Point(inter, YRange[0]), cv::Point(inter,YRange[1]), Scalar(255, 255, 255), 1);
    cv::line(image, cv::Point(hpeak, YRange[0]), cv::Point(hpeak,YRange[1]), Scalar(0, 255, 255), 1);
    return image;
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

float interploation(vector <vector <vector <double> > > vec_sorted, int ims, vector<vector< double > > vec_temp , int a , int b, int image_width, vector < vector <double > > vec_bin_s, float *offsetX, int numLocations, int nn_fails, std::ofstream& nn_file_out){ 
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
  float pp = (y0*x0+y1*x1+y2*x2+y3*x3)/sum;
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
  return pp;
}



void resizeFeatures(cv::Mat &descriptors1,cv::Mat &descriptors2,std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, int numFeatures){
  int numRemove = cv::max(0,descriptors1.rows-numFeatures);
  keypoints1.resize(numFeatures);
  descriptors1.pop_back(numRemove);
  numRemove = cv::max(0,descriptors2.rows-numFeatures);
  keypoints2.resize(numFeatures);
  descriptors2.pop_back(numRemove);
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



/*initialize the dataset parameters*/
float * initializeDateset(int &seasons,char (&season)[50][50],char (&dataset)[50], int &numLocations)
{
	DIR *dpdf;
	struct dirent *epdf;
	FILE* file; 
	float dum1,dum2;
	int dum3,dum4; //for reading from a file
	char filename[1000];

  float *offsetX;
  float *offsetY;
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
		return NULL;
	}
	for (int i=0;i<seasons;i++)sprintf(season[i],"season_%02i",i);
	//for (int i=0;i<seasons;i++)printf("Season ordered %i %s \n",i,season[i]);
	/*are there at least two?*/
	if (seasons < 2)
    {
      fprintf(stderr,"At least two directories with the prefix season_ have to be present in the datasets directory.\n");
      return NULL;	
    }


	/*check the files*/
	dum4 = 0;
	for (int i = 0;i<seasons;i++){
		sprintf(filename,"%s/%s/displacements.txt",dataset,season[i]);
		file = fopen(filename,"r");
		if (file==NULL)
		{
			fprintf(stderr,"File %s not found - look at the examples on how to fill the datasets.\n",filename);
			return NULL;
		}
		dum3 = 0;
		while (feof(file)==0)
		{
			if (fscanf(file,"%f %f\n",&dum1,&dum2)!=2)
			{
				fprintf(stderr,"File %s corrupt at line %i!\n",filename,dum3);
				return NULL;
			}
			dum3++;
		}
		if (i==0) dum4 = dum3;
		if (dum3 != dum4)
		{
				fprintf(stderr,"Files from %s and %s dataset do not have the same number of lines %i %i!\n",season[0],season[1],dum4,dum3);
				return NULL;
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
	return offsetX;
}



/*initialize detector*/
cv::Ptr<cv::FeatureDetector> initializeDetector(char *nameI)
{
  cv::Ptr<cv::FeatureDetector> detector;
	char name[100];
	strcpy(name,nameI);
	/*modifiers*/
	if (strncmp("up-",name,3)==0)	{upright = true;strcpy(name,&nameI[3]);}
	if (strncmp("norm-",name,5)==0)	{upright = false;strcpy(name,&nameI[5]);}

	/*detectors*/
	//if (strcmp("sift",  name)==0)  	detector = SIFT::create(0,3,0.0,10,1.6);   //this can be used with opecv 4.4+
	//if (strcmp("surf",  name)==0)  	detector = xfeatures2d::SurfFeatureDetector::create(400);	
	//if (strcmp("star",  name)==0)  	detector = xfeatures2d::StarDetector::create(45,0,10,8,5);
	if (strcmp("brisk", name)==0) 	detector = cv::BRISK::create(0,4);
	if (strcmp("orb",   name)==0) 	detector = cv::ORB::create(maxFeatures,1.2f,8,31,0,2); 
	if (strcmp("fast",  name)==0)	  detector = cv::FastFeatureDetector::create(0,true); 

	/*new ones*/
	if (strcmp("mser",  name)==0)	detector = cv::MSER::create(2);
	if (strcmp("gftt",  name)==0)	detector = cv::GFTTDetector::create(1600,0.01,1,3,false,0.04);
	if (strcmp("fake",  name)==0)	detector = new cv::FakeFeatureDetector();
  return detector;
}

void rootSift(cv::Mat *m)
{
	for (int r = 0;r<m->rows;r++)
    {
      float n1 = 0;	
      for (int c = 0;c<m->cols;c++) n1+=fabs(m->at<float>(r,c));
      for (int c = 0;c<m->cols;c++) m->at<float>(r,c)=sqrt(fabs(m->at<float>(r,c)/n1));
    }
}

/*initialize detector*/
cv::Ptr<cv::DescriptorExtractor> initializeDescriptor(char *nameI)
{
  cv::Ptr<cv::DescriptorExtractor> descriptor; 
	char name[100];
	strcpy(name,nameI);
	/*modifiers*/
	if (strncmp("root-",name,5)==0){normalizeSift= true;strcpy(name,&nameI[5]);}
	/*descriptors*/
	//if (strcmp("sift",  name)==0)  	{norm2=true;descriptor = cv::SIFT::create(0,3,0.0,10,1.6);}   //this can be used with opecv 4.4+
	//if (strcmp("surf",  name)==0)   {norm2=true;descriptor = xfeatures2d::SurfDescriptorExtractor::create(0);}
	if (strcmp("brisk", name)==0)   {norm2=false;descriptor = cv::BRISK::create(0,4);}
	//if (strcmp("brief", name)==0)   {norm2=false;descriptor = xfeatures2d::BriefDescriptorExtractor::create(32);}
	if (strcmp("grief", name)==0)   {norm2=false;descriptor = NULL;}
	if (strcmp("orb",   name)==0)   {norm2=false;descriptor = cv::ORB::create(maxFeatures,1.2f,8,31,0,2);} 
	//if (strcmp("freak",  name)==0)	{norm2=false;descriptor = xfeatures2d::FREAK::create();}
  return descriptor;
}


vector<size_t> sort_indexes(const vector<double> &v) {
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  return idx;
}


void cv::FakeFeatureDetector::detectImpl( const cv::Mat& image, vector<cv::KeyPoint>& keypoints, const cv::Mat& mask ) const
{

	FILE *file = fopen(fileInfo,"r");
	keypoints.clear();
	float a,b,c,d;
  cv::KeyPoint kp;
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
