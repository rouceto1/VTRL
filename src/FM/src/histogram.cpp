#include "histogram.hpp"

void readHistogram_max(const char* name) {
	FILE *file = fopen(name,"r");
  vector <vector <double > > vec;

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
    std::vector<double> range;

    range.push_back((32-index)*16 - 16);
    range.push_back ((32-index)*16 + 16);
    //printf("Range is %f %f\n",range[0],range[1]);
    vec.push_back(range);
  }
  std::cout<< "dataset loaded" << std::endl;
	fclose(file);
  //TODO :return
}



std::vector<vector <double> > readHistogram(const char* name,int size_of_batch, int bin_count){
  FILE *file = fopen(name,"r");
  std::vector<std::vector<double > > vec_rec;
  for (int l = 0;l<size_of_batch;l++){
    std::vector<double> vec_temp;
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


vector <vector <vector<double> > >  readHistogram_enthr(vector <vector<double> > vec_temp,double input_bin_count,double image_width){

  vector <vector <vector <double> > > vec_enthropy;

  for(size_t l = 0; l < vec_temp.size(); l++){ // for each line ... TODO: make dinamic for actual file reading
    double enthropy = 0;
    for(size_t bin = 0; bin <vec_temp[l].size(); bin ++){ //calucalte the entropy of one histogram
      float val = vec_temp[l][bin];
      if (val != 0){
        enthropy -= val * log2(val);
      } 
    } 
    double Bc = vec_temp[l].size();
    std::vector<double> range;
    std::vector<vector <double> > vec_bins_e;
    int possibly_valid_bins = int (pow(2,enthropy)); //ehtropy corespons to bits that are usable
    int reject = 0;
    //possibly_valid_bins = 3;
    for (int indx: sort_indexes(vec_temp[l])) {// sort the original histogram by size
      if (reject < possibly_valid_bins){ //ditch all bins that are under the maximum availiable enthropy
        //set the bounderies in images coordiantes
        // (Bi -Bc/2)*Iw/Bc =  (Bi -Bc/2)*Bw = Iw(Bi/Bc - 1/2)  // this is the first pixel in X coordantes in the image of given bin
        //the ending is Bw shifted
        double lower = -image_width*((indx/input_bin_count)-(1.0/2.0));//1 should be in bracket for 63 bins 0 for 64
        double upper = -image_width*((indx+1)/input_bin_count-1.0/2.0);
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
  return vec_enthropy;
}
std::vector<cv::DMatch> internalHistogram2D(std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2, int &sumDev, int &auxMax, int &histMax, int numBins, int (&histogram)[100], int (&bestHistogram)[100] , std::vector<cv::DMatch> matches, int width, int height, int granularity, int verticaLimit){
  std::vector<cv::DMatch> inliers_matches;
	//histogram assembly
  bool hist2D = false;
						if (hist2D){
              int histogram2D[width*2+granularity][height*2+granularity];
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
									if ((fabs(keypoints1[i1].pt.y-keypoints2[i2].pt.y))<verticaLimit){
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

								if ((int)((keypoints1[i1].pt.x-keypoints2[i2].pt.x + numBins/2*granularity+maxS)/granularity) == domDir && fabs(keypoints1[i1].pt.y-keypoints2[i2].pt.y)<verticaLimit)
								{
									sumDev += keypoints1[i1].pt.x-keypoints2[i2].pt.x;
									inliers_matches.push_back(matches[i]);
								}
							}
						}

            return inliers_matches;

}

std::vector<cv::DMatch> internalHistogram(std::vector<cv::KeyPoint> keypoints1,std::vector<cv::KeyPoint> keypoints2, float &displacement, int numBins, int (&histogram)[100], int (&bestHistogram)[100] , std::vector<cv::DMatch> matches, int granularity, int verticaLimit){
  std::vector<cv::DMatch> inliers_matches;
	//histogram assembly
  int auxMax = 0;
	int sumDev,histMax;
  sumDev = histMax = 0;

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
									if ((fabs(keypoints1[i1].pt.y-keypoints2[i2].pt.y))<verticaLimit){
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

								if ((int)((keypoints1[i1].pt.x-keypoints2[i2].pt.x + numBins/2*granularity+maxS)/granularity) == domDir && fabs(keypoints1[i1].pt.y-keypoints2[i2].pt.y)<verticaLimit)
								{
									sumDev += keypoints1[i1].pt.x-keypoints2[i2].pt.x;
									inliers_matches.push_back(matches[i]);
								}
							}

              if (histMax > 0) displacement = (float)sumDev/histMax;
            
            return inliers_matches;

}


std::tuple<vector  <vector <double>  >,  vector <double> > 
histogram_single_sort(vector<double> vec_temp_l,double input_bin_count,double image_width){
  int threshold_count = 0;
  double histmax = *std::max_element(vec_temp_l.begin(), vec_temp_l.end());
  for (size_t i = 0; i < vec_temp_l.size();i++){
    if (histmax * 0.5 < vec_temp_l[i]) {
      threshold_count ++;
    }
  }
  std::vector<vector <double> > vec_bins_e;
  std::vector<double> bns;
  int reject = 0;
  for (int indx: sort_indexes(vec_temp_l)){
    if (reject < threshold_count){
      double lower = -image_width*((indx/input_bin_count)-(1.0/2.0));//1 should be in bracket for 63 bins 0 for 64
      double upper = -image_width*((indx+1)/input_bin_count-1.0/2.0);
      std::vector<double> range;
      lower = -((31.0-indx)*8.0+4.0)*image_width/512.0;
      upper = -((31.0-indx)*8.0-4.0)*image_width/512.0;
      range.push_back(lower);
      range.push_back (upper);
      vec_bins_e.push_back(range);
      bns.push_back(vec_temp_l[indx]);
    }
    reject++;
  }
  return {vec_bins_e, bns};
}


std::tuple<vector <vector <vector <double> > >,
           vector <vector <double> > >
readHistogram_sort( vector <vector<double> > vec_temp,double input_bin_count,double image_width){
  vector <vector <double>  > vec_bin_s;
  vector <vector <vector <double> > > vec_sorted;

  for(size_t l = 0; l < vec_temp.size(); l++){
    auto [vec_bins_e, bns] = histogram_single_sort(vec_temp[l],input_bin_count,image_width);

    vec_sorted.push_back(vec_bins_e);
    vec_bin_s.push_back(bns);
  }
  return {vec_sorted, vec_bin_s};
}
