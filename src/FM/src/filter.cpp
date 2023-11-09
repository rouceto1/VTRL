
#include "filter.hpp"

//vec_in is vector wehere ewach row is each histogram bin number with descending priority.
void pre_filter( const cv::Mat& descriptors1,const cv::Mat& descriptors2, std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2,
                 std::vector<cv::DMatch>& matches, bool crossCheck, float distance_factor ,
                 vector <vector <double > > vec_in ,enum hist_handling_methods hist_method){
  // TODO BUG this does not work properly. mutliple featrus from map image are being matched to one feature in quearry image 
  std::vector<int> counter_2_bank;
  cv::Mat descriptors2_chosens;
	size_t matches_previous_size=0; //for checking if size of mathces changed or not
	for (int counter_1=0 ; counter_1<descriptors1.rows ; counter_1++){
		descriptors2_chosens.release();
		counter_2_bank.clear();
        cv::Mat descriptors2_chosens;
		std::vector<int> counter_2_bank;
		for (int counter_2=0 ; counter_2<descriptors2.rows ; counter_2++){
			if (hist_method == hist_max){
				if ((keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x )> vec_in[0][0] && (keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x )< vec_in[0][1]){
					descriptors2_chosens.push_back(descriptors2.row(counter_2));
					counter_2_bank.push_back(counter_2);
				}
			}else if (hist_method == hist_sorted){
                for(size_t i=0; i < vec_in.size(); i++){
                    //cout << keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x << " " << vec_sorted[ims][i][0] <<" " << vec_sorted[ims][i][1] << endl;
                    if ((keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x ) >= vec_in[i][0] && (keypoints1[counter_1].pt.x - keypoints2[counter_2].pt.x ) <= vec_in[i][1]){
                        descriptors2_chosens.push_back(descriptors2.row(counter_2));
                        counter_2_bank.push_back(counter_2);
                        break; //Should be no break but rather creation of the mask
                    }
                }
            }else if (hist_method == hist_enthropy) {
                for (size_t i = 0; vec_in.size() > i; i++) {
                    float origx = keypoints1[counter_1].pt.x;
                    float newx = keypoints2[counter_2].pt.x;
                    float shift = origx - newx;
                    // cout << keypoints 1[counter_1].pt.x - keypoints2[counter_2].pt.x << " " << vec_enthropy[ims][i][0] <<" " << vec_enthropy[ims][i][1] << endl;
                    if (shift > vec_in[i][0] && shift < vec_in[i][1]) {
                        descriptors2_chosens.push_back(descriptors2.row(counter_2));
                        counter_2_bank.push_back(counter_2);
                        break;
                    }
                }
            }
            else std::cout<<"[-] no prefilter method was selected"<< std::endl;
		}
		if (descriptors2_chosens.rows > 0){
            // the destinctive amtch is called for only one map descriptor at the time. Thereforem matches can only increase by one each iteration. TODO make this masked
			distinctiveMatch(descriptors1.row(counter_1), descriptors2_chosens, matches, norm2, crossCheck,distance_factor);
			if (matches.size() > matches_previous_size){
				matches_previous_size = matches.size();
				matches.back().queryIdx=counter_1;
				matches.back().trainIdx=counter_2_bank[matches.back().trainIdx];
			}
		}
	}
}
void post_filter(const cv::Mat& descriptors1,const cv::Mat& descriptors2, std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches, bool norm, bool crossCheck, float distance_factor)
{
	std::vector<cv::DMatch> matches_post_filtered_temp;
  float range[2]={-100000,100000};
	distinctiveMatch(descriptors1, descriptors2, matches_post_filtered_temp, norm, crossCheck,distance_factor);
	for (size_t counter=0; counter<matches_post_filtered_temp.size() ; counter++){
		cv::Point2f p1= keypoints1[matches_post_filtered_temp[counter].queryIdx].pt;
		cv::Point2f p2= keypoints2[matches_post_filtered_temp[counter].trainIdx].pt;
		if ( abs(p1.x - p2.x )> range[0] && abs(p1.x - p2.x )> range[1]){
			matches.push_back(matches_post_filtered_temp[counter]);
		}
	}
}
