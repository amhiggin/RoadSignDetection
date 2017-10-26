#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include "Utilities.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

//GLOBALS
Mat src, hsv, hue; //ground truth
Mat img, hsvImg, hueImg;//sample image
MatND backproj;//the backprojected image
Mat black_groundtruth, white_groundtruth;//the ground truths for the black and white pixels
Mat black_pixel_image, white_pixel_image;
int bins = 3;//histogram bins
int TN = 0, FN = 0, TP = 0, FP = 0;//false/true positives/negatives for metrics

								   //FUNCTIONS
void printMetrics();//function to print the metrics
void getRedMetrics(Mat ground_truth, Mat back_projection);//function to calculate the metrics
void getBlackandWhiteMetrics(String colour);
void Hist_and_Backproj();//to histogram, threshold and back-project the red pixel image
void getBlackAndWhitePixels();	//to get the black and white pixels in the image



int main(int argc, const char** argv) {

	int i = 0;//counter
	char* file_location = "Media/";
	char* test_image_name = "RoadSignsComposite1.jpg"; char* ground_truth_image_name = "RoadSignsCompositeGroundTruth.jpg";

	// Load the image for backprojection (test_pixels_image)- this is the image that we want to isolate the red pixels in
	string filename(file_location);
	filename.append(test_image_name);
	img = imread(filename, -1);
	if (img.empty()) {
		cout << "Could not open " << img << endl;
		return -1;
	}
	imshow("This is the sample image: ", img);
	//load the ground truth
	filename = file_location;
	filename.append(ground_truth_image_name);
	src = imread(filename, -1);
	if (src.empty())
	{
		cout << "Could not open " << src << endl;
		return -2;
	}
	imshow("This is the gnd truth image: ", src);


	//convert both to HSL colour space
	cvtColor(img, hsvImg, CV_BGR2HSV);	//sample
	cvtColor(src, hsv, CV_BGR2HSV);		//gnd truth
	imshow("Sample hsv image", hsvImg);
	//only use the hue value
	hueImg.create(hsv.size(), hsv.depth());	//for the sample image
	hue.create(hsv.size(), hsv.depth());	//for the ground truth
	int channels[] = { 0,0 };
	mixChannels(&hsvImg, 1, &hueImg, 1, channels, 1);
	mixChannels(&hsv, 1, &hue, 1, channels, 1);
	imshow("Sample hue image", hueImg);
	//backproject the sample image onto the ground truth
	Hist_and_Backproj();
	//calculate the metrics for the backprojected image
	getRedMetrics(src, backproj);

	//distinguish between black and white pixels, and save in the global Mats
	getBlackAndWhitePixels();
	//calculate the metrics for the black and white images
	getBlackandWhiteMetrics("black");
	getBlackandWhiteMetrics("white");

	waitKey(0);
	return 0;
}


//calculates normalised hist and performs thresholding and backprojection 
void Hist_and_Backproj()
{
	MatND hist;
	MatND hist_trs;			//the thresholded histogram
	int histSize = MAX(bins, 2);
	float hue_range[] = { 0, 180 };	//this is the range of red
	const float* ranges = { hue_range };
	int const max_value = 255;	 //max threshold

								 // Get the Histogram and normalize it
	calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	// Thresholding the histogram
	threshold(hist, hist_trs, 176, max_value, THRESH_BINARY);
	// Get Backprojection image for the sample image
	calcBackProject(&hueImg, 1, 0, hist_trs, backproj, &ranges, 1, true);
	// Draw the backproj
	imshow("Histogram Back-Projection", backproj);

	// Draw the histogram
	int w = 400; int h = 400;
	int bin_w = cvRound((double)w / histSize);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);
	for (int i = 0; i < bins; i++) {
		rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist_trs.at<float>(i)*h / 255.0)), Scalar(255, 255, 255), -1);
	}
	imshow("Histogram Image", histImg);
}


//isolate the black and white pixels in the road sign
void getBlackAndWhitePixels() {

	Mat grey_img, grey_gndtruth, inverted_backproj, grey_cc, anded_img, inverted_contours_image;
	cvtColor(img, grey_img, CV_BGR2GRAY);//grey version of sample image
	cvtColor(src, grey_gndtruth, CV_BGR2GRAY);//grey version of ground truth
	invertImage(backproj, inverted_backproj);//to give a black background when doing connected components analysis (easier).

											 //connected components analysis on the back-projection image
	int contour_number = 0;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//find contours in image
	findContours(inverted_backproj, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	Mat contours_image = Mat::zeros(img.size(), CV_8UC3);
	//give them colours
	for (contour_number = 0; (contour_number<(int)contours.size()); contour_number++) {
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		drawContours(contours_image, contours, contour_number, colour, CV_FILLED, 8, hierarchy);
	}
	imshow("Connected Components (on Back-Projection image) in colour", contours_image);

	//connected components image in greyscale
	cvtColor(contours_image, grey_cc, CV_BGR2GRAY);
	threshold(grey_cc, grey_cc, 60, 255, THRESH_BINARY);
	imshow("Connected Components(on Back - Projection image) in greyscale", grey_cc);

	//Bitwise-AND the grey contours image with the backprojected image to remove the red borders
	Mat new_image = backproj.clone();
	bitwise_and(grey_cc, new_image, anded_img);
	imshow("Showing only black and white pixel regions (Red removed)", anded_img);

	//get greyscale version of sample image with all exterior regions set to black
	bitwise_and(grey_img, anded_img, anded_img);
	imshow("ANDed black and white regions with sample image", anded_img);

	//white pixels image from sample image
	threshold(anded_img, white_pixel_image, 76, 255, THRESH_BINARY);
	imshow("White pixel image", white_pixel_image);

	//black pixels from sample image
	Mat inverted_img;
	invertImage(white_pixel_image, inverted_img);//invert to get non-white pixels
	imshow("Inverted white pixel image", inverted_img);
	//and with the grey connected components image to get the black pixels
	bitwise_and(grey_cc, inverted_img, black_pixel_image);
	imshow("White pixels removed", black_pixel_image);
	//this still has the red pixels in it - AND with the backproj to remove them
	bitwise_and(black_pixel_image, backproj, black_pixel_image);
	imshow("Black pixel image", black_pixel_image);

	//white pixel ground truth
	threshold(grey_gndtruth, white_groundtruth, 200, 255, THRESH_BINARY);
	imshow("white gndtruth image", white_groundtruth);
	//black pixel ground truth
	threshold(grey_gndtruth, black_groundtruth, 20, 255, THRESH_BINARY);//threshold
	invertImage(black_groundtruth, black_groundtruth);//invert the image
	imshow("black gndtruthimage", black_groundtruth);

}

//gets the false positives and negatives, true positives and negatives
void getRedMetrics(Mat ground_truth, Mat back_projection) {

	//get dimensions (images are same size)
	int rows = ground_truth.rows; int cols = ground_truth.cols;
	uchar red, green, blue, intensity;

	//go through each pixel and compare
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//get colour intensity of each pixel
			Vec3b gnd_pixel_color = ground_truth.at<Vec3b>(i, j);
			red = gnd_pixel_color.val[2], blue = gnd_pixel_color.val[0], green = gnd_pixel_color.val[1];
			Scalar backproj_pixel_colour = back_projection.at<uchar>(i, j);
			intensity = backproj_pixel_colour.val[0];

			if ((blue == 0) && (green == 0) && (red == 255)) {//if gndtruth pixel is red
				if (intensity == 0) { //if backproj pixel is detected red (i.e. black)
					TP++;
				}
				else FN++;//false negative if gndtruth pixel=red but backproj pixel=white (non-detected red)
			}
			else {	//we have gndtruth pixel that isn't red
				if (intensity != 0) {
					//true negative if groundpixel!=red and samplepixel=white
					TN++;
				}
				else FP++; //false positive if gndtruth pixel!=red but backproj pixel is black (detected red)
			}
		}
	}

	//print the metrics once calculated
	cout << endl << "This is the red pixel metrics: " << endl;
	printMetrics();
}


//gets the false positives and negatives, true positives and negatives
void getBlackandWhiteMetrics(String colour) {

	//get dimensions (images are all same size)
	int rows = black_groundtruth.rows; int cols = black_groundtruth.cols;
	//assume we only take binary images
	uchar gnd_pixel_color, image_pixel_colour;

	//go through each pixel and compare for the two binary images
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//get colour intensity of each pixel
			if (colour == "black") { // we are doing black pixel comparison
				gnd_pixel_color = black_groundtruth.at<uchar>(i, j);
				image_pixel_colour = black_pixel_image.at<uchar>(i, j);
			}
			else {//we are doing white pixel comparison
				gnd_pixel_color = white_groundtruth.at<uchar>(i, j);
				image_pixel_colour = white_pixel_image.at<uchar>(i, j);
			}
			if (gnd_pixel_color > 0) {//if backproj pixel is black (detected red)
				if (image_pixel_colour > 0) { //gnd truth pixel IS red
					TP++;
				}
				else { //false negative if groundpixel!=red and samplepixel=red 
					FN++;
				}
			}
			else {
				//we have gndtruth pixel !(0,0,255)
				if (image_pixel_colour == 0) {
					//true negative if groundpixel!=red and samplepixel!=black
					TN++;
				}
				else FP++;
			}
		}
	}

	//print the metrics once calculated
	cout << endl << "This is the " << colour << " pixel metrics: " << endl;
	printMetrics();
}

void printMetrics() {

	//calculate metrics
	double precision = ((double)TP) / ((double)(TP + FP));
	double recall = ((double)TP) / ((double)(TP + FN));
	double accuracy = ((double)(TP + TN)) / ((double)(TP + FP + TN + FN));
	double specificity = ((double)TN) / ((double)(FP + TN));
	double f1 = 2.0*precision*recall / (precision + recall);
	//output results to console
	cout << "-----------------------------" << endl;
	cout << "True postitives: " << TP << endl;
	cout << "False postitives: " << FP << endl;
	cout << "True negatives: " << TN << endl;
	cout << "False negatives: " << FN << endl;
	cout << "-----------------------------" << endl;
	cout << "Precision: " << precision * 100 << endl;
	cout << "Recall: " << recall * 100 << endl;
	cout << "Accuracy: " << accuracy * 100 << endl;
	cout << "Specificity: " << specificity * 100 << endl;
	cout << "f1: " << f1 * 100 << endl;
	cout << "-----------------------------" << endl << endl;
}