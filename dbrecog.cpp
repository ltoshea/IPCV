#include "opencv.hpp"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

struct Squares
{
	int x, y, width, height;
	double centreX, centreY, radius;
	bool valid;
};

/** Function Headers */
void detectAndSave( Mat frame, int count );
int cannyLines(Mat src, int p);
int hough(Mat src, int p, double &radius);
int cornerHarris( Mat src, int p );
void resizeMat(Mat &src, Size s);
int histCompare(Mat src_test1, int i);
int checkRange(Mat input, int count);
int averageFilter(Mat input);

/** Global variables */
String logo_cascade_name = "training/cascade.xml";
int FrameCount = 0;
CascadeClassifier logo_cascade;
string window_name = "Capture - Face detection";
const int IMGNUM = 21;

/** @function main */
int main( int argc, const char **argv )
{
	CvCapture *capture;
	Mat images[IMGNUM]; // Array of 10 images
	std::stringstream sstm;

	//Load the cascades
	if ( !logo_cascade.load( logo_cascade_name ) ) {
		printf("--(!)Error loading\n"); return -1;
	}
	for (int i = 1; i < IMGNUM; i++)
	{
		//int i = 1;
		sstm << "training/dart" << i << ".jpg";
		images[i] = imread(sstm.str(), CV_LOAD_IMAGE_COLOR);
		sstm.str(std::string()); // clear string stream
		detectAndSave(images[i], i);
	}
	//   cout << "Frames Detected: " << FrameCount << endl;
	waitKey();
	return 0;
}

/** @function detectAndSave */
void detectAndSave( Mat frame, int count )
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	//-- Detect faces
	logo_cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500) );

	int votes;
	Size s;
	s.height = 300;
	s.width = 300;
	string number = static_cast<ostringstream *>( &(ostringstream() << count) )->str();
	Squares potentials[faces.size()];
	int avCount, histCount, cannyCount, houghCount;
	for ( int i = 0; i < faces.size(); i++ )
	{
		potentials[i].x = faces[i].x;
		potentials[i].y = faces[i].y;
		potentials[i].width = faces[i].width;
		potentials[i].height = faces[i].height;
		potentials[i].valid = true;
		potentials[i].centreX = (potentials[i].x + potentials[i].width) / 2;
		potentials[i].centreY = (potentials[i].y + potentials[i].height) / 2;
		potentials[i].radius = 0;
		votes = 0;
		Mat detectedTargetGray (frame_gray, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height) );
		Mat detectedTarget (frame, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height) );
		resizeMat(detectedTargetGray, s); // resize potential dartboards to the same size so that the features work the same on each one
		resizeMat(detectedTarget, s);

		votes += averageFilter(detectedTargetGray);
		votes += checkRange(detectedTargetGray, i);
		votes += cannyLines(detectedTargetGray, i);
		double rad = potentials[i].radius;
		votes += hough(detectedTarget, i, rad);
		if (votes < 4) {
			potentials[i].valid = false;
		}
	}
	string face = static_cast<ostringstream *>( &(ostringstream() << faces.size()) )->str();
	for ( int i = 0; i < faces.size(); i++ )
	{
		if (potentials[i].valid == false) {}
		else {
			//if there is a square within a square, remove the square that has the smallest circle as detected by the hough circle function.
			for (int j = 0; j < faces.size(); j++) {
				if ((j == i) || (potentials[j].valid == false)) {}
				else {
					if ((potentials[i].x <= potentials[j].x) && ((potentials[i].x + potentials[i].width) >=  (potentials[j].x + potentials[j].width))
					        && (potentials[i].y <= potentials[j].y) && ((potentials[i].y + potentials[i].height) >=  (potentials[j].y + potentials[j].height))) {
						if (potentials[i].radius > potentials[j].radius)
						{
							potentials[j].valid = false;
						} else {
							potentials[i].valid = false;
						}
					}
				}
			}
		}
	}
	//Print out valid dartboards
	for (int i = 0; i < faces.size(); i++)
	{
		if (potentials[i].valid == true) {
			rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
		}
	}
	//-- Show image
	imshow( "output" + number, frame );
}

//check that the average pixel value is in the range of a dartboard
int averageFilter(Mat input)
{
	int total = 0;
	double average;
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			total += input.at<uchar>(i, j);
		}
	}
	average = total / (input.rows * input.cols);
	if (average < 60 || average > 240) {
		return 0;
	}
	else {
		return 1;
	}

}

//check that the range of pixel values in an image is consistent with that of a
//dartboard
int checkRange(Mat input, int count)
{
	int MaxCount = 0;
	int MinCount = 0;
	int MaxThreshold = 230;
	int MinThreshold  = 100;
	int MaxRange = 0;
	int MinRange = 0;


	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			if (input.at<uchar>(i, j) >= MaxThreshold)
			{
				MaxCount++;

			}
			if (input.at<uchar>(i, j) <= MinThreshold)
			{
				MinCount++;
			}
		}
	}
	if (MaxCount > 1691 || MinCount > 30000)
	{
		return 1;

	}
	else
	{
		return 0;
	}
}



void resizeMat(Mat &src, Size s)
{
	resize(src, src, s);
}



int cornerHarris( Mat src, int p )
{
	int thresh = 200;
	int max_thresh = 1000;
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros( src.size(), CV_32FC1 );

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris( src, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

	/// Normalizing
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	/// Drawing a circle around corners
	int count = 0;
	for ( int j = 0; j < dst_norm.rows ; j++ )
	{	for ( int i = 0; i < dst_norm.cols; i++ )
		{
			if ( (int) dst_norm.at<float>(j, i) > thresh )
			{
				count ++;
				circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
			}
		}
	}
	/// Showing the result
	string number = static_cast<ostringstream *>( &(ostringstream() << p) )->str();
	imshow( "harris " + number, dst_norm_scaled );
	if (count > 10)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

//Does a canny edge detect to detect edges then a Hough Line transform to draw on
//any straight lines. The function then checks to see if there are more than 4 line
//interceptions in the middle 10x10 section of the
int cannyLines(Mat src, int p) {
	int interceptFlag = 0;
	string number = static_cast<ostringstream *>( &(ostringstream() << p) )->str();
	Mat dst, cdst;
	Size s = src.size();
	Mat linesIntercept1(s.height, s.width, CV_AA, Scalar(0));
	Mat linesIntercept2(s.height, s.width, CV_AA, Scalar(0));
	Canny(src, dst, 20, 100, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 2, CV_PI / 180, 50, 50, 50);
	int centerBuffer = 10;
	for ( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 0.5, CV_AA); //draw line on image
		line( linesIntercept1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 0.005, CV_AA); // draw line on temporary image
		Mat im_gray;
		im_gray = Scalar(0);
		cvtColor(linesIntercept1, im_gray, CV_RGB2GRAY);
		//   imshow("grey" + number, cdst);
		// cout << "lines = "<< endl << " "  << im_gray << endl << endl;
		for (int j = 0; j < s.width; j++)
		{
			for (int k = 0; k < s.height; k++)
			{
				if (im_gray.at<unsigned char>(j, k) != 0)
				{
					linesIntercept2.at<unsigned char>(j, k) += 5;
					if (linesIntercept2.at<unsigned char>(j, k) > 20 && (j > ((s.width / 2) - centerBuffer)) && (j < ((s.width / 2) + centerBuffer)) && (k > ((s.height / 2) - centerBuffer)) && (k < ((s.height / 2) + centerBuffer))) //change to 150 to get less false positives but 1 less false negative
					{
						interceptFlag = 1;
					}
				}
			}
		}
		//imshow("int" + number, linesIntercept2);
		// imshow("int" + number, linesIntercept2);
		linesIntercept1 = Scalar(0);
	}
	// imshow("source " + number, cdst);
	//imshow("detected lines" + number, linesIntercept1);
	return interceptFlag;
}


//This function checks for any circles with an aptly sized radius to be a dartboard. If it detects
//atleast one cirlce of the required size it returns 1
int hough(Mat src, int p, double &radius)
{
	Mat src_gray;
	string number = static_cast<ostringstream *>( &(ostringstream() << p) )->str();
	/// Convert it to gray
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	// HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/11, 50, 30, 0, 0 );
	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 40, 40, 100, 0 );

	// cout << "circles = " << circles.size() << endl;
	/// Draw the circles detected
	/*  for( size_t i = 0; i < circles.size(); i++ )

	{
	 //  cout << "lines = " << endl;
	 Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	 int radius = cvRound(circles[i][2]);
	 // circle center
	 circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
	 // circle outline
	 circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}

	imshow( "Hough" + number, src );*/
	radius = 0;
	if (circles.size() > 0) {
		for (int i = 0; i < circles.size(); i++)
		{
			if (cvRound(circles[i][2] < radius)) {
				radius = circles[i][2];
			}
		}
		return 1;
	}
	else return 0;

	/// Show your results

}


int histCompare(Mat src_test, int count)
{
	//imshow("Before Blur",src_test);
	GaussianBlur( src_test, src_test, Size(0, 0), 3, 3, BORDER_DEFAULT );
	//imshow("After Blur",src_test);

	Mat src_base1 = imread("training/dart.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	equalizeHist( src_test, src_test );
	equalizeHist( src_base1, src_base1 );

	/// Establish the number of bins
	int argBinSize = 10;//strtol(argv[4],NULL,10);
	int histSize = argBinSize;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 255 } ;
	const float *histRange = { range };
	bool uniform = true; bool accumulate = false;

	/// Compute the histograms:
	Mat base1_hist, test_hist;
	calcHist( &src_base1, 1, 0, Mat(), base1_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &src_test, 1, 0, Mat(), test_hist, 1, &histSize, &histRange, uniform, accumulate );

	double base1_base1 = compareHist( base1_hist, base1_hist, 3 );
	double testResult = compareHist( base1_hist, test_hist, 3 );
	//printf( "In image %d (method 1)- Perfect, TestImg (dart.bmp): %f,%f \n", count, base1_base1, testResult );
	//cout << endl;


	if (testResult > 0.272 && testResult < 0.471)
	{
		//FrameCount++;
		return 1;
	}
	else
	{
		return 0;
	}


}
