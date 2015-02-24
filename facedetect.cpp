#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#define NELEMS(x)  (sizeof(x) / sizeof(x[0]));

using namespace std;
using namespace cv;

void analyseFrame(Mat curFrame, Mat prevFrame, Mat &Ix, Mat &Iy, Mat &It, int &mod);
void getRange(Mat matrix, double &range, double &range1);
void normalise(cv::Mat &matrix, double dataHigh, double dataLow);
void LKTracker(Mat video, Mat ix, Mat iy, Mat it, Mat &xMove, Mat &yMove, Mat &tMove, Mat &avMove, double dir[], double centreX, double centreY, int gridsize);

int main( int argc, const char **argv )
{
	cv::VideoCapture cap;
	if (argc > 1)
	{
		cap.open(string(argv[1]));
	}
	else
	{
		cap.open(CV_CAP_ANY);
	}
	if (!cap.isOpened())
	{
		printf("Error: could not load a camera or video.\n");
	}

	int kernelSize = atoi(argv[2]);
	//string number = static_cast<ostringstream*>( &(ostringstream() << count) )->str();
	Mat curFrame;
	cap >> curFrame;
	Mat prevFrame = curFrame.clone();
	namedWindow("video", 1);
	namedWindow("Ix",1);
	namedWindow("Ix",1);
	namedWindow("Iy",1);
	namedWindow("v",1);
	double xDirection[5];
	double yDirection[5];
	cout << " go " << endl;
	int mod = 0;
	for (;;)
	{
		prevFrame = curFrame.clone();
		cap >> curFrame;
		cv::flip(curFrame, curFrame, 1);
		waitKey(10);
		Size s;
		s.height = curFrame.rows;
		s.width = curFrame.cols;
		Mat Ix(s, CV_64F);
		Mat Iy(s, CV_64F);
		Mat It(s, CV_64F);
		Mat xMove(s, CV_64F);
		Mat yMove(s, CV_64F);
		Mat tMove(s, CV_64F);
		Mat avMove(s, CV_64F);
		double dir[2];
		xMove = curFrame.clone();
		yMove = curFrame.clone();
		tMove = curFrame.clone();
		avMove = curFrame.clone();
		double centreX;
		double centreY;
		double Xav = 0;
		double Yav = 0;
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 2;
		int thickness = 3;
		cv::Point textOrg(10, 130);
		cv::Point textOrg2(10, 190);


		analyseFrame(curFrame, prevFrame, Ix, Iy, It, mod);
		int gridsize = 20;
		for (int i = 1; i < (s.width / gridsize); i++) {
			for (int j = 1; j < (s.height / gridsize); j++) {
				centreX = ((i * gridsize));
				centreY = ((j * gridsize) );
				LKTracker(curFrame, Ix, Iy, It, xMove, yMove, tMove, avMove, dir, centreX, centreY , gridsize);
				Xav += dir[0];
				Yav += dir[1];
			}
		}
		//cout << Xav << endl;
		int size = sizeof(xDirection) / sizeof(xDirection[0]);
		//cout << "  results " << endl;
		for (int k = size - 1; k > 0; k--) {
			xDirection[k] = xDirection[k - 1];
		}
		xDirection[0] = Xav;
		double equals = 0;
		for (int l = 0; l < size - 1; l++) {
			equals += xDirection[l];
		}
		equals = equals / size;
		//cout << "equals "<<equals << endl;
		if ((equals > 100) && xDirection[4] > 0) {
			cv::putText(avMove, "Left", textOrg, fontFace, fontScale, Scalar::all(100), thickness, 8);
		} else if (equals < 0 && xDirection[4] < 0) {
			cv::putText(avMove, "Right", textOrg, fontFace, fontScale, Scalar::all(100), thickness, 8);
		}

		//cout << "  results " << endl;
		for (int m = size - 1; m > 0; m--) {
			yDirection[m] = yDirection[m - 1];
		}
		yDirection[0] = Yav;
		double equals2 = 0;
		for (int b = 0; b < size - 1; b++) {
			equals2 += yDirection[b];
		}
		equals2 = equals2 / size;
		//cout << "equals "<<equals << endl;
		if ((equals2 > 100) && yDirection[4] > 0 ) {
			cv::putText(avMove, "Down", textOrg2, fontFace, fontScale, Scalar::all(100), thickness, 8);
		} else if (equals2 < 0 && yDirection[4] < 0 ) {
			cv::putText(avMove, "Up", textOrg2, fontFace, fontScale, Scalar::all(100), thickness, 8);
		}

		imshow("xMove", xMove);
		imshow("yMove", yMove);
		imshow("avMove", avMove);


		double range = 0;
		double range1 = 0;

		//getRange(Ix, range, range1);
		getRange(Iy, range, range1);
		normalise(Ix, range1, range); ;
		normalise(Iy, range1, range);
		normalise(It, range1, range);

		imshow("Ix", Ix);
		imshow("Iy", Iy);
		imshow("It", It);

	}

}

void LKTracker(Mat video, Mat ix, Mat iy, Mat it, Mat &xMove, Mat &yMove, Mat &tMove, Mat &avMove, double dir[], double centreX, double centreY, int gridsize) {
	Mat A = Mat::zeros(2, 2, CV_64F);
	Mat B = Mat::zeros(2, 1, CV_64F);
	for (int i = centreY - (gridsize / 2); i < centreY + (gridsize / 2); i++) {
		for (int j = centreX - (gridsize / 2); j < centreX + (gridsize / 2); j++) {
			double x = ix.at<double>(i, j);
			double y = iy.at<double>(i, j);
			double t = it.at<double>(i, j);

			A.at<double>(0, 0) = A.at<double>(0, 0) + pow(x, 2);
			A.at<double>(1, 1) = A.at<double>(1, 1) + pow(y, 2);
			A.at<double>(0, 1) = A.at<double>(0, 1) + x * y;
			A.at<double>(1, 0) = A.at<double>(0, 1) + x * y;

			B.at<double>(0, 0) = B.at<double>(0, 0) + t * -x;
			B.at<double>(0, 1) = B.at<double>(0, 1) + t * -y;
		}
	}

	Mat aInv = A.inv();
	Mat v = aInv * B;
	double yInc = v.at<double>(0, 0) * 2;
	double xInc = v.at<double>(0, 1) * 2;
	double x2 = centreX - xInc;
	double y2 = centreY + yInc;
	double xAv = (centreX + x2) / 2;
	double yAv = (centreY + y2) / 2;
	double lowerThresh = 30;
	double upperThresh = 80;
	//cout << xInc << endl;
	//cout << yInc << endl;
	if ((abs(xInc) > lowerThresh && abs(xInc) < upperThresh) | (abs(yInc) > lowerThresh && abs(yInc) < upperThresh)) {
		dir[0] = xInc;
		dir[1] = yInc;



		line(avMove, Point(centreX, centreY), Point(xAv, yAv), Scalar(0, 255, 255), 2);
		circle(avMove, Point(xAv, yAv ), 4, Scalar(255, 0, 0));
		line(yMove, Point(centreX, centreY), Point(centreX, y2), Scalar(0, 255, 255), 2);
		circle(yMove, Point(centreX, y2), 4, Scalar(255, 0, 0));
		line(xMove, Point(centreX, centreY), Point(x2, centreY), Scalar(0, 255, 255), 2);
		circle(xMove, Point(x2, centreY), 4, Scalar(255, 0, 0));
	}
	else {
		dir[0] = 0;
		dir[1] = 0;
	}





}

void getRange(Mat matrix, double &range, double &range1)
{
	range = 100000000;
	range1 = -10000000;
	double pixelVal = 0;
	// now we can do the convoltion
	for ( int i = 0; i < matrix.rows; i++ )
	{
		for ( int j = 0; j < matrix.cols; j++ )
		{

			// find the range
			pixelVal = matrix.at<double>(i, j);
			if (pixelVal > range1)
			{
				range1 = pixelVal;
			}
			if (pixelVal < range)
			{
				range = pixelVal;
			}
		}
	}
}


void normalise(cv::Mat &matrix, double dataHigh, double dataLow) {
	for ( int i = 0; i < matrix.rows; i++ )
	{
		for ( int j = 0; j < matrix.cols; j++ )
		{
			matrix.at<double>(i, j) = (((matrix.at<double>(i, j) - dataLow) / (dataHigh - dataLow)));//* (newHigh - newLow)) + newLow;

		}
	}
}



void analyseFrame(Mat curFrame, Mat prevFrame, Mat &Ix, Mat &Iy, Mat &It, int &mod)
{
	cvtColor( curFrame, curFrame, COLOR_BGR2GRAY );
	cvtColor( prevFrame, prevFrame, COLOR_BGR2GRAY );
	//Mat kernel(n, n);

	int totX = 0;
	int totY = 0;
	int totT = 0;




	for (int i = 1; i < curFrame.rows - 1; i++)
	{
		for (int j = 1; j < curFrame.cols - 1; j++)
		{

			totX = curFrame.at<uchar>(i, j - 1) - curFrame.at<uchar>(i - 1, j - 1);
			totX += curFrame.at<uchar>(i, j) - curFrame.at<uchar>(i - 1, j);
			totX += prevFrame.at<uchar>(i, j - 1) - prevFrame.at<uchar>(i - 1, j - 1);
			totX += prevFrame.at<uchar>(i, j) - prevFrame.at<uchar>(i - 1, j);
			Ix.at<double>(i, j) = totX / 4;

			totY = curFrame.at<uchar>(i - 1, j - 1) - curFrame.at<uchar>(i - 1, j);
			totY += curFrame.at<uchar>(i, j - 1) - curFrame.at<uchar>(i, j);
			totY += prevFrame.at<uchar>(i - 1, j - 1) - prevFrame.at<uchar>(i - 1, j);
			totY += prevFrame.at<uchar>(i, j - 1) - prevFrame.at<uchar>(i, j);
			Iy.at<double>(i, j) = totY / 4;


			totT = curFrame.at<uchar>(i, j) - prevFrame.at<uchar>(i, j);
			totT += curFrame.at<uchar>(i + 1, j) - prevFrame.at<uchar>(i + 1, j);
			totT += curFrame.at<uchar>(i, j + 1) - prevFrame.at<uchar>(i, j + 1);
			totT += curFrame.at<uchar>(i + 1, j + 1) - prevFrame.at<uchar>(i + 1, j + 1);
			It.at<double>(i, j) = totT / 4;
		}
	}

}