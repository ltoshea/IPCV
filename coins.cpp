#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <math.h>

using namespace cv;
using namespace std;

struct brightpix {
  int x, y;
  double pixval, radius;
};

void sort_pixels(brightpix input[], int n);
void sobelTrans(cv::Mat &input, cv::Mat &Output);
void applyKernel(cv::Mat &input, cv::Mat &Output, cv::Mat kernel, double range[]);
void findGradient(cv::Mat matrixX, cv::Mat matrixY, cv::Mat &Magnitude, cv::Mat &Direction);
void normalise(cv::Mat &matrix, double dataHigh, double dataLow, double newHigh, double newLow);
void convertToUchar(cv:: Mat matrix, cv::Mat &Output);
void findDirection(cv::Mat matrixX, cv::Mat matrixY, cv::Mat &Output);
void getRange(cv::Mat matrix, double range[]) ;
void SetThreshold(cv::Mat input, cv::Mat output, int threshold);
void HoughTransform(cv::Mat inputMag, cv::Mat inputDir, double threshold, cv::Mat output);
void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput);
void FindCentres(cv::Mat input, cv::Mat inputNorm, cv::Mat output);
void finalCentres(brightpix input[], int k);


int main( int argc, char **argv )
{
  cv::Mat coins1 = cv::imread("images/coins1.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat coins2 = cv::imread("images/coins2.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat coins3 = cv::imread("images/coins3.png", CV_LOAD_IMAGE_GRAYSCALE);

  //Gaussian Blurr the image first as it improves the direction and magnitude images
  cv::Mat coins1Blurred;
  GaussianBlur(coins1, 3, coins1Blurred);
  cv::Mat sobelOutput;
  sobelTrans(coins1Blurred, sobelOutput);

  cv::namedWindow("Coins 1", CV_WINDOW_AUTOSIZE);
  //cv::namedWindow("Coins 2", CV_WINDOW_AUTOSIZE);
  //cv::namedWindow("Coins 3", CV_WINDOW_AUTOSIZE);

  cv::imshow("Coins 1", coins1);
  // cv::imshow("Coins 2", coins2);
  // cv::imshow("Coins 3", coins3);

  cv::waitKey();

  return 0;
}


void FindCentres(cv::Mat input, cv::Mat inputNorm, cv::Mat radi,
                 cv::Mat output)
// Attempts to find the centre of the coins
{
  const int threshold = 150;
  const int size = 1000;
  brightpix temppospix[size];
  int k = 0;


  for (int i = 0; i < input.rows; i++ )
  {
    for ( int j = 0; j < input.cols; j++ )
    {
      if (inputNorm.at<double>(i, j) > threshold)
      {
        temppospix[k].x = i;
        temppospix[k].y = j;
        temppospix[k].pixval = input.at<double>(i, j);
        temppospix[k].radius = radi.at<double>(i, j) / temppospix[k].pixval;
        //cout << "pospix: "<< k<<" x: "<<temppospix[k].x<<" y: "<<temppospix[k].y<<" val: "<<temppospix[k].pixval << " r: " << temppospix[k].radius <<endl;
        k++;

      }
    }
  }
  brightpix pospix[k];
  for (int b = 0; b < k; b++)
  {
    pospix[b] = temppospix[b];
  }

  sort_pixels(pospix, k);
  finalCentres(pospix, k);
}

void finalCentres(brightpix input[], int k)
{
  brightpix finallist[k];
  for (int w = 0; w < k; w++) {
    finallist[w].x = 6666;
    //cout<<"finallist:" <<  finallist[k].x << endl;
  }

  bool end;
  for (int i = 0; i < k; i++)
  {
    end = false;
    for (int j = 0; j < k; j++)
    {
      if (!end)
      {

        if (finallist[j].x = 6666)
        {

          finallist[j] = input[i];
          end = true;

        }

        else
        {
          double magnitude = sqrt(pow(((input[i].x) - (finallist[j].x)), 2) + pow(((input[i].y) - (finallist[j].y)), 2));
          if (magnitude < input[i].radius)
          {
            end = true;

          }
        }
      }
    }
  }
  for (int p = 0; p < 11; p++)
  {
    cout << endl;
    std::cout << " x:" << finallist[p].x << " y:" << finallist[p].y << " pixval:" << finallist[p].pixval << " r:" << finallist[p].radius << endl;
  }
}

void sort_pixels(brightpix input[], int n)
{
  brightpix temp; // Local variable used to swap records
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      // If s[i].student_number is greater than s[i+1].student_number, swap the records
      if (input[j].pixval < input[j + 1].pixval)
      {
        temp = input[j];
        input[j] = input[j + 1];
        input[j + 1] = temp;
      }
    }
  }
  for (int k = 0; k < n; k++)
  {
    cout << "pixels: " << input[k].radius << endl;
  }
}

int countWhite(cv::Mat input)
{
  int count = 0;
  for (int i = 0; i < input.rows; i++)
  {
    for (int j = 0; j < input.cols; j++)
    {
      if (input.at<uchar>(i, j) == 255)
      {
        count++;
      }
    }
  }
  return count;

}

void HoughTransform(cv::Mat inputMag, cv::Mat inputDir, double threshold, cv::Mat output)//, double radius)
{
  cv::Size s = inputMag.size();
  //int size = inputMag.size()+70;
  cv::Mat houghSpace(s.height, s.width, CV_64F); // +70 for padding (THIS COULD BE IMPLEMENTED MUCH BETTER)
  cv::Mat houghSpaceNorm(s.height, s.width, CV_64F); // +70 for padding (THIS COULD BE IMPLEMENTED MUCH BETTER)

  cv::Mat radi(s.height, s.width, CV_64F); // +70 for padding (THIS COULD BE IMPLEMENTED MUCH BETTER)

  cv:: Mat houghOutput(s.height, s.width, inputMag.type());

  // Data structure needs some improvement
  double temp;
  double xShift;
  double yShift;
  double xNegShift;
  double yNegShift;
  double pi = 3.1415;
  for ( int i = 0; i < inputMag.rows; i++ )
  {
    for ( int j = 0; j < inputMag.cols; j++ )
    {
      if (inputMag.at<uchar>(i, j) == 255 )
      {
        for (int m = 20; m < 50; m += 1)
        {
          for (double k = -0.1; k < 0.1; k += 0.05)
          {
            xShift = m * cos(inputDir.at<double>(i, j) + k + pi / 2);
            yShift = m * sin(inputDir.at<double>(i, j) + k + pi / 2);
            xNegShift = (m * cos(inputDir.at<double>(i, j) + k + pi / 2));
            yNegShift = (m * sin(inputDir.at<double>(i, j) + k + pi / 2));

            if (((i + xShift > 0 && i + xShift < inputMag.rows) && (j + yShift > 0 && j + yShift < inputMag.cols)) && ((i + xNegShift > 0 && i + xNegShift < inputMag.rows) && (j + yNegShift > 0 && j + yNegShift < inputMag.cols))) {
              houghSpace.at<double>(i + xShift, j + yShift) = houghSpace.at<double>(i + xShift, j + yShift) + 1;
              houghSpace.at<double>(i + xNegShift, j + yNegShift) = houghSpace.at<double>(i + xNegShift, j + yNegShift) + 1;

              houghSpaceNorm.at<double>(i + xShift, j + yShift) = houghSpace.at<double>(i + xShift, j + yShift) + 1;
              houghSpaceNorm.at<double>(i + xNegShift, j + yNegShift) = houghSpace.at<double>(i + xNegShift, j + yNegShift) + 1;

              radi.at<double>(i + xShift, j + yShift) = radi.at<double>(i + xShift, j + yShift) + m;
              radi.at<double>(i + xNegShift, j + yNegShift) = radi.at<double>(i + xNegShift, j + yNegShift) + m;
            }
            //  houghSpace.at<double>(i-xShift, j-yShift) =houghSpace.at<double>(i-xShift, j-yShift) + 1;
          }
        }

      }

    }
  }
  cv: Mat houghTemp(houghSpace.size(), houghSpace.type());
  double houghRange[2];
  getRange(houghSpaceNorm, houghRange);
  normalise(houghSpaceNorm, houghRange[1], houghRange[0], 255, 0);

  FindCentres(houghSpace, houghSpaceNorm, radi, houghTemp);

  convertToUchar(houghSpace, houghOutput);
  SetThreshold(houghOutput, houghOutput, 150);

  // int count = 0;
  // for ( int i = 0; i < houghOutput.rows; i++ )
  //   {
  //   for( int j = 0; j < houghOutput.cols; j++ )
  //   {
  //     if(houghOutput.at<uchar>(i,j) > 200)
  //     {
  //       count++;
  //       cout << "X = "<< endl << " "  << i << endl << endl;
  //       cout << "Y = "<< endl << " "  << j << endl << endl;
  //     }
  //   }
  // }
  // cout <<"Number of coins = "<< endl << " "  << count << endl << endl;
  cv::namedWindow("Hough", CV_WINDOW_AUTOSIZE);
  cv::imshow("Hough", houghOutput);



}


//All the transforms and normalising stuff is spot on now, no need to touch it up
void sobelTrans (cv::Mat &input, cv::Mat &Output)
{
  Mat horizontal = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  Mat vertical = (Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);


  cv:: Mat tempHoriImage(input.size(), CV_64F);
  cv:: Mat tempVertImage(input.size(), CV_64F);
  cv:: Mat HoriImage(input.size(), input.type());
  cv:: Mat VertImage(input.size(), input.type());
  cv:: Mat Magnitude(input.size(), input.type());
  cv:: Mat Direction(input.size(), input.type());
  cv:: Mat tempMagnitude(input.size(), CV_64F);
  cv:: Mat tempDirection(input.size(), CV_64F);
  cv:: Mat MagThreshold(input.size(), input.type());
  double horizontalRange[2];
  double verticalRange[2];
  double magRange[2];
  double dirRange[2];

  // we need to create a padded version of the input
  // or there will be border effects
  applyKernel(input, tempHoriImage, horizontal, horizontalRange);
  //normalise(tempHoriImage, horizontalRange[0], horizontalRange[1], 255, 0);
  //convertToUchar(tempHoriImage, HoriImage);
  applyKernel(input, tempVertImage, vertical, verticalRange);
  //normalise(tempVertImage, verticalRange[0], verticalRange[1]);
  //convertToUchar(tempVertImage, VertImage);

  findGradient(tempHoriImage, tempVertImage, tempMagnitude, tempDirection);

  getRange(tempMagnitude, magRange);
  normalise(tempMagnitude, magRange[1], magRange[0], 255, 0);
  convertToUchar(tempMagnitude, Magnitude);
  SetThreshold(Magnitude, MagThreshold, 20);

  //getRange(tempDirection, dirRange);
  //normalise(tempDirection, dirRange[1], dirRange[0], 255, 0);
  //convertToUchar(tempDirection, Direction);

  normalise(tempHoriImage, horizontalRange[0], horizontalRange[1], 255, 0);
  convertToUchar(tempHoriImage, HoriImage);
  normalise(tempVertImage, horizontalRange[0], verticalRange[1], 255, 0);
  convertToUchar(tempVertImage, VertImage);

  //   convertToUchar(tempImage1, Magnitude);
  //  convertToUchar(tempImage2, Direction);
  // cv::namedWindow("Horizontal" , CV_WINDOW_AUTOSIZE);
  // cv::namedWindow("Vertical", CV_WINDOW_AUTOSIZE);
  // cv::namedWindow("Magnitude", CV_WINDOW_AUTOSIZE);
  // cv::namedWindow("Direction", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("MagThreshold", CV_WINDOW_AUTOSIZE);

  // cv::imshow("Horizontal", HoriImage);
  // cv::imshow("Vertical", VertImage);
  // cv::imshow("Magnitude", Magnitude);
  // cv::imshow("Direction", Direction);
  cv::imshow("MagThreshold", MagThreshold);
  double threshold = 0;
  cv::Mat output1;
  HoughTransform(MagThreshold, tempDirection, threshold, output1);

}
// Set pixels as white or black depending on threshold
void SetThreshold(cv::Mat input, cv::Mat output, int threshold)
{
  for ( int i = 0; i < input.rows; i++ )
  {
    for ( int j = 0; j < input.cols; j++ )
    {
      if (input.at<uchar>(i, j) < threshold)
      {
        output.at<uchar>(i, j) = 0;
      }
      else
      {
        output.at<uchar>(i, j) = 255;
      }
    }
  }

}


void getRange(cv::Mat matrix, double range[])
{
  range[0] = 0;
  range[1] = 0;
  double pixelVal;
  // now we can do the convoltion
  for ( int i = 0; i < matrix.rows; i++ )
  {
    for ( int j = 0; j < matrix.cols; j++ )
    {
      // find the range
      pixelVal = matrix.at<double>(i, j);
      if (pixelVal > range[1]) {
        range[1] = pixelVal;
      }
      if (pixelVal < range[0]) {
        range[0] = pixelVal;
      }
    }
  }
}


void findGradient(cv::Mat matrixX, cv::Mat matrixY, cv::Mat &Magnitude, cv::Mat &Direction)
{
  double tempX;
  double tempY;
  sqrt( matrixX.mul(matrixX) + matrixY.mul(matrixY), Magnitude);
  for ( int i = 0; i < Magnitude.rows; i++ )
  {
    for ( int j = 0; j < Magnitude.cols; j++ )
    {
      tempX = matrixX.at<double>(i, j); // - matrixX.at<double>(i-1,j);
      tempY = matrixY.at<double>(i, j); // - matrixY.at<double>(i,j-1);
      if (tempX == 0) {
        Direction.at<double>(i, j) = 0;
      }
      else {
        Direction.at<double>(i, j) = atan(tempY / tempX);
      }
      //cout << "atan = "<< endl << " "  << atan2(tempY,tempX) << endl << endl;
      //cout << "atan2 = "<< endl << " "  << atan(tempY/tempX) << endl << endl;
    }
  }
}

void normalise(cv::Mat &matrix, double dataHigh, double dataLow, double newHigh, double newLow)
{
  for ( int i = 0; i < matrix.rows; i++ )
  {
    for ( int j = 0; j < matrix.cols; j++ )
    {
      matrix.at<double>(i, j) = (((matrix.at<double>(i, j) - dataLow) / (dataHigh - dataLow)) * (newHigh - newLow)) + newLow;
      /*double scalar = (1/dataHigh)*255;
      matrix.at<double>(i,j) = matrix.at<double>(i,j)*scalar;*/
    }
  }
}

void convertToUchar(cv:: Mat matrix, cv::Mat &Output)
{
  for ( int i = 0; i < matrix.rows; i++ )
  {
    for ( int j = 0; j < matrix.cols; j++ )
    {
      Output.at<uchar>(i, j) = (uchar) matrix.at<double>(i, j);
    }
  }
}

void applyKernel(cv::Mat &input, cv::Mat &Output, cv::Mat kernel, double range[])
// Apply our image Kernel to the image
{

  // intialise the output using the input
  Output.create(input.size(), CV_64F);

  //CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
  //TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

  // we need to create a padded version of the input
  // or there will be border effects
  int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
  int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

  cv::Mat paddedInput;
  cv::copyMakeBorder( input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE );

  // now we can do the convoltion
  for ( int i = 0; i < input.rows; i++ )
  {
    for ( int j = 0; j < input.cols; j++ )
    {
      double sum = 0.0;
      for ( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
      {
        for ( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
        {
          // find the correct indices we are using
          int imagex = i + 1 + m;
          int imagey = j + 1 + n;
          int kernelx = m + kernelRadiusX;
          int kernely = n + kernelRadiusY;

          // get the values from the padded image and the kernel
          int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
          double kernalval = kernel.at<double>( kernelx, kernely );

          // do the multiplication
          sum += imageval * kernalval;
        }
      }
      // find the range
      Output.at<double>(i, j) = sum;
      if (sum > range[0]) {
        range[0] = sum;
      }
      if (sum < range[1]) {
        range[1] = sum;
      }
    }
  }
}


void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
  // intialise the output using the input
  blurredOutput.create(input.size(), input.type());

  // create the Gaussian kernel in 1D
  cv::Mat kX = cv::getGaussianKernel(size, -1);
  cv::Mat kY = cv::getGaussianKernel(size, -1);

  // make it 2D multiply one by the transpose of the other
  cv::Mat kernel = kX * kY.t();

  //CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
  //TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

  // we need to create a padded version of the input
  // or there will be border effects
  int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
  int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

  cv::Mat paddedInput;
  cv::copyMakeBorder( input, paddedInput,
                      kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                      cv::BORDER_REPLICATE );

  // now we can do the convoltion
  for ( int i = 0; i < input.rows; i++ )
  {
    for ( int j = 0; j < input.cols; j++ )
    {
      double sum = 0.0;
      for ( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
      {
        for ( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
        {
          // find the correct indices we are using
          int imagex = i + 1 + m;
          int imagey = j + 1 + n;
          int kernelx = m + kernelRadiusX;
          int kernely = n + kernelRadiusY;

          // get the values from the padded image and the kernel
          int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
          double kernalval = kernel.at<double>( kernelx, kernely );

          // do the multiplication
          sum += imageval * kernalval;
        }
      }
      // set the output value as the sum of the convolution
      blurredOutput.at<uchar>(i, j) = (uchar) sum;
    }
  }
}