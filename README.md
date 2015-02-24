# IPCV
Image Processing and Computer Vision Techniques Using C++, OPENCV and an off the shelf webcam.
This is experimental code and needs to be better commented and tidied up. 

coins.cpp - Will recognise and highlight coins in a set of images <br>
dbrecog.cpp - Will recognise and highlight dartboards in a set of images <br>
facedetect.cpp - Will recognise and highlight faces when presented with a webcam stream<br>

From memory opencv needed to produce a file for the detection of dartboards and faces. 
You can use --haar-features or -traincascade for this, the -traincascade is newer, faster and produced better results than using just --haar-features

