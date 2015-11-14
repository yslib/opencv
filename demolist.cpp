//
//  demolist.cpp
//  OpenCVDemo
//
//  Created by 杨朔柳 on 15/11/10.
//  Copyright © 2015年 杨朔柳. All rights reserved.
//
//#pragma warning(disable:4996)

#include "demolist.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
//#include <string.h>


//OpenCV for C++
#include <opencv2/opencv.hpp>		//for all
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect.hpp>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>



//#define __cplusplus			

#include "getopt.h"	
		

//
using namespace std;
//
using namespace cv;


//************************************
// Method:    Demo_Smooth
// FullName:  Demo_Smooth
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************

int Demo_Smooth(int argc,char ** argv)
{
    //
    //parse the cmd parameters
    //
    int ch;
    opterr=0;
	string path;
    while((ch=getopt(argc, argv, "d:")) != -1)
    {
        switch (ch) {
            case 'd':
				path = optarg;        //the image or video path
                break;  
            default:
                break;
        }
    }
    //parse ended
	//printf("%s\n", path);
	Mat srcImage = imread(path);
	if (srcImage.empty())
	{
		printf("object read faild\n");
		return -1;
	}
    return 0;
}



///////////////////////////////////////////////////////////////////
//************************************
// Method:    Demo_FaceDetect
// FullName:  Demo_FaceDetect
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************



Scalar color;
Mat smallImgROI;
vector<Rect> faces;
vector<Rect> nestedObjects;
Point center;
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{


	int i = 0;
	double t = 0;
	
	const static Scalar colors[] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cvtColor(img, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double)cvGetTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE
		,
		Size(30, 30));
	//if (tryflip)
	//{
	//	flip(smallImg, smallImg, 1);
	//	cascade.detectMultiScale(smallImg, faces2,
	//		1.1, 2, 0
	//		//|CASCADE_FIND_BIGGEST_OBJECT
	//		//|CASCADE_DO_ROUGH_SEARCH
	//		| CASCADE_SCALE_IMAGE
	//		,
	//		Size(30, 30));
	//	for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
	//	{
	//		faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
	//	}
	//}
	t = (double)cvGetTickCount() - t;
	printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));

	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); ++r, i++)
	{
		color = CV_RGB(0, 0, 255);
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		circle(img, center, radius, color, 3, 8, 0);

		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(*r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE
			,
			Size(30, 30));
		for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); ++nr)
		{
			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
			radius = cvRound((nr->width + nr->height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}

	}
	cv::imshow("result", img);
}
int Demo_FaceDetect(int argc, char ** argv)
{
	
	//parse the cmd parameters
	
	int ch;
	opterr = 0;
	string path="C:\\haar\\demo.avi";
	const string nestedCascadeName = "C:\\haar\\haarcascade_eye_tree_eyeglasses.xml";
	const string cascadeName = "C:\\haar\\haarcascade_frontalface_alt.xml";
	int Mode = 0;
	while ((ch = getopt(argc, argv, "cv:")) != -1)
	{
		switch (ch) {
		case 'v':
			Mode = 0;
			path = optarg;        //the image or video path
			break;
		case 'c':				//capture from the camara
			Mode = 1;
			break;
		default:
			break;
		}
	}

	CvCapture * cap=NULL;
	CascadeClassifier cascade, nestedCascade;

	if (!nestedCascade.load(nestedCascadeName))
	{
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		return -1;
	}

	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
	if (Mode == 0)
	{
		cap = cvCaptureFromAVI(path.c_str());
	}
	else if (Mode == 1)
	{
		cap = cvCaptureFromCAM(0);
	}
	if (cap == NULL)
	{
		cerr << "Capture from deveice or file failed.\n";
		return -1;
	}

	cout << "In capture...\n";
	IplImage * image = cvQueryFrame(cap);
	Mat frame;
	for (;;)
	{
		image = cvQueryFrame(cap);
		frame = cv::cvarrToMat(image);
		if (frame.empty())
		{
			break;
		}
		detectAndDraw(frame, cascade, nestedCascade, 1, false);
		if (waitKey(10) >= 0)
			goto _cleanup_;
	}
	
_cleanup_:
	cout << "Capture end\n";
	cvReleaseCapture(&cap);
	waitKey(0);

}

//************************************
// Method:    ImageResize
// FullName:  ImageResize
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************
int ImageResize(int argc, char ** argv)
{

	//parse the cmd parameters

	int ch;
	opterr = 0;
	string pathIn = "";
	string pathOut = "";
	int width=200, height=200;
	if (argc == 0)
	{
		cerr << "需要参数\n";
		return -1;
	}
	while ((ch = getopt(argc, argv, "i:n:w:h:o:")) != -1)
	{
		switch (ch) {
		case 'i':
			
			break;
		case 'n':
			pathIn = optarg;
			break;
		case 'w':
			width = atoi(optarg);			
			break;
		case 'h':
			height = atoi(optarg);			
			break;
		case 'o':
			pathOut = optarg;
			break;
		default:
			break;
		}
	}
	Mat srcImage = imread(pathIn);
	cout << pathIn << endl;
	if (srcImage.empty() == true)
	{
		cerr << "image loaded failed\n";
		return -1;
	}
	Mat outImage;

	resize(srcImage, outImage, Size(width, height),0,0,1 );


	imshow("result", outImage);
	//waitKey(0);

	vector<int> comparessionParams;
	comparessionParams.push_back(CV_IMWRITE_JPEG_QUALITY);
	comparessionParams.push_back(95);

	imwrite(pathOut, outImage,comparessionParams);
	return 0;
}