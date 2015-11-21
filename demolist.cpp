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

#include "diropr.h"
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
	return 0;
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

//
//分离字符串
//
void MySplit(const string & str, vector<string> & vecStrs,char  spl)
{
	string::size_type pos1, pos2;
	pos2 = str.find(spl);
	pos1 = 0;
	while (pos1 != pos2)
	{
		vecStrs.push_back(str.substr(pos1, pos2 - pos1));

		pos1 = pos2 + 1;
		pos2 = str.find(spl, pos1);
	}
	vecStrs.push_back(str.substr(pos1));
}
int ImageResize(int argc, char ** argv)
{
	//parse the cmd parameters
	int ch;
	opterr = 0;
	string dirIn = "\\";
	string dirOut = "\\";
	string filePath = "";
	int width=200, height=200;
	if (argc == 0)
	{
		cerr << "需要参数\n";
		return -1;
	}
	int mode = -1;
	while ((ch = getopt(argc, argv, "i:n:w:h:o:")) != -1)
	{
		switch (ch) {
		case 'i':
			dirIn = optarg;
			if (dirIn[dirIn.size() - 1] != '\\')
				dirIn+="\\";
			mode = 1;	//批量转换
			break;
		case 'n':
			filePath = optarg;
			mode = 2;	//单个文件转换
			break;
		case 'w':
			width = atoi(optarg);			
			break;
		case 'h':
			height = atoi(optarg);			
			break;
		case 'o':
			dirOut = optarg;
			if (dirOut[dirOut.size() - 1] != '\\')
				dirOut += "\\";
			break;
		default:
			break;
		}
	}

	//得到目录dirIn和文件名filesName

	vector<string> filesName;
	if (mode == 1)
	{
		vector<string> ext;
		ext.push_back("*.png");
		ext.push_back("*.jpg");
		GetFilesName(dirIn, filesName,ext);
		
	}
	else if (mode == 2)
	{
		vector<string> strs;
		MySplit(filePath, strs, '\\');
		filesName.push_back(*(--strs.end()));
		//构造所在目录
		dirIn = "";
		for (string::size_type i = 0; i < strs.size() - 1; i++)
		{
			dirIn += strs[i];
			dirIn += "\\";
		}
	}
	Mat srcImage, outImage;
	for (vector<string>::iterator itr = filesName.begin(); itr != filesName.end(); ++itr)
	{
		srcImage = imread(dirIn+*itr);
		if (srcImage.empty() == true)
		{
			cerr << "file <"<< *itr <<"> loaded failed\n";
			continue;
		}
		resize(srcImage, outImage, Size(width, height), 0, 0, 1);
		if (imwrite(dirOut + "RSZ_" + *itr, outImage) == false)
		{
			cerr << "file " << *itr << " written faild\n";
		}
	}
	return 0;
}
//************************************
// Method:    Demo_Camara
// FullName:  Demo_Camara
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************
int Demo_Camara(int argc, char ** argv)
{
	CvCapture * capture;
	if ((capture = cvCreateCameraCapture(0)) == NULL)
	{
		printf("Camera Open Fail!Please check your Camera.");
		//char c=cvWaitKey(33);
		return -1;
	}
	cvNamedWindow("Camera Show", CV_WINDOW_AUTOSIZE);
	IplImage * frame;
	
	while (1)
	{
		frame = cvQueryFrame(capture);

		if (!frame)
			break;
		cvShowImage("Camera Show", frame);
		char c = cvWaitKey(33);
		if (c == 27)
			break;

	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Camera Show");
	
	return 0;
}
//int List_Dir(int argc, char ** argv)
//{
//	int ch;
//	opterr = 0;
//	string path = "C:\\";
//	string fileType = "";
//	while ((ch = getopt(argc, argv, "d:t:")) != -1)
//	{
//		switch (ch) {
//		case 'd':
//			path = optarg;        //the image or video path
//			break;
//		default:
//			break;
//		}
//	}
//	vector<string> filesName;
//	GetFilesName(path, filesName);
//	for (vector<string>::iterator itr = filesName.begin(); itr != filesName.end(); ++itr)
//	{
//		cout << *itr << endl;
//	}
//	return 0;
int Kmeans(int argc, char ** argv)
{

}