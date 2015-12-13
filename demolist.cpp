//
//  demolist.cpp
//  OpenCVDemo
//
//  Created by 杨朔柳 on 15/11/10.
//  Copyright © 2015年 杨朔柳. All rights reserved.
//


#pragma warning(disable:4996)
#define __windows__

#include "demolist.h"
#include <stdio.h>
#include <iostream>
#include <queue>
#include <cstdlib>

#ifdef __windows__
#include "diropr.h"
#endif

#ifdef __posix__
#include <unistd.h>
#endif
#include <string.h>


//OpenCV for C++
#include "BaseCls.h"
#include <opencv2/opencv.hpp>		//for all
#include <opencv2/ml/ml.hpp>


#include "cv.h"  
#include "highgui.h"  
#include "cxcore.h"  
#include <ml.h>  
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/objdetect.hpp>

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

int Demo_Smooth(int argc, char ** argv)
{
	//
	//parse the cmd parameters
	//
	int ch;
	opterr = 0;
	string path;
	while ((ch = getopt(argc, argv, "d:")) != -1)
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
class FaceDetect :public VideoProcessor
{
private:

	Scalar color;
	Mat smallImgROI;
	vector<Rect> faces;			//脸的位置
	vector<Rect> nestedObjects;		//五官的位置
	Point center;			//中心坐标
	CascadeClassifier nestedCascade;
	CascadeClassifier cascade;
	Mat frame;
	FaceDetect() {}

public:
	FaceDetect(const string & _nestedCascadeName, const string & _cascadeName, const string & fileName) :VideoProcessor(fileName) {
		if (!nestedCascade.load(_nestedCascadeName))
		{
			cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		}

		if (!cascade.load(_cascadeName))
		{
			cerr << "ERROR: Could not load classifier cascade" << endl;
		}
	}
	void detectAndDraw(Mat& img,
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
	void Process()
	{
		while (1)
		{
			frame = cvarrToMat(cvQueryFrame(_cvCap));
			detectAndDraw(frame, 1, false);
		}
	}
	void Display()const {}

};
int Demo_FaceDetect(int argc, char ** argv)
{
	//parse the cmd parameters
	int ch;
	opterr = 0;
	string path = "C:\\haar\\demo.avi";
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
	FaceDetect objFaceDetect(nestedCascadeName, cascadeName, path);
	objFaceDetect.Process();
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
void MySplit(const string & str, vector<string> & vecStrs, char  spl)
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
	int width = 200, height = 200;
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
				dirIn += "\\";
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
		//GetFilesName(dirIn, filesName,ext);

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
		srcImage = imread(dirIn + *itr);
		if (srcImage.empty() == true)
		{
			cerr << "file <" << *itr << "> loaded failed\n";
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
	return 0;
}
int Kmeans(int argc, char ** argv)
{
	return 0;
}
//************************************
// Method:    Demo_Coutours
// FullName:  Demo_Coutours
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************
void ColorThreshold(IplImage * _src, IplImage * _binary, int threshold, int b0, int g0, int r0);
int Demo_Coutours(int argc, char ** argv)
{
	CvCapture * cap;
	IplImage * input;
	cap = cvCaptureFromAVI("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\test1.avi");
	if (cap == NULL)
	{
		cerr << "can not load video\n";
		return 1;
	}
	input = cvQueryFrame(cap);
	if (input == NULL)
	{
		cerr << "can not load the first frame\n";
		return -1;
	}

	cvNamedWindow("output1");
	cvNamedWindow("output2");

	IplImage * gray, *binary;
	gray = cvCreateImage(cvGetSize(input), 8, 1);
	binary = cvCreateImage(cvGetSize(input), 8, 1);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq * contour = 0;

	int contours = 0;
	while (1)
	{
		input = cvQueryFrame(cap);
		cvCvtColor(input, gray, CV_BGR2GRAY);
		cvThreshold(gray, binary, 128, 255, 0);
		contours = cvFindContours(binary, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		for (; contour != 0; contour = contour->h_next)
		{
			cvDrawContours(gray, contour, cvScalarAll(255), cvScalarAll(255), 1, 2, 8);
		}
		ColorThreshold(input, binary, 200, 175, 118, 66);
		waitKey(33);
		cvShowImage("output1", binary);
		cvShowImage("output2", input);
	}
	cvReleaseCapture(&cap);
	cvDestroyWindow("output1");
	cvDestroyWindow("output2");

	return 0;
}
//************************************
// Method:    Demo_ColorDetect
// FullName:  Demo_ColorDetect
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************
int g_ColorDetect_R;
int g_ColorDetect_G;
int g_ColorDetect_B;
int g_ColorDetect_dist;
int g_ColorDetect_b, g_ColorDetect_g, g_ColorDetect_r, g_ColorDetect_d;
IplImage * g_ColorDetect_input, *g_ColorDetect_binary, *g_ColorDetect_gray;

inline int GetColorDistance(int b, int g, int r, int b0, int g0, int r0)
{
	return (b - b0 > 0 ? b - b0 : b0 - b) + (g - g0 > 0 ? g - g0 : g0 - g) + (r - r0 > 0 ? r - r0 : r0 - r);
}
void ColorThreshold(IplImage * _src, IplImage * _binary, int threshold, int b0, int g0, int r0)
{
	char * data = _src->imageData;
	char * b, *g, *r;
	for (int i = 0; i < _src->height; i++)
	{
		for (int j = 0; j < _src->width; j++)
		{
			b = data + i*_src->widthStep + j*_src->nChannels;
			g = data + i*_src->widthStep + j*_src->nChannels + 1;
			r = data + i*_src->widthStep + j*_src->nChannels + 2;

			if (GetColorDistance(*b, *g, *r, b0, g0, r0) < threshold)
			{
				*(_binary->imageData + i*_binary->widthStep + j*_binary->nChannels) = 255;
			}
			else
			{
				*(_binary->imageData + i*_binary->widthStep + j*_binary->nChannels) = 0;
			}
		}
	}
}

void g_ColorDetect_update()
{
	ColorThreshold(g_ColorDetect_input, g_ColorDetect_binary, g_ColorDetect_dist, g_ColorDetect_B, g_ColorDetect_G, g_ColorDetect_R);
	cvShowImage("output", g_ColorDetect_binary);
	cvWaitKey(0);
}
void OnTrackBar_Blue(int pos)
{
	g_ColorDetect_B = pos;
	g_ColorDetect_update();
}
void OnTrackBar_Green(int pos)
{
	g_ColorDetect_G = pos;
	g_ColorDetect_update();
}
void OnTrackBar_Red(int pos)
{
	g_ColorDetect_R = pos;
	g_ColorDetect_update();
}
void OnTrackBar_Distance(int pos)
{
	g_ColorDetect_dist = pos;
	g_ColorDetect_update();
}
int Demo_ColorDetect(int argc, char ** argv)
{
	CvCapture * _cvCap = cvCaptureFromAVI("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\test1.avi");
	if (_cvCap == NULL)
	{
		return -1;
	}
	g_ColorDetect_input = cvQueryFrame(_cvCap);
	g_ColorDetect_gray = cvCreateImage(cvGetSize(g_ColorDetect_input), g_ColorDetect_input->depth, 1);
	g_ColorDetect_binary = cvCreateImage(cvGetSize(g_ColorDetect_input), g_ColorDetect_input->depth, CV_THRESH_BINARY);

	cvNamedWindow("output");
	cvCreateTrackbar("B", "output", &g_ColorDetect_b, 255, OnTrackBar_Blue);
	cvCreateTrackbar("G", "output", &g_ColorDetect_g, 255, OnTrackBar_Green);
	cvCreateTrackbar("R", "output", &g_ColorDetect_r, 255, OnTrackBar_Red);
	cvCreateTrackbar("D", "output", &g_ColorDetect_d, 255, OnTrackBar_Distance);


	while (g_ColorDetect_input != NULL)
	{
		ColorThreshold(g_ColorDetect_input, g_ColorDetect_binary, g_ColorDetect_dist, g_ColorDetect_B, g_ColorDetect_G, g_ColorDetect_R);
		cvShowImage("output", g_ColorDetect_binary);
	}

	cvReleaseImage(&g_ColorDetect_input);
	cvReleaseImage(&g_ColorDetect_gray);
	cvReleaseImage(&g_ColorDetect_binary);
	cvDestroyWindow("output");

	return 0;
}
//************************************
// Method:    Demo_Convolution
// FullName:  Demo_Convolution
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************
void Convolution(unsigned char * _ArrIn, unsigned char * _ArrOut, int _width, int _height, int _nChannal, unsigned char * _Knl, int _knlSize = 9)
{
	//归一化卷积核3*3
	float  * normalize = (float *)malloc(_knlSize*sizeof(float));
	float sum = 0.0;
	for (int i = 0; i < _knlSize; i++)
	{
		sum += *(_Knl + i);
	}
	for (int i = 0; i < _knlSize; i++)
	{
		*(normalize + i) = ((float)*(_Knl + i)) / sum;
	}
	int _center_x, _center_y, _KnlWidth;
	int widthSteps = _width*_nChannal;
	switch (_knlSize)
	{
	case 9:
		_center_y = 1,
			_center_x = 1;
		_KnlWidth = 3;
		break;
	default:
		break;
	}
	//处理每个通道
	for (int c = 0; c < 0; c++)
	{
		//处理上边界
		//处理(0,0)
		*(_ArrOut + 0 + 0 + c) = (*(_ArrIn + 0 * widthSteps + 0 * _nChannal + c))*(*(normalize + 4)) +
			(*(_ArrIn + 0 * widthSteps + 1 * _nChannal + c))*(*(normalize + 3)) +
			(*(_ArrIn + 1 * widthSteps + 0 * _nChannal + c))*(*(normalize + 1)) +
			(*(_ArrIn + 1 * widthSteps + 1 * _nChannal + c))*(*(normalize + 0));
		//处理(0,1)---->(0,width-1)
		for (int i = 1; i < _width - 1; i++)
		{
			*(_ArrOut + i*_nChannal + c) = (*(_ArrIn + 0 + i * _nChannal + c))*(*(normalize + 4)) +
				(*(_ArrIn + (0 - 0)*widthSteps + (i - 1) * _nChannal + c))*(*(normalize + 5)) +
				(*(_ArrIn + (0 + 0)*widthSteps + (i + 1) * _nChannal + c))*(*(normalize + 3)) +
				(*(_ArrIn + (0 + 1)*widthSteps + (i - 1) * _nChannal + c))*(*(normalize + 2)) +
				(*(_ArrIn + (0 + 1)*widthSteps + (i + 0) * _nChannal + c))*(*(normalize + 1)) +
				(*(_ArrIn + (0 + 1)*widthSteps + (i + 1) * _nChannal + c))*(*(normalize + 0));
		}
		//处理(0,width)
		*(_ArrOut + 0 + (_width - 1) * _nChannal + c) = (*(_ArrIn + 0 * widthSteps + (_width - 1) * _nChannal + c))*(*(normalize + 4)) +
			(*(_ArrIn + 0 * widthSteps + (_width - 2) * _nChannal + c))*(*(normalize + 5)) +
			(*(_ArrIn + 1 * widthSteps + (_width - 1) * _nChannal + c))*(*(normalize + 1)) +
			(*(_ArrIn + 1 * widthSteps + (_width - 2) * _nChannal + c))*(*(normalize + 2));
		//处理内部
		for (int j = 1; j < _height - 1; j++)
		{
			int i = 0;
			//(j,0)
			*(_ArrOut + (j + 0)*widthSteps + 0 + c) = (*(_ArrIn + (j + 0) * widthSteps + 0 * _nChannal + c))*(*(normalize + 4)) +
				(*(_ArrIn + (j + 0) * widthSteps + 1 * _nChannal + c))*(*(normalize + 3)) +
				(*(_ArrIn + (j + 1) * widthSteps + 0 * _nChannal + c))*(*(normalize + 1)) +
				(*(_ArrIn + (j + 1) * widthSteps + 1 * _nChannal + c))*(*(normalize + 0)) +
				(*(_ArrIn + (j - 1) * widthSteps + 0 * _nChannal + c))*(*(normalize + 7)) +
				(*(_ArrIn + (j - 1) * widthSteps + 1 * _nChannal + c))*(*(normalize + 6));

			for (++i; i < _width - 1; i++)
			{
				*(_ArrOut + (j + 0)*widthSteps + (i + 0) * _nChannal + c) = (*(_ArrIn + (j + 0)*widthSteps + (i + 0) * _nChannal + c))*(*(normalize + 4)) +
					(*(_ArrIn + (j - 1)*widthSteps + (i - 1) * _nChannal + c))*(*(normalize + 8)) +
					(*(_ArrIn + (j - 1)*widthSteps + (i + 0) * _nChannal + c))*(*(normalize + 7)) +
					(*(_ArrIn + (j - 1)*widthSteps + (i + 1) * _nChannal + c))*(*(normalize + 6)) +
					(*(_ArrIn + (j + 0)*widthSteps + (i - 1) * _nChannal + c))*(*(normalize + 5)) +
					(*(_ArrIn + (j + 0)*widthSteps + (i + 1) * _nChannal + c))*(*(normalize + 3)) +
					(*(_ArrIn + (j + 1)*widthSteps + (i - 1) * _nChannal + c))*(*(normalize + 2)) +
					(*(_ArrIn + (j + 1)*widthSteps + (i + 0) * _nChannal + c))*(*(normalize + 1)) +
					(*(_ArrIn + (j + 1)*widthSteps + (i + 1) * _nChannal + c))*(*(normalize + 0));
			}
			//(j,width-1)
			*(_ArrOut + j*widthSteps + (_width - 1) * _nChannal + c) = (*(_ArrIn + (j + 0) * widthSteps + (_width - 1) * _nChannal + c))*(*(normalize + 4)) +
				(*(_ArrIn + (j + 0) * widthSteps + (_width - 2) * _nChannal + c))*(*(normalize + 5)) +
				(*(_ArrIn + (j + 1) * widthSteps + (_width - 1) * _nChannal + c))*(*(normalize + 1)) +
				(*(_ArrIn + (j + 1) * widthSteps + (_width - 2) * _nChannal + c))*(*(normalize + 2)) +
				(*(_ArrIn + (j - 1) * widthSteps + (_width - 1) * _nChannal + c))*(*(normalize + 7)) +
				(*(_ArrIn + (j - 1) * widthSteps + (_width - 2) * _nChannal + c))*(*(normalize + 8));

		}
		//处理下边界

		*(_ArrOut + (_height - 1)*widthSteps + 0 + c) = (*(_ArrIn + (_height - 1) * widthSteps + 0 * _nChannal + c))*(*(normalize + 4)) +
			(*(_ArrIn + (_height - 1) * widthSteps + 1 * _nChannal + c))*(*(normalize + 3)) +
			(*(_ArrIn + (_height - 2) * widthSteps + 0 * _nChannal + c))*(*(normalize + 7)) +
			(*(_ArrIn + (_height - 2) * widthSteps + 1 * _nChannal + c))*(*(normalize + 6));
		//处理(0,1)---->(0,width-1)
		for (int i = 1; i < _width - 1; i++)
		{
			*(_ArrOut + (_height - 1)*widthSteps + (i + 0) * _nChannal + c) = (*(_ArrIn + (_height - 1)*widthSteps + (i + 0) * _nChannal + c))*(*(normalize + 4)) +
				(*(_ArrIn + (_height - 1)*widthSteps + (i - 1) * _nChannal + c))*(*(normalize + 5)) +
				(*(_ArrIn + (_height - 1)*widthSteps + (i + 1) * _nChannal + c))*(*(normalize + 3)) +
				(*(_ArrIn + (_height - 2)*widthSteps + (i - 1) * _nChannal + c))*(*(normalize + 8)) +
				(*(_ArrIn + (_height - 2)*widthSteps + (i + 0) * _nChannal + c))*(*(normalize + 7)) +
				(*(_ArrIn + (_height - 2)*widthSteps + (i + 1) * _nChannal + c))*(*(normalize + 6));
		}
		//处理(height,width)
		*(_ArrOut + (_height - 1)*widthSteps + (_width - 1)*_nChannal + c) = (*(_ArrIn + (_height - 1) * widthSteps + (_width - 1)*_nChannal + c))*(*(normalize + 4)) +
			(*(_ArrIn + (_height - 1) * widthSteps + (_width - 2)*_nChannal + c))*(*(normalize + 5)) +
			(*(_ArrIn + (_height - 2) * widthSteps + (_width - 1)*_nChannal + c))*(*(normalize + 7)) +
			(*(_ArrIn + (_height - 2) * widthSteps + (_width - 2)*_nChannal + c))*(*(normalize + 8));
	}
	free(normalize);
}

int Demo_Convolution(int argc, char ** argv)
{
	IplImage * input = cvLoadImage("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\testimg.jpg");
	IplImage * output = cvCreateImage(cvGetSize(input), 8, 3);
	IplImage * output2 = cvCreateImage(cvGetSize(input), 8, 3);
	unsigned char krnl[] = { -1,0,1,-2,0,2,-1,0,1 };
	Convolution((unsigned char *)input->imageData, (unsigned char *)output->imageData, input->width, input->height, 3, krnl, 9);
	cvSmooth(input, output2, CV_GAUSSIAN, 3, 3, 0, 0);
	cvNamedWindow("src");
	cvNamedWindow("dst");
	cvNamedWindow("dst2");
	cvShowImage("src", input);
	cvShowImage("dst", output);
	cvShowImage("dst", output2);
	cvWaitKey(0);
	cvDestroyWindow("src");
	cvDestroyWindow("dst");
	cvDestroyWindow("dst2");
	cvReleaseImage(&input);
	cvReleaseImage(&output);
	cvReleaseImage(&output2);
	return 0;
}
//************************************
// Method:    Divide_HSV
// FullName:  Divide_HSV
// Access:    public 
// Returns:   int
// Qualifier:
// Parameter: int argc
// Parameter: char * * argv
//************************************
void HSV2RGB(int h, double s, double v, unsigned char * r, unsigned char * g, unsigned char * b)
{
	double hi = (double)h / 60.0;
	double f = hi - (int)hi;
	double p = v*(1 - s);
	double q = v*(1 - f*s);
	double t = v*(1 - (1 - f)*s);
	double R, G, B;
	switch ((int)hi)
	{
	case 0:
		R = v; G = t; B = p;
		break;
	case 1:
		R = q; G = v; B = p;
		break;
	case 2:
		R = p; G = v; B = t;
		break;
	case 3:
		R = p; G = q; B = v;
		break;
	case 4:
		R = t; G = p; B = v;
		break;
	case 5:
		R = v; G = p; B = q;
		break;
	default:
		break;
	}
	*r = R * 255; *g = G * 255; *b = B * 255;
}
void ColorThreshold(IplImage * _src, IplImage * _binary, int threshold, int h0)
{
	char * data = _src->imageData;
	float * h, *s, *v;
	for (int i = 0; i < _src->height; i++)
	{
		for (int j = 0; j < _src->width; j++)
		{
			h = (float*)((data)+i*_src->widthStep + j*(_src->nChannels*sizeof(float)));
			//s = ((float*)data + i*_src->widthStep + j*_src->nChannels + 1);
			//v = ((float*)data + i*_src->widthStep + j*_src->nChannels + 2);

			if ((*h - h0 > 0 ? *h - h0 : h0 - *h) < threshold)
			{
				*(_binary->imageData + i*_binary->widthStep + j*_binary->nChannels) = 255;
			}
			else
			{
				*(_binary->imageData + i*_binary->widthStep + j*_binary->nChannels) = 0;
			}
		}
	}
}
int _HSV_h;
int _HSV_d;
IplImage * _HSV_src;
IplImage * _HSV_srcto32;
IplImage * _HSV_hsv;
IplImage * _HSV_binary2, *_HSV_binary_coutour;
CvMemStorage* _HSV_storage;
CvSeq * _HSV_contour = 0;
void _HSV_update2()
{

	ColorThreshold(_HSV_hsv, _HSV_binary2, _HSV_d, _HSV_h);
	cvErode(_HSV_binary2, _HSV_binary2);		//腐蚀
	cvErode(_HSV_binary2, _HSV_binary2);
	cvDilate(_HSV_binary2, _HSV_binary2);		//膨胀
	cvDilate(_HSV_binary2, _HSV_binary2);
	cvFindContours(_HSV_binary2, _HSV_storage, &_HSV_contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cvZero(_HSV_binary_coutour);
	//画轮廓
	for (; _HSV_contour != 0; _HSV_contour = _HSV_contour->h_next)
	{
		cvDrawContours(_HSV_binary_coutour, _HSV_contour, cvScalarAll(255), cvScalar(255, 0, 0), 1, 1, 8);
	}
	cvShowImage("output1", _HSV_binary2);
	cvShowImage("output2", _HSV_binary_coutour);
	waitKey(20);
}
int Divide_HSV(int argc, char ** argv)
{
	/*src = cvLoadImage("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\testimg3.png");
	if (src == NULL)
	{
		cerr << "can not load image\n";
		return -1;
	}*/

	CvCapture * cap = cvCaptureFromAVI("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\test1.avi");
	if (cap == NULL)
	{
		return -1;
	}

	_HSV_src = cvQueryFrame(cap);

	_HSV_srcto32 = cvCreateImage(cvGetSize(_HSV_src), IPL_DEPTH_32F, 3);
	_HSV_hsv = cvCreateImage(cvGetSize(_HSV_src), IPL_DEPTH_32F, 3);
	_HSV_binary2 = cvCreateImage(cvGetSize(_HSV_src), IPL_DEPTH_8U, 1);
	_HSV_binary_coutour = cvCreateImage(cvGetSize(_HSV_src), IPL_DEPTH_8U, 1);
	cvNamedWindow("output1");
	cvNamedWindow("output2");
	_HSV_storage = cvCreateMemStorage(0);

	_HSV_d = 20; _HSV_h = 208;
	unsigned char r, g, b;

	while (_HSV_src != NULL)
	{
		_HSV_src = cvQueryFrame(cap);
		cvConvertScale(_HSV_src, _HSV_srcto32, 1.0, 0);
		cvCvtColor(_HSV_srcto32, _HSV_hsv, CV_BGR2HSV);
		_HSV_update2();
	}

	//cvCreateTrackbar("h", "output", &h, 360, On_H);
	//cvCreateTrackbar("d", "output", &d, 360, On_T);
	//d = 16; h = 202;
	//update2();
	cvDestroyWindow("output1");
	cvDestroyWindow("output2");
	cvReleaseImage(&_HSV_srcto32);
	cvReleaseImage(&_HSV_hsv);
	cvReleaseImage(&_HSV_binary2);
	return 0;
}
int Demo_SVM(int argc, char ** argv)
{
	return 0;
}
//
//
//Demo_Hough
//
//
Mat g_Hough_input;
Mat g_Hough_gray;
vector<Vec3f> g_Hough_circles;
int g_Hough_dp;
int g_Hough_minDist;
int g_Hough_cannyThreshold;
int g_Hough_Threshold;
void Hough_update()
{
	IplImage display;
	HoughCircles(g_Hough_gray, g_Hough_circles,
		CV_HOUGH_GRADIENT,
		g_Hough_dp,
		g_Hough_minDist,
		g_Hough_cannyThreshold,
		g_Hough_Threshold,
		0,
		0);
	int count = 0;
	Mat g_Hough_display = g_Hough_input.clone();
	for (size_t i = 0; i < g_Hough_circles.size(); i++)
	{
		Point center(cvRound(g_Hough_circles[i][0]), cvRound(g_Hough_circles[i][1]));
		int r = cvRound(g_Hough_circles[i][2]);
		circle(g_Hough_input, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(g_Hough_input, center, r, Scalar(255, 0, 0), 3, 8, 0);
	}
	//imshow("res", g_Hough_input);
	display = g_Hough_display;
	cvShowImage("res", &display);
	waitKey(0);
}
void On_minDistChanged(int pos)
{
	g_Hough_minDist = pos;
	Hough_update();
}
void On_dpChanged(int pos)
{
	g_Hough_dp = pos;
	Hough_update();
}
void On_cannyThresholdChanged(int pos)
{
	g_Hough_cannyThreshold = pos;
	Hough_update();
}
void On_ThresholdChanged(int pos)
{
	g_Hough_Threshold = pos;
	Hough_update();
}
int Demo_Hough(int argc, char ** argv)
{
	g_Hough_input = imread("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\circle1.jpg");
	if (g_Hough_input.empty() == true)
	{
		return -1;
	}
	cvtColor(g_Hough_input, g_Hough_gray, CV_BGR2GRAY);
	GaussianBlur(g_Hough_gray, g_Hough_gray, Size(9, 9), 2, 2);
	g_Hough_dp = 1;
	g_Hough_minDist = 10;
	g_Hough_cannyThreshold = 150;
	g_Hough_Threshold = 50;
	cvNamedWindow("res");
	cvCreateTrackbar("dp", "res", &g_Hough_dp, 30, On_dpChanged);
	cvCreateTrackbar("minDist", "res", &g_Hough_minDist, 200, On_minDistChanged);
	cvCreateTrackbar("canny", "res", &g_Hough_cannyThreshold, 255, On_cannyThresholdChanged);
	cvCreateTrackbar("thres", "res", &g_Hough_Threshold, 300, On_ThresholdChanged);
	Hough_update();
	cvvDestroyWindow("res");
	return 0;
}

//
//Canny
//
//
Mat g_Canny_input;
Mat g_Canny_output;
int g_Canny_threshold1;
int g_Canny_threshold2;
int g_Canny_apertureSize = 3;
void Canny_update()
{
	Canny(g_Canny_input, g_Canny_output, g_Canny_threshold1, g_Canny_threshold2, g_Canny_apertureSize, false);
	imshow("res", g_Canny_output);
	cvWaitKey(0);
}
void On_threshold1Changed(int pos, void *)
{
	g_Canny_threshold1 = pos;
	Canny_update();
}
void On_threshold2Changed(int pos, void *)
{
	g_Canny_threshold2 = pos;
	Canny_update();
}
void On_apertureSizeChanged(int pos, void *)
{
	g_Canny_apertureSize = pos;
	Canny_update();
}
int Demo_Canny(int argc, char ** argv)
{
	g_Canny_input = imread("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\testimg.jpg");
	if (g_Canny_input.empty() == true)
	{
		return -1;
	}
	g_Canny_threshold1 = 50;
	g_Canny_threshold2 = 100;
	g_Canny_apertureSize = 3;
	namedWindow("res");
	createTrackbar("thres1", "res", &g_Canny_threshold1, 255, On_threshold1Changed);
	createTrackbar("thres2", "res", &g_Canny_threshold2, 255, On_threshold2Changed);
	createTrackbar("apertureSize", "res", &g_Canny_apertureSize, 5, On_apertureSizeChanged);

	Canny_update();
	return 0;
}
//
//
// Demo_Elipse
//
int Demo_Elipse(int argc, char ** argv)
{
	IplImage * input, *gray, *binary, *img_coutours;
	input = cvLoadImage("C:\\Users\\yangs\\Documents\\Visual Studio 2015\\Projects\\opencvfrmwrk\\Debug\\ellipsetest.png");
	gray = cvCreateImage(cvGetSize(input), input->depth, 1);
	binary = cvCreateImage(cvGetSize(input), input->depth, 1);
	img_coutours = cvCreateImage(cvGetSize(input), input->depth, 3);
	cvZero(img_coutours);
	CvMemStorage * storage = cvCreateMemStorage(0);
	CvSeq * seq = NULL;

	int cnt = cvFindContours(binary, storage, &seq, sizeof(CvContour), 2);
	cvCvtColor(input, gray, CV_BGR2GRAY);

	//cvThreshold(gray, binary, 0, 255, CV_THRESH_BINARY);

	cvCanny(gray, binary, 50, 150);

	cvFindContours(binary, storage, &seq, sizeof(CvContour),
		CV_RETR_TREE,		//以线性表形式存储
		CV_CHAIN_APPROX_NONE);	//

	//层序遍历画轮廓

	queue<CvSeq *> que;
	for (CvSeq * sibling = seq; sibling != NULL; sibling = sibling->h_next)
	{
		que.push(sibling);
	}
	while (que.empty() == false)
	{
		CvSeq * s = que.front();
		//draw contours
		cvDrawContours(img_coutours, s,
			CvScalar(255, 0, 0),
			CvScalar(0, 255, 0),
			0,					//max_level
			1,
			8);
		//fit ellipse
		CvBox2D b = cvFitEllipse2(s);
		b.angle = 90 - b.angle;
		printf("-----------------\narea:%f\n", cvContourArea(s));
		printf("arclenth:%f\n", cvArcLength(s));
		printf("enter of ellipse:(%f %f)\n", b.center.x, b.center.y);
		printf("width and height:%d %d\n------------------\n", b.size.width, b.size.height);
		
		que.pop();
		cvShowImage("coutours", img_coutours);
		cvWaitKey(0);
		for (CvSeq * child = s->v_next; child != NULL; child = child->h_next)
		{
			que.push(child);
		}
	}
	cvNamedWindow("res");
	cvNamedWindow("coutours");
	cvShowImage("res", input);
	cvShowImage("coutours", img_coutours);
	cvWaitKey(0);
	cvDestroyWindow("res");
	cvDestroyWindow("coutours");
	return 0;
}