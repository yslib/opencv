#include <opencv2/opencv.hpp>

#include <string>
using namespace std;
using namespace cv;
class BaseProcessor
{
private:
	string _FileName;
public:
	BaseProcessor() {}
	BaseProcessor(const string & fileName) :_FileName(fileName) {}
	virtual bool Open(const string & fileName)=0;
	virtual bool IsOpen()const=0;
	virtual void Display()const = 0;
	virtual void Process()=0;
	virtual ~BaseProcessor() {}
};


class ImageProcessor:public BaseProcessor
{
	protected:
	IplImage _IplSrc;
	IplImage * _pIplSrc;
	Mat _MatSrc;
	void _Sync() { _IplSrc = _MatSrc; _pIplSrc = &_IplSrc; }
public:
	ImageProcessor():BaseProcessor(){}
	ImageProcessor(const string & fileName) :BaseProcessor(fileName) 
	{
		_MatSrc = imread(fileName);
		_Sync();
	}
	~ImageProcessor() {}
	bool Open(const string & fileName)
	{
		if (_MatSrc.empty() == true)
		{
			_MatSrc = imread(fileName);
			_Sync();
			return IsOpen();
		}
	}
	bool IsOpen()const { return _MatSrc.empty() == false; }
	void Display()const 
	{
		imshow("Display", _MatSrc);
	}
	void Process() {}
};
class VideoProcessor:public BaseProcessor
{
	protected:
	CvCapture * _cvCap;

public:
	VideoProcessor() {}
	VideoProcessor(const string & fileName) :BaseProcessor(fileName) {
		_cvCap = cvCaptureFromAVI(fileName.c_str());
	}
	VideoProcessor(int device) {
		_cvCap = cvCaptureFromCAM(device);
	}
	~VideoProcessor() { cvReleaseCapture(&_cvCap); }

	bool Open(const string & fileName)
	{
		if (_cvCap == NULL)
		{
			_cvCap = cvCaptureFromAVI(fileName.c_str());
		}
		return IsOpen();
	}
	bool Open(int device) {
		if (_cvCap == NULL)
		{
			_cvCap = cvCaptureFromCAM(device);
		}
		return IsOpen();
	}
	bool IsOpen()const { return _cvCap != NULL; }
	void Process() {}
	void Display()const {}
};
