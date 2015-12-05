//
//  main.cpp
//  OpenCVDemo
//
//  Created by 杨朔柳 on 15/11/10.
//  Copyright © 2015年 杨朔柳. All rights reserved.
//



#include <stdio.h>
#include <cstdlib>

#include "demolist.h"
#include "menu.h"
using namespace std;

int main(int argc, const char * argv[])
{
    MenuConfig("l", "list demos", help);
    MenuConfig("q", "quit from the demo", quit);
    MenuConfig("1", "smooth image demo", Demo_Smooth);
	MenuConfig("2", "face detect", Demo_FaceDetect);
	//MenuConfig("3", "image resize", ImageResize);
	MenuConfig("4", "camara", Demo_Camara);
    MenuConfig("5", "coutours", Demo_Coutours);
	MenuConfig("6", "colordetect", Demo_ColorDetect);
	MenuConfig("7", "Convolution", Demo_Convolution);
	MenuConfig("8", "HSV", Divide_HSV);
    ExecuteMenu();
}
