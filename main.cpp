//
//  main.cpp
//  OpenCVDemo
//
//  Created by 杨朔柳 on 15/11/10.
//  Copyright © 2015年 杨朔柳. All rights reserved.
//
extern "C"
{
#include "menu.h"
}

#include <stdio.h>
#include <cstdlib>
#include "demolist.h"

using namespace std;

int main(int argc, const char * argv[])
{
    MenuConfig("list", "list demos", help);
    MenuConfig("quit", "quit from the demo", quit);
    MenuConfig("1", "smooth image demo", Demo_Smooth);
	MenuConfig("2", "face detect", Demo_FaceDetect);
	MenuConfig("3", "image resize", ImageResize);
    ExecuteMenu();
}
