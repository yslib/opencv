
#include <iostream>
#include <string.h>
#include <io.h>
#include "diropr.h"
using namespace std;
void PrintFiles(const char * DirName)
{
	using namespace std;
	_finddata_t file;
	int k;
	long HANDLE;

	k = HANDLE = _findfirst("C:\\*.*", &file);

	while (k != -1)
	{
		cout << file.name << endl;
		k = _findnext(HANDLE, &file);
	}
	_findclose(HANDLE);
}


void GetFilesName(const std::string & dirName, std::vector<std::string> & fileName, const std::vector<string> & extension)
{
	_finddata_t file;
	string filePath;		//all files
	long fHandle;
	for (vector<string>::const_iterator itr = extension.begin(); itr != extension.end(); ++itr)
	{
		filePath = dirName + *itr;
		fHandle = _findfirst(filePath.c_str(), &file);
		long k = fHandle;
		while (k != -1)
		{
			fileName.push_back(string(file.name));
			k = _findnext(fHandle, &file);
		}
	}
	_findclose(fHandle);
}
