#pragma once
#ifndef _DIROPR_H_
//#ifdef __cplusplus
//extern "C"
//{
//#endif
//
#include <vector>
#include <string>
	//void PrintFiles(const char * DirName);

	int DirectoryExists(const char * dirName);

//#ifdef __cplusplus
//}
//#endif // __cplusplus

void GetFilesName(const std::string & dirName,
	std::vector<std::string> & fileName,
	const std::vector<std::string> & extension = std::vector<std::string>(1,std::string("*.*")));
#endif/* _DIROPR_H_ */