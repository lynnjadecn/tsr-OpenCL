// laneDCL.h: interface for the ClaneDCL class.
//
//////////////////////////////////////////////////////////////////////
#include "cv.h"
#include "highgui.h"
#include <vector>

using namespace std;
#if !defined(AFX_LANEDCL_H__4C7EF046_2589_46FA_9D7F_5E2B7D92D8DC__INCLUDED_)
#define AFX_LANEDCL_H__4C7EF046_2589_46FA_9D7F_5E2B7D92D8DC__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CsignDcl  
{
public:
	CsignDcl();
	virtual ~CsignDcl();
public:
	static bool RecogResult[32];
	static int RGB2HSI(int R,int G,int B,double& H,double& S,double& I);//RGB2HSI
	static CvHaarClassifierCascade* load_object_detector( const char* cascade_path );//ºÏ≤‚∑À¿‡∆˜‘ÿ»Î
	static int CompareRect(CvRect rect1, CvRect rect2);
	static double Distance2Rect(CvRect rect1,CvRect rect2);
	static void Image2RBYImage(IplImage*image,IplImage*imageR,IplImage*imageB,IplImage*imageY,IplImage*imageG/*,IplImage*imageBlack*/);
	static void TrafficSignDwithColor(IplImage* image,IplImage* imagegray,IplImage* imagedetect,CvHaarClassifierCascade*cascade[],int n,vector<int> *signs);
	static int SingleRectRecog(IplImage*image);
	static void LoadRecogPara();
	static void TrafficSignRecognition(IplImage*image,vector<int> *signs);
	static bool TrafficLightDwithRect(IplImage*image,CvRect rect,int n);
	static double RectBlackRate(IplImage*image,CvRect rect);
	static void TrafficLightRwithImg(IplImage*image,IplImage*imagegray,int n);
	static void Image2RBYImageforLight(IplImage*image,IplImage*imageR,IplImage*imageY,IplImage*imageG);
	static void TrafficLightRecognition(IplImage* image);
	
};

#endif // !defined(AFX_LANEDCL_H__4C7EF046_2589_46FA_9D7F_5E2B7D92D8DC__INCLUDED_)
