// laneDCL.h: interface for the ClaneDCL class.
//
//////////////////////////////////////////////////////////////////////
#include "openCVHeaders.h"

#if !defined(AFX_LANEDCL_H__4C7EF046_2589_46FA_9D7F_5E2B7D92D8DC__INCLUDED_)
#define AFX_LANEDCL_H__4C7EF046_2589_46FA_9D7F_5E2B7D92D8DC__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

//检测一张图片内标志牌结果的结构体
struct TraSign
{
	CvMemStorage *storage;
	CvSeq* seq;
	CvMemStorage *idstorage;
	CvSeq* id;
} ;
TraSign* CreateTraSign();
void ReleaseTraSign(TraSign *trs);

class ClaneDCL  
{
public:
	ClaneDCL();
	virtual ~ClaneDCL();
public:
	static BOOL RecogResult[32];
	static int RGB2HSI(int R,int G,int B,double& H,double& S,double& I);//RGB2HSI
	static void color_statistics(IplImage *image, double & redration, 
		double & blueration, double & yellowration, double & blackration, double & whiteration);//统计一幅图里红蓝黄黑白的比例
	static void color_circle_points(int xc,int yc,int x,int y,
		IplImage*image,int&nred,int&nblue,int&ntotal);//圆形目标检测
	static void get_circle(int xc,int yc,int radius,IplImage* image,int&nred,int&nblue,int&ntotal);//圆形目标检测
	static int circlehist(IplImage*image,int ncolor);//圆形目标检测
	static CvHaarClassifierCascade* load_object_detector( const char* cascade_path );//检测匪类器载入
	static int rect_decide(IplImage*image, CvRect rect,int n_color);
	static CvSeq* detect_and_draw_objects( IplImage* image, CvHaarClassifierCascade* cascade,int do_pyramids, int n_color );
	static TraSign* TrafficSignDetection(IplImage*image,CvHaarClassifierCascade*cascade[],int n);
	static int CompareRect(CvRect rect1, CvRect rect2);
	static double Distance2Rect(CvRect rect1,CvRect rect2);
	static CvRect GetRedRect(IplImage*image);
	static CvRect GetBlueRect(IplImage*image);
	static CvRect GetYellowRect(IplImage*image);
	static CvSeq* detect_withRedRect(IplImage*image,CvRect rect,CvHaarClassifierCascade*cascade[],int n);
	static CvSeq* detect_and_draw_objects2( IplImage* image, CvHaarClassifierCascade* cascade,int do_pyramids, int n_color );
	static void DetectRedRect(IplImage*image,CvHaarClassifierCascade*cascade[],int n);
	static void DetectBlueRect(IplImage*image,CvHaarClassifierCascade*cascade[],int n);
	static void DetectYellowRect(IplImage*image,CvHaarClassifierCascade*cascade[],int n);
	static void Image2RBYImage(IplImage*image,IplImage*imageR,IplImage*imageB,IplImage*imageY,IplImage*imageG/*,IplImage*imageBlack*/);
	static void TrafficSignDwithColor(IplImage*image,IplImage*imagegray,CvHaarClassifierCascade*cascade[],int n);
	static int SingleRectRecog(IplImage*image);
	static void LoadRecogPara();
	static void TrafficSignRecognition(IplImage*image);
	static bool TrafficLight(IplImage*image,CvRect rect,int n);
	static double RectBlackRate(IplImage*image,CvRect rect);
};

#endif // !defined(AFX_LANEDCL_H__4C7EF046_2589_46FA_9D7F_5E2B7D92D8DC__INCLUDED_)
