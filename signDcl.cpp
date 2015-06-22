// LaneDCL.cpp: implementation of the LaneDCL class.
//
//////////////////////////////////////////////////////////////////////
//#include "stdAfx.h"
#include "signDcl.h"
#include "iostream"
#include <cmath>
//#include "LightPrep.h"
//#include "FaceFeature.h"
//#include "Subspace.h"
#include"cv.h"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "contrib/contrib.hpp"
#include"FaceFeature.h"
#include "LightPrep.h"
using namespace std;
 using namespace cv;
////////////////////////检测器///////////////////////////////
CvHaarClassifierCascade* cascadeR[5];	
CvHaarClassifierCascade* cascadeB[4];
CvHaarClassifierCascade* cascadeY[2];
CvHaarClassifierCascade* cascadeG;

Ptr<FaceRecognizer> model = createFisherFaceRecognizer(10,100);

CLightPrep p_recog;
 CFaceFeature f_recog;
 CvSize size_recog; 

  int featureSz;

//CvSize size_recog; //识别分类器训练时的尺寸
//int nTsign;//用于训练的样本个数
//int mdSz;//最后采用的特征维数
//CvMat*tsignNumTruthMat;//标志的类号，在训练时train.txt里写清楚。
//CvMat*W_pcafldT;//特征向量矩阵，用于最后的变换
//CvMat*mu_total;//均值
//CvMat**modellda;//样本最后的模式
//int featureSz;
/////chenlong//////
//
//////////////用于对样本进行变换的参数，用到了这三个类//////////////
//CLightPrep p_recog;
//CFaceFeature f_recog;
//CSubspace ss_recog;
////////////////////////////////////////////////////////////////////
//////////////////数据传输//////////////////////////////
// WSADATA w;								/* Used to open Windows connection */
// unsigned short port_number;				/* The port number to use */
// SOCKET sd;								/* The socket descriptor */
// int server_length;						/* Length of server struct */
// char send_buffer[100] = "";/* Data to send */
// time_t current_time;					/* Time received */
// struct hostent *hp;						/* Information about the server */
// struct sockaddr_in server;				/* Information about the server */
// struct sockaddr_in client;				/* Information about the client */
// int a1=192;
// int a2=168;
// int a3=1;
// int a4=123;						/* Server address components in xxx.xxx.xxx.xxx form */
// int b1, b2, b3, b4;						/* Client address components in xxx.xxx.xxx.xxx form */
// char host_name[256];					/* Host name of this computer */
///////////////////////////////////////////////////////////////////////////

bool CsignDcl::RecogResult[32] = {FALSE};
CsignDcl::CsignDcl()
{

}

CsignDcl::~CsignDcl()
{

}


int CsignDcl::RGB2HSI(int R,int G,int B,double& H,double& S,double& I)
{
	if (R<0 || R>255 || G<0 || G>255 || B<0 || B>255)
	{
		cout << "Value out of range!" << endl;
		return 1;
	}
	double min, mid, max;
	if (R>G && R>B)
	{
		max = R;
		mid = MAX(G, B);
		min = MIN(G, B);
	}
	else 
	{
		if (G>B) 
		{
			max = G;
			mid = MAX(R, B);
			min = MIN(R, B);
		}
		else 
		{
			max = B;
			mid = MAX(R, G);
			min = MIN(R, G);
		}
	}
	I = max / 255;
	S = 0;
	H = 0;
	if (I==0 || max==min)
	{
		// this is a black image or grayscale image
		S = 0;
		H = 0;
	}
	else 
	{
		S = (I - min/255) / I;
		// domains are 60 degrees of
		// red, yellow, green, cyan, blue or magenta
		double domainBase = 0.0;
		double oneSixth = 1.0/6.0;
		double domainOffset = ( (mid-min) / (max-min) ) / 6.0;
		
		if (R==max) 
		{
			if (mid==G)
			{ // green is ascending
				domainBase = 0; // red domain
			}
			else 
			{ // blue is descending
				domainBase = 5.0/6.0; // magenta domain
				domainOffset = oneSixth - domainOffset;
			}
		}
		else {
			if (G==max) 
			{
				if (mid==B)
				{ // blue is ascending
					domainBase = 2.0/6.0; // green domain
				}
				else
				{ // red is ascending
					domainBase = 1.0/6.0; // yellow domain
					domainOffset = oneSixth - domainOffset;
				}
			}
			else 
			{
				if (mid==R) 
				{ // red is ascending
					domainBase = 4.0/6.0; // blue domain
				}
				else 
				{ // green is descending
					domainBase = 3.0/6.0; // cyan domain
					domainOffset = oneSixth - domainOffset;
				}
			}
		}
		H = domainBase + domainOffset;
	}			
	return 0;
}

//分类器载入
CvHaarClassifierCascade* CsignDcl:: load_object_detector( const char* cascade_path )
{
    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}

int CsignDcl::CompareRect(CvRect rect1, CvRect rect2)
{
	int s;
	CvRect result;
	CvPoint p1_tl;
	CvPoint p1_br;
	
	CvPoint p2_tl;
	CvPoint p2_br;

	CvPoint pc_tl;
	CvPoint pc_br;

	p1_tl.x = rect1.x;
	p1_tl.y = rect1.y;
	
	p1_br.x = rect1.x+rect1.width;
	p1_br.y = rect1.y +rect1.height;
	
	p2_tl.x = rect2.x;
	p2_tl.y = rect2.y;
	
	p2_br.x = rect2.x+rect2.width;
	p2_br.y = rect2.y +rect2.height;

	pc_tl.x = max(p1_tl.x,p2_tl.x);
	pc_tl.y = max(p1_tl.y,p2_tl.y);

	pc_br.x = min(p1_br.x,p2_br.x);
	pc_br.y = min(p1_br.y,p2_br.y);
	
	if   (   pc_tl.x <pc_br.x     &&   pc_tl.y   <pc_br.y   )
	s   =   (pc_br.x-pc_tl.x)*(pc_br.y-pc_tl.y );
	else
	s   =   0;	
	
	result.x = pc_tl.x;
	result.y = pc_tl.y;
	result.width = pc_br.x - pc_tl.x;
	result.height = pc_br.y - pc_tl.y;
	return s;
}

double CsignDcl::Distance2Rect(CvRect rect1,CvRect rect2)
{
	
	int x1 = rect1.x+rect1.width/2;
	int y1 = rect1.y+rect1.height/2;
	int x2 = rect2.x+rect2.width/2;
	int y2 = rect2.y+rect2.height/2;
	double d = sqrt((double)(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
	return d;
}

void CsignDcl::Image2RBYImage(IplImage*image,IplImage*imageR,IplImage*imageB,IplImage*imageY,IplImage*imageG)
{
	IplImage *Imagerect = cvCreateImage(cvGetSize(image),8,3);
	cvCopy(image,Imagerect);
	//cvSmooth(image,Imagerect,CV_GAUSSIAN,3,3);	//高斯模糊
	cvZero(imageR);
	cvZero(imageB);
	cvZero(imageY);
	cvZero(imageG);
	//cvZero(imageBlack);
	int height=Imagerect->height;
	int width=Imagerect->width;
	int step=Imagerect->widthStep;
	int step1 = imageR->widthStep;
	int channels=Imagerect->nChannels;
	uchar *data1=(uchar*)Imagerect->imageData;
	uchar *data_R=(uchar*)imageR->imageData;
	uchar *data_B=(uchar*)imageB->imageData;
	uchar *data_Y=(uchar*)imageY->imageData;
	uchar *data_G=(uchar*)imageG->imageData;
	//uchar *data_Black=(uchar*)imageBlack->imageData;
	int i,j;
	double r,g,b,s,h,v,H;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (((H>0&&H<10) || (H>340&&H<360))&&s>0.4&&v>0.2)//R (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data_R[i*step1+j] = 255;
			}
			else if (H>10&&H<60&&s>0.4&&v>0.5)//Y (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data_Y[i*step1+j] = 255;
			}
			else if (H>190&&H<230&&s>0.5&&v>0.5)//B(H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data_B[i*step1+j] = 255;
			}
			else if (H>65&&H<100&&s>0.2&&v>0.5)
			{
				data_G[i*step1+j] = 255;
			}
		}
		cvDilate(imageR, imageR, 0, 1);
		cvErode(imageR, imageR, 0, 1);
		cvDilate(imageB, imageB, 0, 1);
		cvErode(imageB, imageB, 0, 1);
		cvDilate(imageY, imageY, 0, 4);
		cvErode(imageY, imageY, 0, 1);
		cvDilate(imageG, imageG, 0, 1);
		cvErode(imageG, imageG, 0, 1);
		cvReleaseImage(&Imagerect);
}

void CsignDcl::Image2RBYImageforLight(IplImage*image,IplImage*imageR,IplImage*imageY,IplImage*imageG)
{
	int height=image->height;
	int width=image->width;
	int step=image->widthStep;
	int step1 = imageR->widthStep;
	int channels=image->nChannels;
	uchar *data1=(uchar*)image->imageData;
	uchar *data_R=(uchar*)imageR->imageData;
	uchar *data_Y=(uchar*)imageY->imageData;
	uchar *data_G=(uchar*)imageG->imageData;

	int i,j;
	double r,g,b,s,h,v,H;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (((H>0&&H<10) || (H>320&&H<360))&&s<0.2&&v>0.9)////////红灯
			{
				data_R[i*step1+j] = 255;
			}
			else if (H>20&&H<50&&s>0.4&&v>0.7)//Y 黄灯
			{
				data_Y[i*step1+j] = 255;
			}
			else if (H>100&&H<180&&s<0.2&&v>0.9)//绿灯
			{
				data_G[i*step1+j] = 255;
			}

		}
	cvDilate(imageR, imageR, 0, 2);
	cvErode(imageR, imageR, 0, 1);
	cvDilate(imageY, imageY, 0, 2);
	cvErode(imageY, imageY, 0, 1);
	cvDilate(imageG, imageG, 0, 2);
	cvErode(imageG, imageG, 0, 1);
}

void CsignDcl::TrafficSignDwithColor(IplImage*image,IplImage*imagegray,IplImage* imagedetect,CvHaarClassifierCascade*cascade[],int n, vector<int> *signs)
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	int height=image->height;
	int width=image->width;
	int classID = -1;
	cvFindContours(imagegray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
	for( ; contour != 0; contour = contour->h_next )
	{
		CvRect r = ((CvContour*)contour)->rect;
		if (r.height>20&&r.width>20)
		{
			///////////////////////////////////////////RGYLIGHT///////////////////////////////////////////
			////////////////////////////////确定检测框上下左右都扩大20/////////////////////////////////
			int r_w=0;
			(r.x>20)?(r.x-=20):(r.x=0);
			(r.y>20)?(r.y-=20):(r.y=0);
			(r.width>r.height)?(r_w=r.width):(r_w=r.height);
			(r.x+r_w+40<width)?(r.width=r_w+40):(r.width=width-r.x);
			(r.y+r_w+40<height)?(r.height=r_w+40):(r.height=height-r.y);

			IplImage*detectImage = cvCreateImage(cvSize(r.width,r.height),8,imagedetect->nChannels);
			cvSetImageROI(imagedetect,r);
			cvCopy(imagedetect,detectImage);
			cvResetImageROI(imagedetect);

			for (int j=0;j<n;j++)
			{
				CvMemStorage* storage1 = cvCreateMemStorage(0);
				CvSeq* signt = cvHaarDetectObjects( detectImage, cascade[j], storage1, 1.1, 1,CV_HAAR_DO_CANNY_PRUNING,cvSize(20,20) );
				/////////////////////选取其中最大块进行识别/////////////////////////////
				CvRect rect_max = cvRect(0,0,0,0);
				for(int i = 0; i < signt->total; i++ )
				{
					CvRect Sign_rect = *(CvRect*)cvGetSeqElem(signt, i );
					//if(rect_max.width<Sign_rect.width) 
						rect_max = Sign_rect;//////////////////////////////////////
				}
				if(rect_max.width!=0)
				{
					CvRect Recog_rect = cvRect(rect_max.x+r.x,rect_max.y+r.y,rect_max.width,rect_max.height);
					IplImage* recogimage= NULL;
					IplImage* recogimageRe= NULL;
					IplImage* recogimageReG= NULL;
					Mat mtx;
					switch (n)
					{
					case 5:
						recogimage = cvCreateImage(cvSize(Recog_rect.width,Recog_rect.height),8,3);
						recogimageRe = cvCreateImage(cvSize(40,40),8,3);
						recogimageReG = cvCreateImage(cvSize(40,40),8,1);
						cvSetImageROI(image,Recog_rect);
						cvCopy(image,recogimage);
						cvResetImageROI(image);
						cvResize(recogimage,recogimageRe);
						cvCvtColor(recogimageRe,recogimageReG,CV_BGR2GRAY);
						classID = SingleRectRecog(recogimageReG);
					    //mtx=Mat(recogimageReG);
					   // classID = model->predict(mtx);
						//classID = 1;
						cvReleaseImage(&recogimage);
						if (classID!=1&&classID!=2&&classID!=3&&classID!=4&&classID!=5&&classID!=6&&classID!=7&&classID!=8&&classID!=9&&classID!=10&&classID!=25&&classID!=26)
						{
							classID = -1;
						}
						else
						{
							//int* nTemp = new int;
							//nTemp = &classID;
							signs->push_back(classID);
						}

						break;
					case 4:
						recogimage = cvCreateImage(cvSize(Recog_rect.width,Recog_rect.height),8,3);
						recogimageRe = cvCreateImage(cvSize(40,40),8,3);
						recogimageReG = cvCreateImage(cvSize(40,40),8,1);
						cvSetImageROI(image,Recog_rect);
						cvCopy(image,recogimage);
						cvResetImageROI(image);
						//Mat mtx(recogimage);
						cvResize(recogimage, recogimageRe);
						cvCvtColor(recogimageRe,recogimageReG,CV_BGR2GRAY);
						classID = SingleRectRecog(recogimageReG);
					  // mtx=Mat(recogimageReG);
					   //classID = model->predict(mtx);

						//classID = 1;
						cvReleaseImage(&recogimage);
						if (classID!=11&&classID!=12&&classID!=13&&classID!=14&&classID!=15&&classID!=16&&classID!=17&&classID!=18&&classID!=19&&classID!=20&&classID!=21&&classID!=22&&classID!=23)
						{
							classID = -1;
						}
							else
						{
							//int* nTemp = new int;
							//nTemp = &classID;
							signs->push_back(classID);
						}

						break;
					case 2:
						recogimage = cvCreateImage(cvSize(Recog_rect.width,Recog_rect.height),8,3);
						recogimageRe = cvCreateImage(cvSize(40,40),8,3);
						recogimageReG = cvCreateImage(cvSize(40,40),8,1);
						cvSetImageROI(image,Recog_rect);
						cvCopy(image,recogimage);
						cvResetImageROI(image);
						cvResize(recogimage, recogimageRe);
						cvCvtColor(recogimageRe,recogimageReG,CV_BGR2GRAY);
					   // mtx=Mat(recogimageReG);
					   //classID = model->predict(mtx);
						classID = SingleRectRecog(recogimageReG);
						cvReleaseImage(&recogimage);
						if (classID!=22&&classID!=23&&classID!=26&&classID!=27&&classID!=29)
						{
							classID = -1;
						}
							else
						{
							//int* nTemp = new int;
							//nTemp = &classID;
							signs->push_back(classID);
						}

						break;
					case 1:
						recogimage = cvCreateImage(cvSize(Recog_rect.width,Recog_rect.height),8,3);
						cvSetImageROI(image,Recog_rect);
						cvCopy(image,recogimage);
						cvResetImageROI(image);
					   mtx=Mat(recogimage);
					    classID = model->predict(mtx);
						cvReleaseImage(&recogimage);
						if (classID!=24&&classID!=30)
						{
							classID = -1;
						}
						break;
					}
					if (classID!=-1)
					{
						cvRectangle( image, cvPoint(rect_max.x+r.x,rect_max.y+r.y),cvPoint(rect_max.x+rect_max.width+r.x,rect_max.y+rect_max.height+r.y),CV_RGB(0,255,255), 3 );
						RecogResult[classID] = TRUE;
					}
					///////////////////////////////////////
				}
				cvReleaseMemStorage(&storage1);
			// }
			}
			cvReleaseImage(&detectImage);
		}
	}
	cvReleaseMemStorage(&storage);
}



void CsignDcl::LoadRecogPara()
{
	////////////////////////////////////识别检测器载入////////////////////
	cascadeR[0] = CsignDcl::load_object_detector(".\\cascade\\cascade_RedCircle_RC.xml");
	cascadeR[1] = CsignDcl::load_object_detector(".\\cascade\\cascade_NoEntry_RC.xml");
	cascadeR[2] = CsignDcl::load_object_detector(".\\cascade\\cascade_rang2.xml");
	cascadeR[3] = CsignDcl::load_object_detector(".\\cascade\\cascade_Stop_RC.xml");
	cascadeR[4] = CsignDcl::load_object_detector(".\\cascade\\cascade_Zui2.xml");

	cascadeB[0] = CsignDcl::load_object_detector(".\\cascade\\cascade_BlueCircle_BC.xml");
	cascadeB[1] = CsignDcl::load_object_detector(".\\cascade\\cascade_BlueRect_BC.xml");
	cascadeB[2] = CsignDcl::load_object_detector(".\\cascade\\YellowTriangle.xml");
	cascadeB[3] = CsignDcl::load_object_detector(".\\cascade\\cascade_EPS.xml");

	cascadeY[0] = CsignDcl::load_object_detector(".\\cascade\\cascade_YTRI915.xml");
	cascadeY[1] = CsignDcl::load_object_detector(".\\cascade\\cascade_LRT.xml");

	cascadeG = CsignDcl::load_object_detector(".\\cascade\\cascade_LRT.xml");

	model->load(".\\cascade\\trafficsign-train100.yml");
	///////////////////用于识别的变量载入////////////////////////////////
	 size_recog.height=40;
     size_recog.width=40;

	  if (! p_recog.Init(size_recog, true)) return;
	  f_recog.InitGaborKernel(5, 8);
	 featureSz = f_recog.InitSamplePoints(p_recog.m_mask,size_recog,6);
	//	CvFileStorage * fileStorage;
	//fileStorage = cvOpenFileStorage( ".\\cascade\\TSRdata_40_UP.xml", 0, CV_STORAGE_READ );
	//if( !fileStorage )
	//{
	//	fprintf(stderr, "Can't open facedata.xml\n");
	//	return;
	//}////////////////////////读入已经训练完的用于分类的文件

	//size_recog.height = cvReadIntByName(fileStorage, 0, "sizeHeight", 0);
	//size_recog.width = cvReadIntByName(fileStorage, 0, "sizeWidth", 0);
	//nTsign  = cvReadIntByName(fileStorage, 0, "nTsign", 0);
	//mdSz  = cvReadIntByName(fileStorage, 0, "mdSz", 0);

	//tsignNumTruthMat = (CvMat *)cvReadByName(fileStorage, 0, "tsignNumTruthMat", 0);
	//W_pcafldT = (CvMat *)cvReadByName(fileStorage, 0, "W_pcafldT", 0);
	//mu_total = (CvMat *)cvReadByName(fileStorage, 0, "mu_total", 0);
	//modellda = new CvMat* [nTsign];

	//ss_recog.W_pcafldT = W_pcafldT;
	//ss_recog.mu_total = mu_total;

	//if (! p_recog.Init(size_recog, true)) return;
	//f_recog.InitGaborKernel(5, 8);
	//featureSz = f_recog.InitSamplePoints(p_recog.m_mask,size_recog,6);

	//for(int i=0; i<nTsign; i++)
	//{
	//	char varname[200];
	//	sprintf( varname, "model_%d", i );
	//	modellda[i] = (CvMat *)cvReadByName(fileStorage, 0, varname, 0);
	//}
	//// TODO: Add your control notification handler code here
	//cvReleaseFileStorage( &fileStorage );
}


void CsignDcl::TrafficSignRecognition(IplImage*image, vector<int> *signs)
{
	for (int i=0;i<32;i++)
	{
		RecogResult[i] = FALSE;
	}
	IplImage*imageR = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageB = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageY = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageG = cvCreateImage(cvGetSize(image),8,1);
	IplImage*rImg = cvCreateImage(cvGetSize(image),8,1);
	IplImage*gImg = cvCreateImage(cvGetSize(image),8,1);
	IplImage*bImg = cvCreateImage(cvGetSize(image),8,1);
	cvCvtPixToPlane(image,rImg,gImg,bImg,NULL);
	Image2RBYImage(image,imageR,imageB,imageY,imageG);
	vector<int> *RedSigns = new vector<int> ();
	vector<int> *BlueSigns = new vector<int> ();
	vector<int> *YellowSigns = new vector<int> ();
	TrafficSignDwithColor(image,imageR,rImg,cascadeR,5,RedSigns);
	TrafficSignDwithColor(image,imageB,bImg,cascadeB,4,BlueSigns);
	TrafficSignDwithColor(image,imageY,image,cascadeY,2,YellowSigns);
	signs->insert(signs->begin(),RedSigns->begin(),RedSigns->end());
	signs->insert(signs->begin(),BlueSigns->begin(),BlueSigns->end());
	signs->insert(signs->begin(),YellowSigns->begin(),YellowSigns->end());
	///////////////////////////
	cvReleaseImage(&imageR);
	cvReleaseImage(&imageB);
	cvReleaseImage(&imageY);
	cvReleaseImage(&imageG);
	cvReleaseImage(&rImg);
	cvReleaseImage(&bImg);
	cvReleaseImage(&gImg);
	delete RedSigns;
	delete BlueSigns;
	delete YellowSigns;
}

bool CsignDcl::TrafficLightDwithRect(IplImage*image,CvRect rect,int n)
{
	double rat= 0.0;
	double rat1 = 0.0;
	double rat2= 0.0;
	double rat3= 0.0;
	int templ = 0
	(rect.width>rect.height)?(templ = rect.width):(templ=rect.height);
	rect.width = templ;
	rect.height = templ;
	cvRectangle(image,cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width,rect.y+rect.height),CV_RGB(255,0,0));
	switch (n)
	{
	case 1://red light
		if(rect.y+rect.height*3>image->height) return FALSE;
		CvRect rect_all;
		rect_all.x = rect.x;
		rect_all.y = rect.y+rect.height;
		rect_all.width = rect.width;
		rect_all.height = rect.height*2;
		rat = RectBlackRate(image,rect_all);
		printf("rat is %f\n",rat);
		if(rat>0.7) return TRUE;
		else return FALSE;
		break;
	case 2://yellow light
		if(rect.y-rect.height<0||rect.y+rect.height*2>image->height) return FALSE;
		CvRect r1,r2;
		r1.x=rect.x;
		r1.y=rect.y-rect.height;
		r1.width = rect.width;
		r1.height = rect.height;
		r2.x = rect.x;
		r2.y = rect.y+rect.height;
		r2.width = rect.width;
		r2.height = rect.height;
		rat1 = RectBlackRate(image,r1);
		rat2 = RectBlackRate(image,r2);
		printf("rat1,rat2 is %f,%f\n",rat1,rat2);
		if (rat1>0.7&&rat2>0.7) return TRUE;
		else return FALSE;
		break;
	case 3://green light
		if (rect.y-rect.height*2<0) return FALSE;
		CvRect rect_all2;
		rect_all2.x = rect.x;
		rect_all2.y =rect.y-rect.height*2;
		rect_all2.width = rect.width;
		rect_all2.height = rect.height*2;
		rat3 = RectBlackRate(image,rect_all2);
		printf("rat3 is %f\n",rat3);
		if (rat3>0.7) return TRUE;
		else return FALSE;
		break;
	default:
		return FALSE;
		break;

	}
}

double CsignDcl::RectBlackRate(IplImage*image,CvRect rect)
{
	IplImage*tempImage = cvCreateImage(cvSize(rect.width,rect.height),8,3);
	IplImage*tempGray = cvCreateImage(cvSize(rect.width,rect.height),8,1);
	cvSetImageROI(image,rect);
	cvCopy(image,tempImage);
	cvResetImageROI(image);
	cvCvtColor(tempImage,tempGray,CV_RGB2GRAY);
	cvSaveImage("gra.jpg",tempGray);
	uchar*data = (uchar*)tempGray->imageData;
	int blackn = 0;
	for (int i=0;i<tempGray->height;i++)
		for (int j=0;j<tempGray->width;j++)
		{
			if(data[i*tempGray->widthStep+j]<60) blackn++;
		}
		double rat = (double)blackn/(tempGray->width*tempGray->height);
		cvReleaseImage(&tempImage);
		cvReleaseImage(&tempGray);
		return rat;
}

void CsignDcl::TrafficLightRwithImg(IplImage*image,IplImage*imagegray,int n/*1red2yellow3green*/)
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	int height=image->height;
	int width=image->width;
	int classID = -1;
	cvFindContours(imagegray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
	for( ; contour != 0; contour = contour->h_next )
	{
		CvRect r = ((CvContour*)contour)->rect;
		if (r.height>10&&r.width>10)
		{
			classID = -1;
			///////////////////////////////////////////RGYLIGHT///////////////////////////////////////////
			////////////////////////////////确定检测框上下左右都扩大20/////////////////////////////////
			bool tlight = FALSE;
			if(abs(r.height - r.width)<10)
			{
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(0,255,0));
				switch(n)
				{
					case 1:
						tlight = TrafficLightDwithRect(image,r,1);
						if(tlight) classID = 28;
						break;
					case 2:
						tlight = TrafficLightDwithRect(image,r,2);	
						if(tlight) classID = 29;
						break;
					case 3:
						tlight = TrafficLightDwithRect(image,r,3);
						if(tlight) classID = 30;
						break;
					default:
						break;
				}
			}
			if(classID != -1)
			{
				RecogResult[classID] = TRUE;
				printf("light is %d\n",classID);
			}

		}

	}
	
	cvReleaseMemStorage(&storage);	
}


void CsignDcl::TrafficLightRecognition(IplImage* image)
{
	for (int i=0;i<32;i++)
	{
		RecogResult[i] = FALSE;
	}

	IplImage*imageR = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageY = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageG = cvCreateImage(cvGetSize(image),8,1);
	cvZero(imageR);
	cvZero(imageY);
	cvZero(imageG);
	CsignDcl::Image2RBYImageforLight(image,imageR,imageY,imageG);
	CsignDcl::TrafficLightRwithImg(image,imageR,1);
	CsignDcl::TrafficLightRwithImg(image,imageY,2);
	CsignDcl::TrafficLightRwithImg(image,imageG,3);
	cvReleaseImage(&imageR);
	cvReleaseImage(&imageY);
	cvReleaseImage(&imageG);
	
}

int CsignDcl::SingleRectRecog(IplImage*img)
{     
		    CvMat *tempM1 = cvCreateMat(size_recog.height,size_recog.width,CV_32FC1);
	        cvConvertScale(img,tempM1,1.0/255);
	        p_recog.HomographicFilter(tempM1);
	        CvMat*tfeatureL = cvCreateMat(featureSz, 1, CV_FC1);
		     f_recog.GaborConvAndSample(tempM1,tfeatureL);
	          Mat im(tfeatureL);
			 //Mat img = imread(w, CV_LOAD_IMAGE_GRAYSCALE);
			 int label = model->predict(im);
	         return label;
			 cvReleaseMat(&tempM1);
}

