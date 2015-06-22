// LaneDCL.cpp: implementation of the LaneDCL class.
//
//////////////////////////////////////////////////////////////////////

#include "LaneDCL.h"
#include "iostream"
#include <cmath>
#include "LightPrep.h"
#include "FaceFeature.h"
#include "Subspace.h"

using namespace std;

//////////////识别参数载入//////////////////////////////////////////////
//////////////用于对样本进行变换的参数，用到了这三个类//////////////
CLightPrep p_recog;
CFaceFeature f_recog;
CSubspace ss_recog;
CvSize size_recog; //识别分类器训练时的尺寸
int nTsign;//用于训练的样本个数
int mdSz;//最后采用的特征维数
CvMat*tsignNumTruthMat;//标志的类号，在训练时train.txt里写清楚。
CvMat*W_pcafldT;//特征向量矩阵，用于最后的变换
CvMat*mu_total;//均值
CvMat**model;//样本最后的模式
int featureSz;
///////////////////////////////检测分类器参数//////////////////////////
CvHaarClassifierCascade* cascadeR[5];
CvHaarClassifierCascade* cascadeB[4];
CvHaarClassifierCascade* cascadeY[2];
/////////////////////////////////////////////////////////////////////////
//static ClaneDCL::RecogResult[31] = {0};

TraSign* CreateTraSign()
{
	TraSign *trs = new TraSign;
	trs->storage = cvCreateMemStorage(0);
	trs->idstorage = cvCreateMemStorage(0);
	trs->seq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvRect),trs->storage);
	trs->id = cvCreateSeq(0,sizeof(CvSeq),sizeof(int),trs->storage);
	return trs;
}
void ReleaseTraSign(TraSign *trs)
{
	if (trs->storage!=NULL)
	{
		cvReleaseMemStorage(&trs->storage);
	}
	if (trs->id!=NULL)
	{
		cvReleaseMemStorage(&trs->idstorage);
	}
	delete trs;
}
ClaneDCL::ClaneDCL()
{

}

ClaneDCL::~ClaneDCL()
{

}

BOOL ClaneDCL::RecogResult[32] = {FALSE};

int ClaneDCL::RGB2HSI(int R,int G,int B,double& H,double& S,double& I)
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
//色彩信息统计,输入一个图像在HSV空间下的红，蓝，黄，黑，白色的比例
void ClaneDCL:: color_statistics(IplImage *image, double & redration, double & blueration, double & yellowration, double & blackration, double & whiteration)
{
	int height,width,step,channels;
	height=image->height;
	width=image->width;
	step=image->widthStep;
	channels=image->nChannels;
	uchar *data1=(uchar*)image->imageData;
	int i,j;
	double H;
	double s,h,v;
	double r,g,b;
	int nred,nblue,nyellow,nblack,nwhite;
	nred=0;nblue=0;nyellow=0;nblack=0;nwhite=0;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if ((H>0&&H<20) || (H>320&&H<360)&&s>0.2)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
				nred++;
			else if(H>200&&H<270&&s>0.3)//blue
				nblue++;
			else if(H>25&&H<60&&s>0.5)//yellow
				nyellow++;
			else if (s<0.3&&v<0.4) //black
				nblack++;
			else if (s<0.3&&v>0.4) //white
				nwhite++;
		}
	redration = (double)nred/(height*width);
	blueration = (double)nblue/(height*width);
	yellowration = (double)nyellow/(height*width);
	blackration = (double)nblack/(height*width);
	whiteration = (double)nwhite/(height*width);
}
//以图像中心为圆心的圆上像素点统计
void ClaneDCL::color_circle_points(int xc,int yc,int x,int y,IplImage*image,int&nred,int&nblue,int&ntotal)
{
	int height,width,step,channels;
	height=image->height;
	width=image->width;
	step=image->widthStep;
	channels=image->nChannels;
	IplImage*img = cvCreateImage(cvSize(width,height),8,3);
	cvCopy(image,img);
	double H;
	double s,h,v;
	double r,g,b;
	uchar *data=(uchar*)img->imageData;
	for (int n=1;n<9;n++)
	{
		int i,j;
		switch (n)
		{
			case 1:
				i=xc+x;
				j=yc+y;
				break;

			case 2:
				i=xc-x;
				j=yc+y;
				break;
			case 3:
				i=xc+x;
				j=yc-y;
				break;
			case 4:
				i=xc-x;
				j=yc-y;
				break;
			case 5:
				i=xc+y;
				j=yc+x;
				break;
			case 6:
				i=xc-y;
				j=yc+x;
				break;
			case 7:
				i=xc+y;
				j=yc-x;
				break;
			case 8:
				i=xc-y;
				j=yc-x;
				break;
		}
		b=(double)data[i*step+j*channels+0];
		g=(double)data[i*step+j*channels+1];
		r=(double)data[i*step+j*channels+2];
		RGB2HSI((int)r,(int)g,(int)b,h,s,v);
		H=h*360;
		ntotal++;
		if ((H>0&&H<20) || (H>320&&H<360)&&s>0.01)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
		{
			//data[i*step+j*channels+0]=255;
			//data[i*step+j*channels+1]=255;
			//data[i*step+j*channels+2]=255;
			nred++;
		}
		else if(H>200&&H<270&&s>0.3)//blue
		{
			//data[i*step+j*channels+0]=0;
			//data[i*step+j*channels+1]=0;
			//data[i*step+j*channels+2]=0;
			nblue++;
		}
	}
	cvReleaseImage(&img);
}
//得到圆上点，xc,yc为圆心的横纵坐标，radius为半径，nred,nblue,ntotal 为图像红色 蓝色 的数量
void ClaneDCL::get_circle(int xc,int yc,int radius,IplImage* image,int&nred,int&nblue,int&ntotal)
{
	int x,y,p;
	x=0;
	y=radius;
	p=3-2*radius;
	while(x<y)
	{
		color_circle_points(xc,yc,x,y,image,nred,nblue,ntotal);
		if(p<0)
			p=p+4*x+6;
		else
		{
			p=p+4*(x-y)+10;
			y-=1;
		}
		x+=1;
	}
	if(x==y)
		color_circle_points(xc,yc,x,y,image,nred,nblue,ntotal);
}
//根据圆周上色彩点的个数做判断，ncolor指用不同的分类器识别要求的结果不一样
int ClaneDCL::circlehist(IplImage*image,int ncolor)
{
	int height,width,step,channels;
 	height=image->height;
	width=image->width;
	step=image->widthStep;
	channels=image->nChannels;
	CvPoint pC;
	pC.x = height/2;
	pC.y = width/2;
	double red[10];
	double blue[10];
	for (int n=0;n<8;n++)
	{
		int nr=0;
		int nb=0;
		int ntotal=0;
		int ra = height*(n+1)/16;
		get_circle(pC.x,pC.y,ra,image,nr,nb,ntotal);
		double rred = double(nr)/ntotal;
		double rblue = double(nb)/ntotal;
		red[n] = rred;
		blue[n] = rblue;
	//	printf("%f\n",red[n]);
	}
	
	switch (ncolor)
	{
	case 0:
		if (red[5]>0.7||red[6]>0.7||red[7]>0.7)
			return 1;
		break;
	case 1:
		if (blue[3]>0.8||blue[3]>0.8||blue[3]>0.8)
			return 1;
		break;
	case 3:
		if (blue[2]>red[2]&&blue[3]>red[3]&&blue[4]>red[4]&&blue[5]>red[5])
			return 1;
		break;				
	}
	return 0;
}
//分类器载入
CvHaarClassifierCascade* ClaneDCL:: load_object_detector( const char* cascade_path )
{
    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}

//小块区域判断
int ClaneDCL::rect_decide(IplImage*image, CvRect rect,int n_color)
{
	double redration, blueration, yellowration,blackration, whiteration;
	//////////////////////////////////////////////////////////////////////////
	IplImage *rect_image = cvCreateImage(cvSize(rect.width,rect.height),8,3);
	
	cvSetImageROI(image,rect);
	cvCopy(image,rect_image);
	cvResetImageROI(image);
	color_statistics(rect_image,redration,blueration,yellowration,blackration,whiteration);

	switch (n_color)
	{
		case 0:
			if(redration>0.1&&yellowration<0.1&&redration<0.7&&whiteration>0.1)
				return 1;
			else
 				return 0;
			break;
		case 1:
			if(blueration>0.2)
				return 1;
			else
 				return 0;
			break;
		case 2:
			if((blueration>0.1||yellowration>0.1)&&blackration>0.01)
				return 1;
			else
 				return 0;
			break;
		case 3:
			if(redration>0.2&&redration>whiteration&&redration<0.9)
				return 1;
			else
 				return 0;
			break;
		case 4:
			if(redration>0.05&&blackration>0.01&&whiteration>0.1)
				return 1;
			else
 				return 0;
			break;
		case 5:
			if(redration>0.2&&whiteration>0.2)
				return 1;
			else
 				return 0;
			break;
		case 6:
			if(blueration>0.2&&whiteration>0.1)
				return 1;
			else
				return 0;
			break;
	}
	cvReleaseImage(&rect_image);	
}	
//分类器检测和现实
CvSeq* ClaneDCL::detect_and_draw_objects( IplImage* image, CvHaarClassifierCascade* cascade,int do_pyramids, int n_color )
{
    IplImage* small_image = image;
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* faces;
	CvMemStorage* storage1 = cvCreateMemStorage(0);
    CvSeq* faces1;
	faces1 = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvRect), storage1 );
    int i, scale = 1;
	faces = cvHaarDetectObjects( image, cascade, storage, 1.2, 2,CV_HAAR_DO_CANNY_PRUNING,cvSize(20,20) );
    /* draw all the rectangles */
    for( i = 0; i < faces->total; i++ )
    {
        /* extract the rectanlges only */
        CvRect face_rect = *(CvRect*)cvGetSeqElem( faces, i );
        if (rect_decide(image,face_rect,n_color))
			cvSeqPush(faces1,&face_rect);	
    }
    if( small_image != image )
        cvReleaseImage( &small_image );

	cvReleaseMemStorage( &storage );
	return faces1;
    
}

//extern CvHaarClassifierCascade* cascade[7];
//交通标志检测
TraSign* ClaneDCL::TrafficSignDetection(IplImage*image,CvHaarClassifierCascade*cascade[],int n)
{
	TraSign* trs = CreateTraSign();

	for (int j=0;j<n;j++)
	{
		CvSeq* faces;
		faces = detect_and_draw_objects( image, cascade[j], 1 ,j);
		for(int i = 0; i < faces->total; i++ )
		{
			CvRect face_rect = *(CvRect*)cvGetSeqElem(faces, i );
			cvSeqPush(trs->seq,&face_rect);	
			cvSeqPush(trs->id,&j);
		}
		cvReleaseMemStorage(&faces->storage);
	}
	
	for( int i = 0; i < trs->seq->total-1; i++ )
	{
		CvRect face_rect1 = *(CvRect*)cvGetSeqElem( trs->seq, i );
		for(int j=1;j<trs->seq->total-i;j++)
		{
			CvRect face_rect2 = *(CvRect*)cvGetSeqElem( trs->seq, i+j );
// 			double dis = Distance2Rect(face_rect1,face_rect2);	
// 			if (dis<20||dis<face_rect1.height/2||dis<face_rect2.height/2)
// 			{
// 				cvSeqRemove(trs->seq,j+i);
// 				cvSeqRemove(trs->id,j+i);
// 				j = j-1;
// 			}
			double dis = CompareRect(face_rect1,face_rect2);	
			if (dis>10)
			{
				cvSeqRemove(trs->seq,j+i);
				cvSeqRemove(trs->id,j+i);
				j = j-1;
			}
			
		}		
	}
	return trs;
}



int ClaneDCL::CompareRect(CvRect rect1, CvRect rect2)
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

double ClaneDCL::Distance2Rect(CvRect rect1,CvRect rect2)
{
	
	int x1 = rect1.x+rect1.width/2;
	int y1 = rect1.y+rect1.height/2;
	int x2 = rect2.x+rect2.width/2;
	int y2 = rect2.y+rect2.height/2;
	double d = sqrt((double)(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
	return d;
}

CvRect ClaneDCL::GetRedRect(IplImage*image)
{
	IplImage *Imagerect = cvCreateImage(cvGetSize(image),8,3);
	IplImage*gray = cvCreateImage(cvGetSize(Imagerect),8,1);
	cvSmooth(image,Imagerect,CV_GAUSSIAN,3,3);	//高斯模糊
	cvZero(gray);
	int height=Imagerect->height;
	int width=Imagerect->width;
	int step=Imagerect->widthStep;
	int step1 = gray->widthStep;
	int channels=Imagerect->nChannels;
	uchar *data1=(uchar*)Imagerect->imageData;
	uchar *data=(uchar*)gray->imageData;
	int i,j;
	double H;
	double s,h,v;
	double r,g,b;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (((H>0&&H<10) || (H>320&&H<360))&&s>0.4)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data[i*step1+j] = 255;
			}
		}
		cvDilate(gray, gray, 0, 3);
		cvErode(gray, gray, 0, 1);
		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* contour = 0;
		cvFindContours(gray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
				
		int rect_tx=1000000;
		int rect_ty=1000000;
		int rect_bx=-1;
		int rect_by=-1;
		int totalnum = 0;
		for( ; contour != 0; contour = contour->h_next )
		{
			
			/* 用1替代 CV_FILLED  所指示的轮廓外形 */
			CvRect r = ((CvContour*)contour)->rect;
			if (r.height>20&&r.width>20)
			{
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(255,0,0),1,CV_AA,0);
				if(r.x<rect_tx) rect_tx = r.x;
				if(r.y<rect_ty) rect_ty = r.y;
				if(r.x+r.width>rect_bx) rect_bx = r.x+r.width;
				if(r.y+r.height>rect_by) rect_by = r.y+r.height;
				totalnum++;
			}
		}

		if(totalnum==0) return cvRect(0,0,0,0);
		if(rect_tx-20>0) rect_tx -=20;
		else rect_tx=0;
		if (rect_ty-20>0) rect_ty -=20;
		else rect_ty=0;
		if (rect_bx+20<width) rect_bx+=20;
		else rect_bx=width;
		if (rect_by+20<height) rect_by+=20;
		else rect_by=height;

		cvRectangle(image,cvPoint(rect_tx,rect_ty),cvPoint(rect_bx,rect_by),CV_RGB(255,0,0),2,CV_AA,0);
		cvNamedWindow("R");
		cvShowImage("R",image);
		//cvWaitKey();

		
		cvReleaseImage(&Imagerect);
		cvReleaseImage(&gray);
		cvReleaseMemStorage(&storage);
		return cvRect(rect_tx,rect_ty,rect_bx-rect_tx,rect_by-rect_ty);

}

CvRect ClaneDCL::GetBlueRect(IplImage*image)
{
	IplImage *Imagerect = cvCreateImage(cvGetSize(image),8,3);
	IplImage*gray = cvCreateImage(cvGetSize(Imagerect),8,1);
	cvSmooth(image,Imagerect,CV_GAUSSIAN,3,3);	//高斯模糊
	cvZero(gray);
	int height=Imagerect->height;
	int width=Imagerect->width;
	int step=Imagerect->widthStep;
	int step1 = gray->widthStep;
	int channels=Imagerect->nChannels;
	uchar *data1=(uchar*)Imagerect->imageData;
	uchar *data=(uchar*)gray->imageData;
	int i,j;
	double H;
	double s,h,v;
	double r,g,b;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (H>200&&H<270&&s>0.5)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data[i*step1+j] = 255;
			}
		}

		cvDilate(gray, gray, 0, 3);
		cvErode(gray, gray, 0, 1);

		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* contour = 0;
		cvFindContours(gray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

		int rect_tx=1000000;
		int rect_ty=1000000;
		int rect_bx=-1;
		int rect_by=-1;
		int totalnum = 0;
		for( ; contour != 0; contour = contour->h_next )
		{
		
			/* 用1替代 CV_FILLED  所指示的轮廓外形 */
			CvRect r = ((CvContour*)contour)->rect;
			if (r.height>20&&r.width>20)
			{
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(0,0,255),1,CV_AA,0);
				if(r.x<rect_tx) rect_tx = r.x;
				if(r.y<rect_ty) rect_ty = r.y;
				if(r.x+r.width>rect_bx) rect_bx = r.x+r.width;
				if(r.y+r.height>rect_by) rect_by = r.y+r.height;
				totalnum++;
			}
		}
		if(totalnum == 0) return cvRect(0,0,0,0);
		if(rect_tx-20>0) rect_tx -=20;
		else rect_tx=0;
		if (rect_ty-20>0) rect_ty -=20;
		else rect_ty=0;
		if (rect_bx+20<width) rect_bx+=20;
		else rect_bx=width;
		if (rect_by+20<height) rect_by+=20;
		else rect_by=height;

		cvRectangle(image,cvPoint(rect_tx,rect_ty),cvPoint(rect_bx,rect_by),CV_RGB(0,0,255),2,CV_AA,0);
		cvNamedWindow("B");
		cvShowImage("B",image);
		//cvWaitKey();


		cvReleaseImage(&Imagerect);
		cvReleaseImage(&gray);
		cvReleaseMemStorage(&storage);
		return cvRect(rect_tx,rect_ty,rect_bx-rect_tx,rect_by-rect_ty);

}

CvRect ClaneDCL::GetYellowRect(IplImage*image)
{
	IplImage *Imagerect = cvCreateImage(cvGetSize(image),8,3);
	IplImage*gray = cvCreateImage(cvGetSize(Imagerect),8,1);
	cvSmooth(image,Imagerect,CV_GAUSSIAN,3,3);	//高斯模糊
	cvZero(gray);
	int height=Imagerect->height;
	int width=Imagerect->width;
	int step=Imagerect->widthStep;
	int step1 = gray->widthStep;
	int channels=Imagerect->nChannels;
	uchar *data1=(uchar*)Imagerect->imageData;
	uchar *data=(uchar*)gray->imageData;
	int i,j;
	double H;
	double s,h,v;
	double r,g,b;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (H>30&&H<60&&s>0.4&&v>0.5)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data[i*step1+j] = 255;
			}
		}

		cvDilate(gray, gray, 0, 3);
		cvErode(gray, gray, 0, 1);

		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* contour = 0;
		cvFindContours(gray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
	
		int rect_tx=1000000;
		int rect_ty=1000000;
		int rect_bx=-1;
		int rect_by=-1;
		int totalnum =0;
		for( ; contour != 0; contour = contour->h_next )
		{
			/* 用1替代 CV_FILLED  所指示的轮廓外形 */
			CvRect r = ((CvContour*)contour)->rect;
			if (r.height>20&&r.width>20)
			{
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(255,255,0),1,CV_AA,0);
				if(r.x<rect_tx) rect_tx = r.x;
				if(r.y<rect_ty) rect_ty = r.y;
				if(r.x+r.width>rect_bx) rect_bx = r.x+r.width;
				if(r.y+r.height>rect_by) rect_by = r.y+r.height;
				totalnum++;
			}
		}
		if(totalnum == 0) return cvRect(0,0,0,0);

		if(rect_tx-20>0) rect_tx -=20;
		else rect_tx=0;
		if (rect_ty-20>0) rect_ty -=20;
		else rect_ty=0;
		if (rect_bx+20<width) rect_bx+=20;
		else rect_bx=width;
		if (rect_by+20<height) rect_by+=20;
		else rect_by=height;

		cvRectangle(image,cvPoint(rect_tx,rect_ty),cvPoint(rect_bx,rect_by),CV_RGB(255,255,0),2,CV_AA,0);
		cvNamedWindow("Y");
		cvShowImage("Y",image);
		//cvWaitKey();


		cvReleaseImage(&Imagerect);
		cvReleaseImage(&gray);
		cvReleaseMemStorage(&storage);
		return cvRect(rect_tx,rect_ty,rect_bx-rect_tx,rect_by-rect_ty);

}

CvSeq*  ClaneDCL::detect_withRedRect(IplImage*image,CvRect rect,CvHaarClassifierCascade*cascade[],int n)
{
	IplImage*imageRect = cvCreateImage(cvSize(rect.width,rect.height),8,3);
	cvSetImageROI(image,rect);
	cvCopy(image,imageRect);
	cvResetImageROI(image);

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* Signs= cvCreateSeq(0,sizeof(CvSeq),sizeof(CvRect),storage);
	int scale = 1;
	for (int j=0;j<n;j++)
			{
				CvSeq* signt;
				signt = detect_and_draw_objects2( imageRect, cascade[j], 1 ,j);
				for(int i = 0; i < signt->total; i++ )
				{
					CvRect Sign_rect = *(CvRect*)cvGetSeqElem(signt, i );
					cvRectangle( image, cvPoint(Sign_rect.x*scale+rect.x,Sign_rect.y*scale+rect.y),cvPoint((Sign_rect.x+Sign_rect.width+rect.x)*scale,(Sign_rect.y+Sign_rect.height+rect.y)*scale),CV_RGB(255,255,255), 3 );
					cvSeqPush(Signs,&Sign_rect);	
				}
				cvReleaseMemStorage(&signt->storage);
			}
	cvNamedWindow("circle");
	cvShowImage("circle",image);
	//cvWaitKey();
	return Signs;
}

CvSeq* ClaneDCL::detect_and_draw_objects2( IplImage* image, CvHaarClassifierCascade* cascade,int do_pyramids, int n_color )
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* faces;
	faces = cvHaarDetectObjects( image, cascade, storage, 1.2, 2,CV_HAAR_DO_CANNY_PRUNING,cvSize(24,24) );
	return faces;
}


void ClaneDCL::DetectRedRect(IplImage*image,CvHaarClassifierCascade*cascade[],int n)
{
	IplImage *Imagerect = cvCreateImage(cvGetSize(image),8,3);
	IplImage*gray = cvCreateImage(cvGetSize(Imagerect),8,1);
	cvSmooth(image,Imagerect,CV_GAUSSIAN,3,3);	//高斯模糊
	cvZero(gray);
	int height=Imagerect->height;
	int width=Imagerect->width;
	int step=Imagerect->widthStep;
	int step1 = gray->widthStep;
	int channels=Imagerect->nChannels;
	uchar *data1=(uchar*)Imagerect->imageData;
	uchar *data=(uchar*)gray->imageData;
	int i,j;
	double H;
	double s,h,v;
	double r,g,b;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (((H>0&&H<10) || (H>320&&H<360))&&s>0.4)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data[i*step1+j] = 255;
			}
		}
		cvDilate(gray, gray, 0, 3);
		cvErode(gray, gray, 0, 1);
		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* contour = 0;
		int total = cvFindContours(gray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
		//int w=0;
		for( ; contour != 0; contour = contour->h_next )
		{
			//w++;
			/* 用1替代 CV_FILLED  所指示的轮廓外形 */
			CvRect r = ((CvContour*)contour)->rect;
			if (r.height>20&&r.width>20)
			{
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(255,0,0),1,CV_AA,0);
				(r.x>20)?(r.x-=20):(r.x=0);
				(r.y>20)?(r.y-=20):(r.y=0);
				(r.x+r.width+40<width)?(r.width+=40):(r.width=width-r.x);
				(r.y+r.height+40<height)?(r.height+=40):(r.height=height-r.y);
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(255,0,0),1,CV_AA,0);
				IplImage*detectImage = cvCreateImage(cvSize(r.width,r.height),8,3);
				cvSetImageROI(image,r);
				cvCopy(image,detectImage);
				cvResetImageROI(image);
				
				//CvMemStorage* storage = cvCreateMemStorage(0);
				//CvSeq* Signs= cvCreateSeq(0,sizeof(CvSeq),sizeof(CvRect),storage);
				int scale = 1;
				for (int j=0;j<n;j++)
				{
					CvSeq* signt = detect_and_draw_objects2( detectImage, cascade[j], 1 ,j);
					CvMemStorage* storage = cvCreateMemStorage(0);
					//CvSeq* signt = cvHaarDetectObjects( detectImage, cascade[j], storage, 1.2, 2,CV_HAAR_DO_CANNY_PRUNING,cvSize(24,24) );
					for(int i = 0; i < signt->total; i++ )
					{
						CvRect Sign_rect = *(CvRect*)cvGetSeqElem(signt, i );
						cvRectangle( image, cvPoint(Sign_rect.x*scale+r.x,Sign_rect.y*scale+r.y),cvPoint((Sign_rect.x+Sign_rect.width+r.x)*scale,(Sign_rect.y+Sign_rect.height+r.y)*scale),CV_RGB(255,255,255), 3 );
						
					}
					cvReleaseMemStorage(&signt->storage);
				}
				cvReleaseImage(&detectImage);
			}
			
		}

		cvNamedWindow("circle");
		cvShowImage("circle",image);
		cvWaitKey();
}

void ClaneDCL::DetectBlueRect(IplImage*image,CvHaarClassifierCascade*cascade[],int n)
{
	IplImage *Imagerect = cvCreateImage(cvGetSize(image),8,3);
	IplImage*gray = cvCreateImage(cvGetSize(Imagerect),8,1);
	cvSmooth(image,Imagerect,CV_GAUSSIAN,3,3);	//高斯模糊
	cvZero(gray);
	int height=Imagerect->height;
	int width=Imagerect->width;
	int step=Imagerect->widthStep;
	int step1 = gray->widthStep;
	int channels=Imagerect->nChannels;
	uchar *data1=(uchar*)Imagerect->imageData;
	uchar *data=(uchar*)gray->imageData;
	int i,j;
	double H;
	double s,h,v;
	double r,g,b;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (H>200&&H<270&&s>0.5)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data[i*step1+j] = 255;
			}
		}
		cvDilate(gray, gray, 0, 3);
		cvErode(gray, gray, 0, 1);
		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* contour = 0;
		cvFindContours(gray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

		for( ; contour != 0; contour = contour->h_next )
		{

			/* 用1替代 CV_FILLED  所指示的轮廓外形 */
			CvRect r = ((CvContour*)contour)->rect;
			if (r.height>20&&r.width>20)
			{
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(255,0,0),1,CV_AA,0);
				(r.x>20)?(r.x-=20):(r.x=0);
				(r.y>20)?(r.y-=20):(r.y=0);
				(r.x+r.width+40<width)?(r.width+=40):(r.width=width-r.x);
				(r.y+r.height+40<height)?(r.height+=40):(r.height=height-r.y);
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(0,255,0),1,CV_AA,0);
				IplImage*detectImage = cvCreateImage(cvSize(r.width,r.height),8,3);
				cvSetImageROI(image,r);
				cvCopy(image,detectImage);
				cvResetImageROI(image);
				//CvMemStorage* storage = cvCreateMemStorage(0);
				//CvSeq* Signs= cvCreateSeq(0,sizeof(CvSeq),sizeof(CvRect),storage);
				int scale = 1;
				for (int j=0;j<n;j++)
				{
					CvSeq* signt;
					signt = detect_and_draw_objects2( detectImage, cascade[j], 1 ,j);
					for(int i = 0; i < signt->total; i++ )
					{
						CvRect Sign_rect = *(CvRect*)cvGetSeqElem(signt, i );
						cvRectangle( image, cvPoint(Sign_rect.x*scale+r.x,Sign_rect.y*scale+r.y),cvPoint((Sign_rect.x+Sign_rect.width+r.x)*scale,(Sign_rect.y+Sign_rect.height+r.y)*scale),CV_RGB(255,255,255), 3 );

					}
					cvReleaseMemStorage(&signt->storage);
				}
				cvReleaseImage(&detectImage);
			}
			

		}

		cvNamedWindow("circle");
		cvShowImage("circle",image);
		cvWaitKey();
}

void ClaneDCL::DetectYellowRect(IplImage*image,CvHaarClassifierCascade*cascade[],int n)
{
	IplImage *Imagerect = cvCreateImage(cvGetSize(image),8,3);
	IplImage*gray = cvCreateImage(cvGetSize(Imagerect),8,1);
	cvSmooth(image,Imagerect,CV_GAUSSIAN,3,3);	//高斯模糊
	cvZero(gray);
	int height=Imagerect->height;
	int width=Imagerect->width;
	int step=Imagerect->widthStep;
	int step1 = gray->widthStep;
	int channels=Imagerect->nChannels;
	uchar *data1=(uchar*)Imagerect->imageData;
	uchar *data=(uchar*)gray->imageData;
	int i,j;
	double H;
	double s,h,v;
	double r,g,b;
	for (i=0;i<height-1;i++)
		for(j=0;j<width-1;j++)
		{ 
			b=(double)data1[i*step+j*channels+0];
			g=(double)data1[i*step+j*channels+1];
			r=(double)data1[i*step+j*channels+2];
			RGB2HSI((int)r,(int)g,(int)b,h,s,v);
			H=h*360;
			if (H>30&&H<60&&s>0.4&&v>0.5)//RED (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data[i*step1+j] = 255;
			}
		}
		cvDilate(gray, gray, 0, 3);
		cvErode(gray, gray, 0, 1);
		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* contour = 0;
		cvFindContours(gray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

		for( ; contour != 0; contour = contour->h_next )
		{

			/* 用1替代 CV_FILLED  所指示的轮廓外形 */
			CvRect r = ((CvContour*)contour)->rect;
			if (r.height>20&&r.width>20)
			{
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(255,0,0),1,CV_AA,0);
				(r.x>20)?(r.x-=20):(r.x=0);
				(r.y>20)?(r.y-=20):(r.y=0);
				(r.x+r.width+40<width)?(r.width+=40):(r.width=width-r.x);
				(r.y+r.height+40<height)?(r.height+=40):(r.height=height-r.y);
				cvRectangle(image,cvPoint(r.x,r.y),cvPoint(r.x+r.width,r.y+r.height),CV_RGB(255,255,0),1,CV_AA,0);
				IplImage*detectImage = cvCreateImage(cvSize(r.width,r.height),8,3);
				cvSetImageROI(image,r);
				cvCopy(image,detectImage);
				cvResetImageROI(image);
				//CvMemStorage* storage = cvCreateMemStorage(0);
				//CvSeq* Signs= cvCreateSeq(0,sizeof(CvSeq),sizeof(CvRect),storage);
				int scale = 1;
				for (int j=0;j<n;j++)
				{
					CvSeq* signt;
					signt = detect_and_draw_objects2( detectImage, cascade[j], 1 ,j);
					for(int i = 0; i < signt->total; i++ )
					{
						CvRect Sign_rect = *(CvRect*)cvGetSeqElem(signt, i );
						cvRectangle( image, cvPoint(Sign_rect.x*scale+r.x,Sign_rect.y*scale+r.y),cvPoint((Sign_rect.x+Sign_rect.width+r.x)*scale,(Sign_rect.y+Sign_rect.height+r.y)*scale),CV_RGB(255,255,255), 3 );

					}
					cvReleaseMemStorage(&signt->storage);
				}
				cvReleaseImage(&detectImage);

			}

		}
		cvNamedWindow("circle");
		cvShowImage("circle",image);
		cvWaitKey();
}

void ClaneDCL::Image2RBYImage(IplImage*image,IplImage*imageR,IplImage*imageB,IplImage*imageY,IplImage*imageG/*,IplImage*imageBlack*/)
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
			else if (H>30&&H<60&&s>0.4&&v>0.5)//Y (H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data_Y[i*step1+j] = 255;
			}
			else if (H>190&&H<230&&s>0.5&&v>0.5)//B(H>0&&H<10) || (H>330&&H<360)BLUEH>200&&H<270&&s>0.3,YELLOWH>20&&H<100&&s>0.2
			{
				data_B[i*step1+j] = 255;
			}
			else if (H>120&&H<160&&s>0.5&&v>0.5)
			{
				data_G[i*step1+j] = 255;
			}
		}
		cvDilate(imageR, imageR, 0, 4);
		cvErode(imageR, imageR, 0, 1);
		cvDilate(imageB, imageB, 0, 4);
		cvErode(imageB, imageB, 0, 1);
		cvDilate(imageY, imageY, 0, 4);
		cvErode(imageY, imageY, 0, 1);
		cvDilate(imageG, imageG, 0, 4);
		cvErode(imageG, imageG, 0, 1);
		cvReleaseImage(&Imagerect);
}

void ClaneDCL::TrafficSignDwithColor(IplImage*image,IplImage*imagegray,CvHaarClassifierCascade*cascade[],int n)
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	int height=image->height;
	int width=image->width;
	cvFindContours(imagegray, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
	for( ; contour != 0; contour = contour->h_next )
	{
		CvRect r = ((CvContour*)contour)->rect;
		if (r.height>20&&r.width>20)
		{
			int r_w=0;
			(r.x>20)?(r.x-=20):(r.x=0);
			(r.y>20)?(r.y-=20):(r.y=0);
			(r.width>r.height)?(r_w=r.width):(r_w=r.height);
			(r.x+r_w+40<width)?(r.width=r_w+40):(r.width=width-r.x);
			(r.y+r_w+40<height)?(r.height=r_w+40):(r.height=height-r.y);
			IplImage*detectImage = cvCreateImage(cvSize(r.width,r.height),8,3);
			cvSetImageROI(image,r);
			cvCopy(image,detectImage);
			cvResetImageROI(image);
			for (int j=0;j<n;j++)
			{
				CvMemStorage* storage = cvCreateMemStorage(0);
				CvSeq* signt = cvHaarDetectObjects( detectImage, cascade[j], storage, 1.2, 2,CV_HAAR_DO_CANNY_PRUNING,cvSize(24,24) );
				CvRect rect_max = cvRect(0,0,0,0);
				for(int i = 0; i < signt->total; i++ )
				{
					CvRect Sign_rect = *(CvRect*)cvGetSeqElem(signt, i );
					if(rect_max.width<Sign_rect.width) rect_max = Sign_rect;//////////////////////////////////////
				}
				if(rect_max.width!=0)
				{
					////////////recog//////////////////////
					CvRect Recog_rect = cvRect(rect_max.x+r.x,rect_max.y+r.y,rect_max.width,rect_max.height);
					IplImage*recogimage = cvCreateImage(cvSize(Recog_rect.width,Recog_rect.height),8,3);
					cvSetImageROI(image,Recog_rect);
					cvCopy(image,recogimage);
					cvResetImageROI(image);
					int classID = SingleRectRecog(recogimage);
					if (classID!=-1)
					{
						RecogResult[classID] = TRUE;
						cvRectangle( image, cvPoint(rect_max.x+r.x,rect_max.y+r.y),cvPoint(rect_max.x+rect_max.width+r.x,rect_max.y+rect_max.height+r.y),CV_RGB(0,255,255), 3 );
					}
					cvReleaseImage(&recogimage);
					printf("%d\n",classID);
				}
				cvReleaseMemStorage(&storage);
			}
			cvReleaseImage(&detectImage);
		}
	}
	cvWaitKey();
}



void ClaneDCL::LoadRecogPara()
{
	///////////////////用于检测的变量载入////////////////////////////////
	cascadeR[0] = ClaneDCL::load_object_detector(".\\cascade\\cascade_circle.xml");
	cascadeR[1] = ClaneDCL::load_object_detector(".\\cascade\\cascade_noentrynoparking.xml");
	cascadeR[2] = ClaneDCL::load_object_detector(".\\cascade\\cascade_rang2.xml");
	cascadeR[3] = ClaneDCL::load_object_detector(".\\cascade\\cascade_Stop2.xml");
	cascadeR[4] = ClaneDCL::load_object_detector(".\\cascade\\cascade_Zui.xml");
	
	cascadeB[0] = ClaneDCL::load_object_detector(".\\cascade\\cascade_blue3.xml");
	cascadeB[1] = ClaneDCL::load_object_detector(".\\cascade\\cascade_BRect2.xml");
	cascadeB[2] = ClaneDCL::load_object_detector(".\\cascade\\YellowTriangle.xml");
	cascadeB[3] = ClaneDCL::load_object_detector(".\\cascade\\cascade_EPS.xml");
	
	cascadeY[0] = ClaneDCL::load_object_detector(".\\cascade\\YellowTriangle.xml");
	cascadeY[1] = ClaneDCL::load_object_detector(".\\cascade\\cascade_LRT.xml");
	///////////////////用于识别的变量载入////////////////////////////////
	CvFileStorage * fileStorage;
	fileStorage = cvOpenFileStorage( "TSRdata.xml", 0, CV_STORAGE_READ );
	if( !fileStorage )
	{
		fprintf(stderr, "Can't open facedata.xml\n");
		return;
	}////////////////////////读入已经训练完的用于分类的文件

	size_recog.height = cvReadIntByName(fileStorage, 0, "sizeHeight", 0);
	size_recog.width = cvReadIntByName(fileStorage, 0, "sizeWidth", 0);
	nTsign  = cvReadIntByName(fileStorage, 0, "nTsign", 0);
	mdSz  = cvReadIntByName(fileStorage, 0, "mdSz", 0);

	tsignNumTruthMat = (CvMat *)cvReadByName(fileStorage, 0, "tsignNumTruthMat", 0);
	W_pcafldT = (CvMat *)cvReadByName(fileStorage, 0, "W_pcafldT", 0);
	mu_total = (CvMat *)cvReadByName(fileStorage, 0, "mu_total", 0);
	model = new CvMat* [nTsign];

	ss_recog.W_pcafldT = W_pcafldT;
	ss_recog.mu_total = mu_total;

	if (! p_recog.Init(size_recog, true)) return;
	f_recog.InitGaborKernel(5, 8);
	featureSz = f_recog.InitSamplePoints(p_recog.m_mask,size_recog,6);

	for(int i=0; i<nTsign; i++)
	{
		char varname[200];
		sprintf( varname, "model_%d", i );
		model[i] = (CvMat *)cvReadByName(fileStorage, 0, varname, 0);
	}
	// TODO: Add your control notification handler code here
	cvReleaseFileStorage( &fileStorage );
}

int ClaneDCL::SingleRectRecog(IplImage*image)
{
	CvSize size = cvSize(40,40);
	IplImage*tempImg2=cvCreateImage(size,8,3);
	IplImage*imgL=cvCreateImage(size,8,1);
	cvResize(image,tempImg2);
	cvCvtColor(tempImg2,imgL,CV_RGB2GRAY);//灰度图像
	CvMat *tempM1 = cvCreateMat(size_recog.height,size_recog.width,CV_32FC1);
	cvConvertScale(imgL,tempM1,1.0/255);
	p_recog.HomographicFilter(tempM1);
	CvMat*tfeatureL = cvCreateMat(featureSz, 1, CV_FC1);
	f_recog.GaborConvAndSample(tempM1,tfeatureL);
	CvMat*modelL = cvCreateMat(mdSz, 1, CV_64FC1);
	ss_recog.Project(tfeatureL,modelL);
	int classn=-1;
	double	minDist = 1e9, curVal; // minDist should be among -1~1 for angle metric
	//int nn = -1;
	for (int i = 0; i < nTsign; i++)
	{
		curVal = ss_recog.CalcVectorDist(model[i], modelL);
		if (curVal < minDist&&curVal<-0.6)
		{
			minDist = curVal;
			classn = tsignNumTruthMat->data.i[i];
		}
	}
	cvReleaseImage(&tempImg2);
	cvReleaseImage(&imgL);
	cvReleaseMat(&tempM1);
	cvReleaseMat(&tfeatureL);
	cvReleaseMat(&modelL);
	return classn;
}

void ClaneDCL::TrafficSignRecognition(IplImage*image)
{
	for (int i=0;i<32;i++)
	{
		RecogResult[i] = FALSE;
	}
	IplImage*imageR = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageB = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageY = cvCreateImage(cvGetSize(image),8,1);
	IplImage*imageG = cvCreateImage(cvGetSize(image),8,1);
	ClaneDCL::Image2RBYImage(image,imageR,imageB,imageY,imageG);
// 	cvNamedWindow("R");
// 	cvShowImage("R",imageR);
// 	cvNamedWindow("B");
// 	cvShowImage("B",imageB);
// 	cvNamedWindow("Y");
// 	cvShowImage("Y",imageY);
// 	cvNamedWindow("G");
// 	cvShowImage("G",imageG);
 	ClaneDCL::TrafficSignDwithColor(image,imageR,cascadeR,5);
	ClaneDCL::TrafficSignDwithColor(image,imageB,cascadeB,4);
	ClaneDCL::TrafficSignDwithColor(image,imageY,cascadeY,2);
	cvReleaseImage(&imageR);
	cvReleaseImage(&imageB);
	cvReleaseImage(&imageY);
	cvReleaseImage(&imageG);
}

bool ClaneDCL::TrafficLight(IplImage*image,CvRect rect,int n)
{
	double rat;
	double rat1;
	double rat2;
	double rat3;
	switch (n)
	{
		case 1:
			if(rect.y+rect.height*3>image->width) return FALSE;
			CvRect rect_all;
			rect_all.x = rect.x;
			rect_all.y = rect.y+rect.height;
			rect_all.width = rect.width;
			rect_all.height = rect.height*2;
			 rat = RectBlackRate(image,rect_all);
			if(rat>0.8) return TRUE;
			else return FALSE;
		break;
		case 2:
			if(rect.y-rect.height<0||rect.y+rect.height*2>image->width) return FALSE;
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
			if (rat1>0.8&&rat2>0.8) return TRUE;
			else return FALSE;
		break;
		case 3:
			if (rect.y-rect.height*2<0) return FALSE;
			CvRect rect_all2;
			rect_all2.x = rect.x;
			rect_all2.y =rect.y-rect.height*2;
			rect_all2.width = rect.width;
			rect_all2.height = rect.height*2;
			rat3 = RectBlackRate(image,rect_all2);
			if (rat>0.8) return TRUE;
			else return FALSE;
		break;

	}
}

double ClaneDCL::RectBlackRate(IplImage*image,CvRect rect)
{
	IplImage*tempImage = cvCreateImage(cvSize(rect.width,rect.height),8,3);
	IplImage*tempGray = cvCreateImage(cvSize(rect.width,rect.height),8,1);
	cvSetImageROI(image,rect);
	cvCopy(image,tempImage);
	cvResetImageROI(image);
	cvCvtColor(tempImage,tempGray,CV_RGB2GRAY);
	uchar*data = (uchar*)tempGray->imageData;
	int blackn = 0;
	for (int i=0;i<tempGray->height;i++)
		for (int j=0;j<tempGray->width;j++)
		{
			if(data[i*tempGray->widthStep+j]<20) blackn++;
		}
		double rat = (double)blackn/(tempGray->width*tempGray->height);
		cvReleaseImage(&tempImage);
		cvReleaseImage(&tempGray);
	return rat;
}