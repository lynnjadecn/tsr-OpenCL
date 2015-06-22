// tsr-cl.cpp : Defines the entry point for the console application.
//
#include "cv.h"
#include "highgui.h"
#include"signDcl.h"
#include <stdio.h>
#include <tchar.h>

int _tmain(int argc, _TCHAR* argv[])
{

	IplImage *TSImgs[27];

   for(int i=1;i<28;i++)
   {
			 std::string name;
             std::stringstream namejpg(name);
             namejpg <<i;
			 string namefinal;
			 if(i>99)
			       namefinal = namejpg.str()+".jpg";
			 else if(i>9)
				   namefinal = "0"+namejpg.str()+".jpg";
			 else
					namefinal = "00"+namejpg.str()+".jpg";
	     string w= ".\\TSImgs\\"+namefinal;
		  TSImgs[i-1]=cvLoadImage(w.c_str());
   }

	IplImage* motion = 0; 
	CvCapture* capture = 0;
	capture = cvCaptureFromAVI( "Video2011824105147859.avi"); 
	if( capture )
	{
		if( !cvGrabFrame( capture ))
			return 0;
		IplImage *image2 = cvRetrieveFrame( capture );
		
		
		CsignDcl::LoadRecogPara();
		////////////////////////////////////识别检测器载入////////////////////
		IplImage* image;
		for(;;)
		{
			if( !cvGrabFrame( capture ))
				break;
			image = cvRetrieveFrame( capture );//Get frame from avi;
			IplImage*imageP = cvCreateImage(cvSize(image->width/2,image->height/2),8,3);
			cvResize(image,imageP);
			vector<int> *signs = new vector<int> ();
			CsignDcl::TrafficSignRecognition(imageP, signs);
			vector<int>::iterator iter; /*iterator 抽象了指针的绝大部分基本特征*/
			int nu = 0;

	/*	 for( iter = signs->begin(); iter != signs->end(); iter++ ) 
		 {
			CvRect rect;
			 rect.x=nu*50;
			 rect.y=0;
			 rect.height = 50;
			 rect.width = 50;
			 if(rect.x>imageP->width-100)
				 break ;
			 int ii = *iter;
			 cvSetImageROI(imageP,rect);
			 cvCopy(TSImgs[ii-1],imageP);
			 cvResetImageROI(imageP);
			 nu++;
			 cout<<ii;
		  }*/
		    cvNamedWindow("Bird");
			cvShowImage("Bird",imageP);
			cvWaitKey(50);
			delete signs;
		}
		cvReleaseImage(&image);

	}
	cvReleaseCapture( &capture );
	return 0;
}

