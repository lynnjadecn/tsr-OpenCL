// tsr-cl.cpp : Defines the entry point for the console application.
//

#include <opencv/cv.h>
//#include <opencv/highgui.h>
#include "signDcl.h"
#include <stdio.h>
#include <tchar.h>
#include <ctime>
#include <iostream>
#include <CL/cl.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <tchar.h>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

// below example is to test whether teh kernel satisfied with porject 
int convertToString_(const char *filename, std::string& s)
{
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return NULL;
        }
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    printf("Error: Failed to open file %s\n", filename);
    return 1;
}
/*
int main(){
			float  H,v,s;
			float com[] = {0,10,340,360,0.4,0.2,60,0.5,190,230,65,100};
			uchar temp = 255;
 			cout << "please input seven numbers:\n";
			cin >> H  >> v >> s;
			// 并行代码如下：
			cl_uint status;
    		cl_platform_id platform;
    		uchar result;
		    //创建平台对象
    		status = clGetPlatformIDs(1, &platform, NULL);
			if (status)
        		cout << status << " platform set up" <<endl;
    		cl_device_id device;
    		//创建 GPU 设备
    		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1,	&device, NULL);
    		//创建context
    		cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    		//创建命令队列
    		cl_command_queue commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
    		if (commandQueue == NULL) 
            	perror("Failed to create commandQueue for device 0.");
            //创建三个 OpenCL 内存对象，并把buf1 的内容通过隐式拷贝的方式
    		//拷贝到clbuf1, buf2 的内容通过显示拷贝的方式拷贝到clbuf2
    		cl_mem memObjects[1] = { 0 };
    		memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * 1, &result, NULL);
    		if (memObjects[0] == NULL) 
        		perror("Error in clCreateBuffer.\n");
			const char * filename = "F:\\My_opencv\\OpenCL\\tsr-cl\\sign-detect-picture\\Judgementation.cl";
    		std::string sourceStr;
		    status = convertToString_(filename, sourceStr);
    		if (status)
        		cout << status << "  convert error" << endl;
    		const char * source = sourceStr.c_str();
    		size_t sourceSize[] = { strlen(source) };
    		//创建程序对象
    		cl_program program = clCreateProgramWithSource( context, 1, &source, sourceSize, NULL);
    		//编译程序对象
    		status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    		if (status)
        		cout << status << "  !!!!!!!!" <<endl;
    		if (status != 0)
    		{
        		printf("clBuild failed:%d\n", status);
        		char tbuf[0x10000];
        		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
        		printf("\n%s\n", tbuf);
        		//return −1;
    		}

		    //创建 Kernel 对象
    		cl_kernel kernel = clCreateKernel(program, "Judgementation_s", NULL);
		    //设置 Kernel 参数
    		//cl_int clnum = NWITEMS;  //r,g,b,s,h,v,H 
			cout << "here is normal or not\n";
    		status = clSetKernelArg(kernel, 0, sizeof(float), &H);
    		status = clSetKernelArg(kernel, 1, sizeof(float), &v);
    		status = clSetKernelArg(kernel, 2, sizeof(float), &s);
			status = clSetKernelArg(kernel, 3, sizeof(float)*12, com);
			status = clSetKernelArg(kernel, 4, sizeof(uchar), &temp);
    		status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[0]);
    		if (status)
        		cout << "Here is error" << endl;
 		   //执行 kernel
    	   cl_event prof_event;
    	   cl_ulong ev_start_time = (cl_ulong)0;
    	   cl_ulong ev_end_time = (cl_ulong)0;
    	   double rum_time;
    	   status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, NULL, NULL, 0, NULL, &prof_event);
    	   if (status)
        		cout << "Error is here\n";
		   cout<< endl;
    	   clFinish(commandQueue);
    	   //读取时间
    	   status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong),&ev_start_time,NULL);
    	   status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong),&ev_end_time,NULL);
    	  if (status) 
             	perror("Error happened when read tine execution \n");
    	  rum_time = (double)(ev_end_time - ev_start_time);
    	  cout << "Executuion time is:" << rum_time << endl;
          //数据拷回 host 内存
    	  status = clEnqueueReadBuffer(commandQueue, memObjects[0],CL_TRUE, 0, sizeof(uchar)* 1, &result, 0, NULL, NULL);
    	  if (status) 
        	 perror("读回数据的时候发生错误\n");
          //结果显示
    	  cout << result << endl;
    	  //删除 OpenCL 资源对象
	      clReleaseMemObject(memObjects[0]);
   	 	  clReleaseProgram(program);
    	  clReleaseCommandQueue(commandQueue);
    	  clReleaseContext(context);

	system("pause");

	return 0;
}
*/

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
	VideoCapture cap0(0);
	VideoCapture cap1(1);
    if(!cap0.isOpened())  
    {  
        return -1;  
    }  
    if(!cap1.isOpened())  
    {  
        return -1;  
    }  
	IplImage image;
    Mat frame0,frame1;   
    bool stop = false;  
    while(!stop)  
    {  
        cap0 >> frame0;
		cap1 >> frame1;
		image = IplImage(frame0); 
      //  imshow("Vedio0",frame0); 
	  //	imshow("Vedio1",frame1);
        if(waitKey(30) >=0)  
            stop = true;  


    }  
	/*source code
	CvCapture* capture = 0;
	capture=cvCaptureFromAVI( "F:\\My_VS\\Detect_Sign\\Video2011824105147859.avi");
	*/
	if( capture )
	{

		if( !cvGrabFrame( capture ))
			return 0;
		IplImage *image2 = cvRetrieveFrame( capture );
		
		CsignDcl::LoadRecogPara();
		////////////////////////////////////Ê¶±ð¼ì²âÆ÷ÔØÈë////////////////////
		IplImage* image;
		//int count = 0;
		//double result=0;
		for(;;)
		{   
		//	 count++;
		//	 clock_t start, finish;
		//	 double CPU_time;
		//	 start = clock();

          
			if( !cvGrabFrame( capture ))
				break;
			image = cvRetrieveFrame( capture );//Get frame from avi;
			IplImage*imageP = cvCreateImage(cvSize(image->width/2,image->height/2),8,3);
			cvResize(image,imageP);
			vector<int> *signs = new vector<int> ();
			CsignDcl::TrafficSignRecognition(imageP, signs);
			vector<int>::iterator iter; /*iterator ³éÏóÁËÖ¸ÕëµÄ¾ø´ó²¿·Ö»ù±¾ÌØÕ÷*/

			int nu = 0;
			for( iter = signs->begin(); iter != signs->end(); iter++ ) 
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
			}

		    cvNamedWindow("Bird");
			cvShowImage("Bird",imageP);
			cvWaitKey(50);
			delete signs;

		//	finish = clock();
		//	CPU_time = (double)(finish-start)/CLOCKS_PER_SEC;
		//	cout<< CPU_time << endl;
		//	result= CPU_time + result;
		//	if (result >= 60)
		//		break;
		}
		
		//cout<< count<<"what the hell"<< endl;

		cvReleaseImage(&image);
	}
	cvReleaseCapture( &capture );
	
	system("pause");

	return 0;
}



/*  
int main()  
{  

    VideoCapture cap0(0);
	VideoCapture cap1(1);
    if(!cap0.isOpened())  
    {  
        return -1;  
    }  
    if(!cap1.isOpened())  
    {  
        return -1;  
    }  
	IplImage image;
    Mat frame0,frame1;   
    bool stop = false;  
    while(!stop)  
    {  
        cap0 >> frame0;
		cap1 >> frame1;
		image = IplImage(frame0); 
        imshow("Vedio0",frame0); 
		imshow("Vedio1",frame1);
        if(waitKey(30) >=0)  
            stop = true;  
    }  
    return 0;  
} 
*/
