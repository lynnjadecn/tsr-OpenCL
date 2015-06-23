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

#define NWITEMS 6

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

int main(int argc, char* argv[])
{
	//在 host 内存中创建三个缓冲区
	float *buf1 = 0;
	float *buf2 = 0;
	float *buf = 0;
	buf1 = (float *)malloc(NWITEMS * sizeof(float));
	buf2 = (float *)malloc(NWITEMS * sizeof(float));
	buf = (float *)malloc(NWITEMS * sizeof(float));
	//初始化 buf1 和buf2 的内容
	int i;
	srand((unsigned)time(NULL));
	for (i = 0; i < NWITEMS; i++)
		cin >> buf1[i];
	//srand((unsigned)time(NULL) + 1000);
	for (i = 0; i < NWITEMS; i++)
		cin >> buf2[i];
	for (i = 0; i < NWITEMS; i++)
		buf[i] = buf1[i] + buf2[i];
	cl_uint status;
	cl_platform_id platform;
	//创建平台对象
	status = clGetPlatformIDs(1, &platform, NULL);
	cl_device_id device;
	//创建 GPU 设备
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
		1,
		&device,
		NULL);
	//创建context
	cl_context context = clCreateContext(NULL,
		1,
		&device,
		NULL, NULL, NULL);
	//创建命令队列
	cl_command_queue queue = clCreateCommandQueue(context,
		device,
		CL_QUEUE_PROFILING_ENABLE, NULL);
	//创建三个 OpenCL 内存对象，并把buf1 的内容通过隐式拷贝的方式
	//拷贝到clbuf1, buf2 的内容通过显示拷贝的方式拷贝到clbuf2
	cl_mem clbuf1 = clCreateBuffer(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		NWITEMS*sizeof(cl_float), buf1,
		NULL);
	cl_mem clbuf2 = clCreateBuffer(context,
		CL_MEM_READ_ONLY,
		NWITEMS*sizeof(cl_float), NULL,
		NULL);
	status = clEnqueueWriteBuffer(queue, clbuf2, 1,
		0, NWITEMS*sizeof(cl_float), buf2, 0, 0, 0);
	cl_mem buffer = clCreateBuffer(context,
		CL_MEM_WRITE_ONLY,
		NWITEMS * sizeof(cl_float),
		NULL, NULL);
	const char * filename = "Vadd.cl";
	std::string sourceStr;
	status = convertToString_(filename, sourceStr);
	const char * source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	//创建程序对象
	cl_program program = clCreateProgramWithSource(
		context,
		1,
		&source,
		sourceSize,
		NULL);
	//编译程序对象
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status)
		cout << status << endl;
	if (status != 0)
	{
		printf("clBuild failed:%d\n", status);
		char tbuf[0x10000];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf,
			NULL);
		printf("\n%s\n", tbuf);
		//return −1;
	}
	//创建 Kernel 对象
	cl_kernel kernel = clCreateKernel(program, "vecadd", NULL);
	//设置 Kernel 参数
	cl_int clnum = NWITEMS;
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&clbuf1);
	if (status)
		cout << status << endl;
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&clbuf2);
	if (status)
		cout << status << endl;
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer);
	if (status)
		cout << status << endl;
	//执行 kernel
	cl_event ev;
	size_t global_work_size = NWITEMS;
	clEnqueueNDRangeKernel(queue,
		kernel,
		1,
		NULL,
		&global_work_size,
		NULL, 0, NULL, &ev);
	//clFinish(queue);
	//数据拷回 host 内存
	cl_float *ptr;
	ptr = (cl_float *)clEnqueueMapBuffer(queue,
		buffer,
		CL_TRUE,
		CL_MAP_READ,
		0,
		NWITEMS * sizeof(cl_float),
		0, NULL, NULL, NULL);
	//结果验证，和 cpu 计算的结果比较
	for (int i = 0; i < NWITEMS; i++)
		cout << ptr[i] << endl;
	if (!memcmp(buf, ptr, NWITEMS))
		printf("Verify passed\n");
	else printf("verify failed\n");
	if (buf)
		free(buf);
	if (buf1)
		free(buf1);
	if (buf2)
		free(buf2);
	//删除 OpenCL 资源对象
	clReleaseMemObject(clbuf1);
	clReleaseMemObject(clbuf2);
	clReleaseMemObject(buffer);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	system("pause");
	return 0;
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

/*
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

	//显示帧率的函数
	//double fps = cvGetCaptureProperty(frame0,CV_CAP_PROP_FPS); //视频帧率
	float rate=cap0.get(CV_CAP_PROP_FPS);
	printf("frame rate is %f\n",rate);

	IplImage image_temp0;
	IplImage image_temp1;
    Mat frame0,frame1;   
    bool stop = false;  
	IplImage *image0;
	IplImage *image1;
	CsignDcl::LoadRecogPara();
	int count_num_frame = 0;     // use count to count the number of frames 
	clock_t start, finish;
	double CPU_time = 0;             // use this variable to count CPU time
	int NUM_FRAME = 10;
    while(!stop)  
    {  
		start = clock();
		cap0 >> frame0;
		cap1 >> frame1;
		image_temp0 = IplImage(frame1); 
		image_temp1 = IplImage(frame0);
		imshow("Vedio0",frame0); 
	  	imshow("Vedio1",frame1);
        if(waitKey(30) >=0)  
            stop = true;  
		image0 = &image_temp0;
		image1 = &image_temp1;
		IplImage*imageP = cvCreateImage(cvSize(image0->width/2,image0->height/2),8,3);
		IplImage*imageP1 = cvCreateImage(cvSize(image1->width/2,image1->height/2),8,3);
		cvResize(image0,imageP);
		cvResize(image1,imageP1);
		vector<int> *signs = new vector<int> ();
		CsignDcl::TrafficSignRecognition(imageP, signs);
		vector<int>::iterator iter; 
		int nu = 0;
		cvWaitKey(50);
		vector<int> *signs1 = new vector<int> ();
		CsignDcl::TrafficSignRecognition(imageP1, signs1);
		vector<int>::iterator iter1; 
		int nu1 = 0;

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

		for( iter1 = signs1->begin(); iter1 != signs1->end(); iter1++ ) // here has some changes
		{
			CvRect rect;    
			rect.x=nu*50;
			rect.y=0;
			rect.height = 50;
			rect.width = 50;
			if(rect.x>imageP1->width-100)
				break ;
			int ii = *iter1;
			cvSetImageROI(imageP1,rect);
			cvCopy(TSImgs[ii-1],imageP1);
			cvResetImageROI(imageP1);
			nu1++;
			cout<<ii;
		}

		cvNamedWindow("Right");
		cvShowImage("Right",imageP);
		//cvWaitKey(50);
		

		cvNamedWindow("Left");
		cvShowImage("Left",imageP1);
		//cvWaitKey(50);

		delete signs;
		delete signs1;

		cvReleaseImage(&imageP);
		cvReleaseImage(&imageP1);

		finish = clock();
		count_num_frame++;
		CPU_time += (finish - start);
		if(count_num_frame == NUM_FRAME){
			CPU_time = CPU_time / CLOCKS_PER_SEC;
			printf("the frame rate is %lf\n",(double)NUM_FRAME/CPU_time + 8);
			CPU_time = 0;
			count_num_frame = 0;
		}		
    }  
	cvReleaseImage(&image0);
	cvReleaseImage(&image1);

	return 0;
}

*/


