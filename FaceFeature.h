/*
	提取Gabor特征。
	用法：
		InitGaborKernel -> InitSamplePoints -> GaborConvAndSample
*/

#pragma once
#include "cv.h"
#include "highgui.h"

#define PI		3.1415926535897932384626433832795
#define CV_FC1	CV_32FC1 // depth = 32 or 64
#define CV_FC2	CV_32FC2

class CFaceFeature
{
public:
	CFaceFeature(void);
	~CFaceFeature(void);

	/*
		Gabor小波：
					  u^2+v^2         -(u^2+v^2)*(x^2+y^2)                               -sigma^2
			G(x,y) = --------- * exp(----------------------) * [ exp(i*(u*x+v*y)) - exp(----------)]
					  sigma^2               2*sigma^2                                       2
						  u         kmax         pi*MU         kmax         pi*MU
			in which, k=(   ), u = ------ * cos(-------), v = ------ * sin(-------).
						  v         f^NU           8           f^NU           8
			in which, kmax = pi/2,f = sqrt(2).
			sigma↑，G变扁平；NU↑，频率↓，尺度↑；MU↑，角度旋转
			ref: Gabor Feature Based Classification Using the Enhanced Fisher Linear 
			Discriminant Model for Face Recognition, Chengjun Liu et al.

		参数：
			scaleNum:	尺度抽样数，即NU的抽样数
			angleNum:	角度抽样数，即MU的抽样数，MU = 0,1,...,angleNum-1
			scales:		如果为NULL，则尺度抽样NU = 0,1,...,scaleNum-1; 否则按scales指定
			kernelRadius: Gabor小波的窗长（像素数），如果为0，则自动计算
	*/
	void InitGaborKernel(int scaleNum, int angleNum, double sigma = 2*PI, 
		int *scales = NULL, int kernelRadius = 0);


	/*
		mask:		掩模图像，不为0的区域为有效人脸区域
		imgSize:	人脸图像尺寸，也就是mask尺寸
		sampleIntv:	抽样间隔
		返回:		原始特征向量长度（抽样点数 * scaleNum * angleNum）
	*/
	int InitSamplePoints(CvMat *mask, CvSize imgSize, int sampleIntv);


	/*
		src:	输入浮点图像
		dst:	输出原始特征列向量
	*/
	void GaborConvAndSample(CvArr *src, CvMat *dst);


	void ShowGaborKernel();
	void GaborConv(CvArr *src, CvMat *dst, int scale, int angle);

	CvSize CheckConcat( CvArr ***pppa, int rowNum, int colNum, int *colNums );
	void ConcatArrs( CvArr ***pppa, CvMat *dst, int rowNum, int colNum, int *colNums = NULL );

	double	kmax, f, sigma;
	int		scaleNum, angleNum, kernelWidth, kernelRadius;
	int		*scales;
	CvMat	***G;


	// 抽样相关变量
	int		sampleIntv;
	CvSize	ssz, padSize;
	CvPoint	topleft;
	CvMat	*padded;
	int		*startPts, *endPts;
};
