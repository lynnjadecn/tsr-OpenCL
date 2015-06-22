/*
	��ȡGabor������
	�÷���
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
		GaborС����
					  u^2+v^2         -(u^2+v^2)*(x^2+y^2)                               -sigma^2
			G(x,y) = --------- * exp(----------------------) * [ exp(i*(u*x+v*y)) - exp(----------)]
					  sigma^2               2*sigma^2                                       2
						  u         kmax         pi*MU         kmax         pi*MU
			in which, k=(   ), u = ------ * cos(-------), v = ------ * sin(-------).
						  v         f^NU           8           f^NU           8
			in which, kmax = pi/2,f = sqrt(2).
			sigma����G���ƽ��NU����Ƶ�ʡ����߶ȡ���MU�����Ƕ���ת
			ref: Gabor Feature Based Classification Using the Enhanced Fisher Linear 
			Discriminant Model for Face Recognition, Chengjun Liu et al.

		������
			scaleNum:	�߶ȳ���������NU�ĳ�����
			angleNum:	�Ƕȳ���������MU�ĳ�������MU = 0,1,...,angleNum-1
			scales:		���ΪNULL����߶ȳ���NU = 0,1,...,scaleNum-1; ����scalesָ��
			kernelRadius: GaborС���Ĵ������������������Ϊ0�����Զ�����
	*/
	void InitGaborKernel(int scaleNum, int angleNum, double sigma = 2*PI, 
		int *scales = NULL, int kernelRadius = 0);


	/*
		mask:		��ģͼ�񣬲�Ϊ0������Ϊ��Ч��������
		imgSize:	����ͼ��ߴ磬Ҳ����mask�ߴ�
		sampleIntv:	�������
		����:		ԭʼ�����������ȣ��������� * scaleNum * angleNum��
	*/
	int InitSamplePoints(CvMat *mask, CvSize imgSize, int sampleIntv);


	/*
		src:	���븡��ͼ��
		dst:	���ԭʼ����������
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


	// ������ر���
	int		sampleIntv;
	CvSize	ssz, padSize;
	CvPoint	topleft;
	CvMat	*padded;
	int		*startPts, *endPts;
};
