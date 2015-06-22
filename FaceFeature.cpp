//#include "StdAfx.h"
#include "FaceFeature.h"
//#include "WinDef.h"
#include <math.h>
//#include "cvUtils.h"

#ifdef _DEBUG
//#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

CFaceFeature::CFaceFeature(void)
{
}

CFaceFeature::~CFaceFeature(void)
{

	delete [scaleNum]scales;
	for (int i = 0; i < scaleNum; i++) // 释放小波组
	{
		for (int j = 0; j < angleNum; j++)
			cvReleaseMat(&G[i][j]);
		delete [angleNum]G[i]; // 这种写法与 delete []G[i]相同
	}
	delete [scaleNum]G;

	cvReleaseMat(&padded);
	delete []startPts;
	delete []endPts;
}

void CFaceFeature::InitGaborKernel( int scaleNum, int angleNum, double sigma /*= 2*PI*/, int *scales /*= NULL*/, int kernelRadius /*= 0*/ )
{
	kmax = PI/2;
	f = sqrt(double(2));
	this->sigma = sigma;
	if (kernelRadius <= 0) // 自动计算核尺寸，以使边界上小波幅度的最大值小于th
	{
		double th = 5e-3;
		kernelRadius = (int)ceil( sqrt( -log( th*sigma*sigma/kmax/kmax ) *2*sigma*sigma/kmax/kmax ) );
	}
	this->kernelRadius = kernelRadius;
	this->kernelWidth = kernelRadius*2 + 1;
	this->angleNum = angleNum;
	this->scaleNum = scaleNum;
	this->scales = new int[scaleNum];

	G = new CvMat **[scaleNum];
	CvMat *R = cvCreateMat(this->kernelWidth, this->kernelWidth, CV_FC1),
		*I = cvCreateMat(this->kernelWidth, this->kernelWidth, CV_FC1);
	double DC = exp(-sigma*sigma/2);

	for (int i = 0; i < scaleNum; i++)
	{
		this->scales[i] = (scales ? scales[i] : i);
		G[i] = new CvMat *[angleNum];
		double k = kmax/pow(f, this->scales[i]),
			tmpV = k*k /sigma/sigma;

		for (int j = 0; j < angleNum; j++)
		{
			G[i][j] = cvCreateMat(this->kernelWidth, this->kernelWidth, CV_FC2);
			double phi = PI/angleNum*j;
			double tmpV0, tmpV1;

			for (int y = -kernelRadius; y <= kernelRadius; y++)
			{
				for (int x = -kernelRadius; x <= kernelRadius; x++)
				{
					tmpV0 = tmpV * exp(-tmpV * (x*x + y*y) / 2.0);
					tmpV1 = k*cos(phi)*x + k*sin(phi)*y;
					cvmSet( R, y+kernelRadius, x+kernelRadius, tmpV0 * cos(tmpV1 - DC) );
					cvmSet( I, y+kernelRadius, x+kernelRadius, tmpV0 * sin(tmpV1) );

					/* G{scale_idx,angle_idx} = k^2/sigma^2 * exp(-k^2*(X.^2+Y.^2)/2/sigma^2)...
						.*(exp(1i*(k*cos(phi)*X+k*sin(phi)*Y) - DC)); */
				}
			}

			cvMerge(R, I, NULL, NULL, G[i][j]); // 复值小波
		}
	}
	cvReleaseMat(&R);
	cvReleaseMat(&I);
}

int CFaceFeature::InitSamplePoints(CvMat *mask, CvSize imgSize, int sampleIntv)
{
	this->sampleIntv = sampleIntv;
	ssz = cvSize( (imgSize.width - 1)/sampleIntv + 1, 
		(imgSize.height - 1)/sampleIntv + 1 );
	topleft = cvPoint(((imgSize.width - 1) % sampleIntv)/2,
		((imgSize.height - 1) % sampleIntv)/2);
	padSize = cvSize( __max(0, kernelRadius - topleft.x), // 卷积时需要延拓
		__max(0, kernelRadius - topleft.y) );
	padded = cvCreateMat(imgSize.height + padSize.height * 2, imgSize.width + padSize.width * 2, CV_FC1);

	CvPoint pt;
	startPts = new int[ssz.height]; // 每行的起始采样点的x坐标
	endPts = new int[ssz.height]; // 每行最后一个采样点的x坐标
	int ptNum = 0;
	for (int i = 0; i < ssz.height; i++)
	{
		pt.y = topleft.y + i*sampleIntv;
		startPts[i] = topleft.x;
		endPts[i] = 0;
		for (int j = 0; j < ssz.width; j++)
		{
			pt.x = topleft.x + j*sampleIntv;
			if (mask && cvGetReal2D(mask, pt.y, pt.x) == 0)
			{
				if (endPts[i] == 0) // 相当于一个flag
					startPts[i] = pt.x + sampleIntv;
			}
			else
			{
				endPts[i] = pt.x;
				ptNum++;
			}
		}
	}

	return ptNum * scaleNum * angleNum;
}

void CFaceFeature::GaborConvAndSample( CvArr *src, CvMat *dst )
{
	/*
	ssz = floor((sz-1)/sampleRate);
	[idxX idxY] = meshgrid(0:ssz(2),0:ssz(1));
	topleft = floor(rem(sz-1,sampleRate)/2)+1;
	idxX = idxX*sampleRate+topleft(2);
	idxY = idxY*sampleRate+topleft(1);
	samplePos = sub2ind(sz,idxY,idxX);
	*/
	
	cvCopyMakeBorder(src, padded, cvPoint(padSize.width, padSize.height) , IPL_BORDER_REPLICATE);

	CvRect	rc = cvRect(0,0, kernelWidth,kernelWidth);
	CvMat	*sub, tmpHeader;
	int		resIdx = 0;
	CvMat	*I = cvCreateMat(kernelWidth, kernelWidth, CV_FC1),
		*C = cvCreateMat(kernelWidth, kernelWidth, CV_FC2),
		*D = cvCreateMat(kernelWidth, kernelWidth, CV_FC2);
	cvSetZero(I);

	
	for (int y = 0; y < ssz.height; y++)
	{
		rc.y = topleft.y + padSize.height + y*sampleIntv - kernelRadius;
		for (int x = startPts[y]; x <= endPts[y]; x += sampleIntv)
		{
			rc.x = x + padSize.width - kernelRadius;
			sub = cvGetSubRect(padded, &tmpHeader, rc);
			cvMerge(sub, I, NULL, NULL, C);
			for (int i = 0; i < scaleNum; i++) // 每次对一个点做所有卷积，以节省一些操作
			{
				for (int j = 0; j < angleNum; j++)
				{
					cvMulSpectrums(C, G[i][j], D, NULL);
					CvScalar sum = cvSum(D);
					cvmSet(dst, resIdx++, 0, sqrt(sum.val[0]*sum.val[0] + sum.val[1]*sum.val[1]));
				}
			}
		}
	}

	cvReleaseMat(&I);
	cvReleaseMat(&C);
	cvReleaseMat(&D);
}

void CFaceFeature::ShowGaborKernel()
{
	int *an = new int[scaleNum];
	
	CvSize	sz = CheckConcat((CvArr***)G, scaleNum, angleNum, NULL);
	CvMat	*m = cvCreateMat(sz.height, sz.width, CV_FC2),
		*r = cvCreateMat(sz.height, sz.width, CV_FC1);
	ConcatArrs((CvArr***)G, m, scaleNum, angleNum, NULL);
	cvSplit(m, r, NULL,NULL,NULL);
	cvNormalize(r, r, 1,0, CV_MINMAX);
	cvShowImage("Gabor kernel real part", r);
	cvWaitKey();
	cvReleaseMat(&m);
	cvReleaseMat(&r);
	cvDestroyWindow("Gabor kernel real part");
}

void CFaceFeature::GaborConv( CvArr *src, CvMat *dst, int scale, int angle )
{/*
	padSize.height = 9;
	padSize.width = 9;


	CvRect	rc = cvRect(0,0, kernelWidth,kernelWidth);
	CvMat	*sub, tmpHeader;
	int		resIdx = 0;
	CvSize	sz = cvGetSize(src);
	CvMat	*I = cvCreateMat(kernelWidth, kernelWidth, CV_FC1),
		*C = cvCreateMat(kernelWidth, kernelWidth, CV_FC2),
		*D = cvCreateMat(kernelWidth, kernelWidth, CV_FC2);
	cvSetZero(I);
	padded = cvCreateMat(sz.height + padSize.height * 2, sz.width + padSize.width * 2, CV_FC1);
	cvCopyMakeBorder(src, padded, cvPoint(padSize.width, padSize.height) , IPL_BORDER_REPLICATE);


	for (int y = 0; y < sz.height; y++)
	{
		rc.y = y + padSize.height - kernelRadius;
		for (int x = 0; x < sz.width; x++)
		{
			rc.x = x + padSize.width - kernelRadius;
			sub = cvGetSubRect(padded, &tmpHeader, rc);
			cvMerge(sub, I, NULL, NULL, C);
			cvMulSpectrums(C, G[scale][angle], D, NULL);
			CvScalar sum = cvSum(D);
			cvmSet(dst, y, x, sqrt(sum.val[0]*sum.val[0] + sum.val[1]*sum.val[1]));
		}
	}

	cvReleaseMat(&I);
	cvReleaseMat(&C);
	cvReleaseMat(&D);*/
}

void CFaceFeature::ConcatArrs( CvArr ***pppa, CvMat *dst, int rowNum, int colNum, int *colNums /*= NULL */)
{
	CvMat	tmpHeader, *sub = 0;
	int		m = 0, n;
	CvSize	sz;
	for (int i = 0; i < rowNum; i++)
	{
		n = 0;
		for (int j = 0; j < (colNum ? colNum : colNums[i]); j++)
		{
			sz = cvGetSize(pppa[i][j]);
			sub = cvGetSubRect(dst, &tmpHeader, cvRect(n,m, sz.width, sz.height));
			cvCopy(pppa[i][j], sub);
			n += sz.width;
		}
		m += sz.height;
	}
}

CvSize CFaceFeature::CheckConcat( CvArr ***pppa, int rowNum, int colNum, int *colNums )
{
	int **h = new int *[rowNum], *w = new int[rowNum], height = 0;
	for (int i = 0; i < rowNum; i++)
	{
		h[i] = new int[(colNum ? colNum : colNums[i])];
		w[i] = 0;
		for (int j = 0; j < (colNum ? colNum : colNums[i]); j++)
		{
			h[i][j] = cvGetSize(pppa[i][j]).height;
			w[i] += cvGetSize(pppa[i][j]).width;
		}
	}
	for (int i = 0; i < rowNum; i++)
	{
		assert(w[i] == w[0]);
		for (int j = 0; j < (colNum ? colNum : colNums[i]); j++)
			assert(h[i][j] == h[i][0]);
		height += h[i][0];
		delete []h[i];
	}
	delete []h;
	int width = w[0];
	delete []w;
	return cvSize(width, height);
}