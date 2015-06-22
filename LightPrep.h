/*
	进行光照预处理。目前方法为：同态滤波+直方图规定化
	使用方法：
	Init->
	浮点人脸图像->HomographicFilter->转换为8位->HistNorm
	初始化需要掩模图像文件 MASK_FN （人脸之外的区域为0）和拥有标准直方图的图像文件 HISTMD_FN
*/

#pragma once
#include "cv.h"
#include "highgui.h"

#define MASK_FN		"0.jpg"
#define HISTMD_FN	"mask.jpg"

class CLightPrep // 必须先Init
{
public:
	CLightPrep();
	~CLightPrep();

	CvMat		*h; // 用于同态滤波的高斯高通核
	int			h_radius;
	CvMat		*m_mask, *m_invMask;

	// 用于直方图规定化
	CvHistogram	*m_histdst, *m_histsrc;
	CvMat		*lutEq2Dst, *m_lutSrc2Eq;
	double		m_scale;

	void InitFilterKernel(CvSize imgSz);
	bool InitMask(CvSize imgSz, bool useMask);
	bool InitHistNorm(CvSize imgSz);
	bool Init(CvSize imgSz, bool useMask);

	// img must be CV_F32C1
	void HomographicFilter(CvMat *img);

	// derived from http://www.opencv.org.cn/forum/viewtopic.php?f=1&t=7055
	// only for 8UC1
	void HistNorm(CvArr *src);

	 // temporarily not in use
	void MaskFace(CvArr *src);

	void GenFilterKernel(CvMat *h, int filterType, /*bool bHighpass,*/ double d0, int order = 1);
};
