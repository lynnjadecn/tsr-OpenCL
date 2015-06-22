/*
	���й���Ԥ����Ŀǰ����Ϊ��̬ͬ�˲�+ֱ��ͼ�涨��
	ʹ�÷�����
	Init->
	��������ͼ��->HomographicFilter->ת��Ϊ8λ->HistNorm
	��ʼ����Ҫ��ģͼ���ļ� MASK_FN ������֮�������Ϊ0����ӵ�б�׼ֱ��ͼ��ͼ���ļ� HISTMD_FN
*/

#pragma once
#include "cv.h"
#include "highgui.h"

#define MASK_FN		"0.jpg"
#define HISTMD_FN	"mask.jpg"

class CLightPrep // ������Init
{
public:
	CLightPrep();
	~CLightPrep();

	CvMat		*h; // ����̬ͬ�˲��ĸ�˹��ͨ��
	int			h_radius;
	CvMat		*m_mask, *m_invMask;

	// ����ֱ��ͼ�涨��
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
