
#include "LightPrep.h"
//#include "cvUtils.h"

#ifdef _DEBUG
//#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

bool CLightPrep::Init( CvSize imgSz, bool useMask )
{
	InitFilterKernel(imgSz);
	return (InitMask(imgSz, useMask) && InitHistNorm(imgSz));
}

void CLightPrep::InitFilterKernel( CvSize imgSz )
{
	double	d0 = 1.414*1;
	h_radius = (imgSz.width + imgSz.height) /2/10;
	double	f_high = 1, f_low = .4;
	h = cvCreateMat(h_radius*2+1, h_radius*2+1, CV_32FC1);
	GenFilterKernel(h, 1, d0); // gauss kernel
	cvScale(h, h, -(f_high - f_low), 0); // derived from a downloaded MATLAB homographic filter function
	cvmSet(h, h_radius, h_radius, cvmGet(h, h_radius, h_radius)+f_high);
}

bool CLightPrep::InitMask( CvSize imgSz, bool useMask )
{
	if (! useMask)
	{
		m_mask = NULL;
		return true;
	}

	IplImage *maskOri = cvLoadImage(MASK_FN, CV_LOAD_IMAGE_GRAYSCALE);
	if(!maskOri) {
		//CString msg;
		//msg.Format("Can't load mask image %s", MASK_FN);
		//::AfxMessageBox(msg);
		return false;
	}

	m_mask = cvCreateMat(imgSz.height, imgSz.width, CV_8UC1);
	m_invMask = cvCreateMat(imgSz.height, imgSz.width, CV_8UC1);
	cvResize(maskOri, m_mask);
	cvCmpS(m_mask, 0, m_invMask, CV_CMP_EQ); // È¡·´
	//DispCvArr(m_mask,"m_mask");
	cvReleaseImage(&maskOri);
	return true;
}

bool CLightPrep::InitHistNorm( CvSize imgSz )
{
	int histSz = 256;
	IplImage *histModel = cvLoadImage(HISTMD_FN, CV_LOAD_IMAGE_GRAYSCALE);
	if(!histModel) {
		//CString msg;
		//msg.Format("Can't load histogram model image %s", HISTMD_FN);
		//::AfxMessageBox(msg);
		return false;
	}


	m_histdst = cvCreateHist(1, &histSz, CV_HIST_ARRAY);
	cvCalcArrHist((CvArr **)&histModel, m_histdst, 0, m_mask);
	cvNormalizeHist(m_histdst, 1);
	cvReleaseImage(&histModel);

	CvMat *lutSumDst = cvCreateMat(1, histSz, CV_8UC1);
	lutEq2Dst = cvCreateMat(1, histSz, CV_8UC1);
	float	*histOfDst = (float *)cvPtr1D(m_histdst->bins, 0);
	float	sum = 0;
	for(int i = 0; i < 256; i++)
	{
		sum += histOfDst[i]*255;
		lutSumDst->data.ptr[i] = (uchar)cvRound(sum);
	}
	for (int i = 0; i < 256; i++)
	{
		int k = 0;
		while(lutSumDst->data.ptr[k] < i) k++;
		lutEq2Dst->data.ptr[i] = k;
	}
	cvReleaseMat(&lutSumDst);

	m_scale = 255.0/(imgSz.width * imgSz.height);
	m_histsrc = cvCreateHist(1, &histSz, CV_HIST_ARRAY); // src's hist
	m_lutSrc2Eq = cvCreateMat(1, 256, CV_8UC1);

	return true;
}

void CLightPrep::HomographicFilter(CvMat *img)
{
 	cvAddS(img, cvScalar(.01), img);
	cvLog(img, img);
	cvFilter2D(img, img, h);
	cvExp(img, img);
	cvNormalize(img, img, 0,1, CV_MINMAX);
}

void CLightPrep::HistNorm(CvArr *src)
{
	float	*histOfSrc;
	float	sum = 0;
	cvCalcArrHist((CvArr **)&src, m_histsrc, 0, m_mask);
	histOfSrc = (float *)cvPtr1D(m_histsrc->bins, 0);
	for(int i = 0; i < 256; i++)
	{
		sum += histOfSrc[i];
		m_lutSrc2Eq->data.ptr[i] = (uchar)cvRound(sum*m_scale);
	}
	m_lutSrc2Eq->data.ptr[0] = 0;
	cvLUT(src, src, m_lutSrc2Eq);
	cvLUT(src, src, lutEq2Dst);
}
CLightPrep::CLightPrep()
{
	h=NULL;
	lutEq2Dst = NULL;
	m_histsrc= NULL;
	m_lutSrc2Eq= NULL;
	m_mask= NULL;
	m_invMask= NULL;

}
CLightPrep::~CLightPrep()
{
	if(h)cvReleaseMat(&h);
	if(lutEq2Dst)cvReleaseMat(&lutEq2Dst);
	if(m_histsrc)cvReleaseHist(&m_histsrc);
	if(m_lutSrc2Eq)cvReleaseMat(&m_lutSrc2Eq);
	if(m_mask)cvReleaseMat(&m_mask);
	if(m_invMask)cvReleaseMat(&m_invMask);
}

void CLightPrep::MaskFace( CvArr *src )
{
	cvSet(src, cvScalar(0), m_invMask);
}

void CLightPrep::GenFilterKernel(CvMat *h, int filterType, /*bool bHighpass,*/ double d0, int order/*=1*/)
{
	CvSize sz = cvGetSize(h);
	assert(d0 > 0/* && d0 < __min(sz.width/2, sz.height/2)*/);
	assert(order > 0);
	assert(filterType >= 0 && filterType <= 3);

	double	i,j;
	double	r,z;
	for (int y = 0; y < sz.height; y++)
	{
		for (int x = 0; x < sz.width; x++)
		{
			i = x - double(sz.width - 1)/2;
			j = y - double(sz.height - 1)/2;
			r = sqrt(i*i+j*j);
			switch(filterType)
			{
			case 0:
				z = (r < d0);
				break;
			case 1:
				z = exp(-r*r/(2*d0*d0));
				break;
			case 2:
				z = 1/(1+pow(r/d0, 2*filterType));
			}
			//if (bHighpass) z = 1-z; // incorrect for spatial domain
			cvmSet(h, y, x, z);
		}
	}
	CvScalar s = cvSum(h);
	cvScale(h, h, 1./s.val[0]);
}