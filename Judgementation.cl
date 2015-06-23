__kernel void Judgementation_s(const float H, const float v, const float s, const float* com, const uchar temp, __global uchar* result)
{
	if (((H>com[0]&&H<com[1]) || (H>com[2]&&H<com[3]))&&s>com[4]&&v>com[5])
	{
		result[0] = temp;
	}
	if (H>com[0]&&H<com[6]&&s>com[4]&&v>com[7])
	{
		result[0] = temp;
	}
	if (H>com[8]&&H<com[9]&&s>com[7]&&v>com[7])
	{
		result[0] = temp;
	}
	if (H>com[10]&&H<com[11]&&s>com[5]&&v>com[7])
	{
		result[0] = temp;
	}
}


