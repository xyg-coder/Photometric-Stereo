//#include "StdAfx.h"
#include ".\pfmaccess.h"
#include "string.h"


CPFMAccess::CPFMAccess(void)
{
	m_data = NULL;
	m_width = -1;
	m_height = -1;
}

CPFMAccess::~CPFMAccess(void)
{
	if(m_data!=NULL)
		delete []m_data;
}


bool CPFMAccess::LoadFromFile(char* fn)
{
	FILE * fp = fopen(fn,"rb");
	if(fp == NULL)
		return false;
	int dataStartPos = ReadPFMHead(fp, &m_width, &m_height);
	if(dataStartPos == 0)
	{
		fclose(fp);
		return false;
	}
	fseek(fp,dataStartPos,SEEK_SET);

	if(m_data != NULL)
	{
		delete []m_data;
		m_data = NULL;
	}
	m_data = new float[3*m_width*m_height];
	if(fread(m_data,sizeof(float),3*m_width*m_height,fp) != 3*m_width*m_height)
	{
		delete []m_data;
		m_data =NULL;
		fclose(fp);
		return false;
	}
	
	fclose(fp);
	return true;
}

bool CPFMAccess::SaveToFile(char* fn)
{
	if(m_data == NULL)
		return false;

	FILE * fp = fopen(fn,"wb");
	if(fp == NULL)
		return false;

	//fprintf(fp ,"PF\x0a%d %d\x0a-1.000000\x0a",m_width, m_height);
	fprintf(fp ,"PF\x0a%d\x0a%d\x0a-1.000000\x0a",m_width, m_height); // by zlzhou

	if(fwrite(m_data, sizeof(float), 3*m_width*m_height,fp) != 3*m_width*m_height)
	{
		fclose(fp);
		return false;
	}

	fclose(fp);
	return true;
}


int	CPFMAccess::ReadPFMHead(FILE * pFile, int* width,int* height)
{
	char Header[513];
	fseek(pFile, 0, SEEK_SET);
	int	len = fread(Header, sizeof(char), 512, pFile);
	Header[len] = 0;
	
	if(len > 3)
	{
		Header[len-1] = 0;
		if( (Header[0] == 'P' && Header[1] == 'F') || 
			(Header[0] == 'p' && Header[1] == 'f') )
		{
			char* p = strchr(Header,0xa);
			if(p)
			{

				p++;


				//for the read of pfm file generated from photoshop
				int cx, cy;
				char* end;
				end = strchr(p,0xa);
				end = &(end[1]);
				end = strchr(end,0xa);
				end[0] = 0;
				if(sscanf(p,"%d %d", &cx,&cy)==2)
				{
					*width	= cx;
					*height	= cy;
					p = &end[1];
					end = strchr(p,0xa);
					if(end)
					{
						return (end-Header)+1;
					}

				}


				//for read of other pfm files
// 				end = strchr(p,0xa);
// 				if(end)
// 				{
// 					end[0] = 0;
// 
// 					int	cx,cy;
// 					if(sscanf(p,"%d %d",&cx,&cy) == 2)
// 					{
// 						*width = cx;
// 						*height = cy;
// 						p = &end[1];
// 						end = strchr(p,0xa);
// 						if(end)
// 						{
// 							return (end-Header)+1;
// 						}
// 					}
// 				} //if (end)


			} //if(p)
		} //if(Header)
	}// if (len)
	return 0;
}

bool CPFMAccess::SetSize(int width,int height)
{
	if(m_width*m_height == width*height)
		return true;
	
	if(m_data !=NULL)
		delete m_data;
	m_width = width;
	m_height = height;
	m_data = new float[3*width*height];
	for(int i=0; i<m_width; i++)
	for(int j=0; j<m_height; j++)
	{
		m_data[3*(i*m_height+j)+0]	= 0;
		m_data[3*(i*m_height+j)+1]	= 0;
		m_data[3*(i*m_height+j)+2]	= 0;
	}
	return true;
}

bool CPFMAccess::SetData(float*data)
{
	if(m_width ==-1||m_height ==-1)
		return false;
	memcpy(m_data,data,3*sizeof(float)*m_height*m_width);
	return true;
}

void CPFMAccess::SetPixelValue(int x,int y,const float r, const float g, const float b)
{
	m_data[0+3*x+3*y*m_width] = r;
	m_data[1+3*x+3*y*m_width] = g;
	m_data[2+3*x+3*y*m_width] = b;
}

void CPFMAccess::SetPixelValue(int x,int y,const float*value)
{
	for(int i=0;i<3;i++)
		m_data[i+3*x+3*y*m_width] = value[i];
}

void CPFMAccess::SetPixelRValue(int x, int y, const float r)
{
	m_data[3*x+3*y*m_width] = r;
}

void CPFMAccess::SetPixelGValue(int x, int y, const float g)
{
	m_data[1+3*x+3*y*m_width] = g;
}


void CPFMAccess::SetPixelBValue(int x, int y, const float b)
{
	m_data[2+3*x+3*y*m_width] = b;
}


void CPFMAccess::SetPixelComponentValue(int x, int y, const int component_id, const float v)
{
	if(component_id<0 || component_id>2)
		return;
	m_data[component_id+3*x+3*y*m_width] = v;
}


void CPFMAccess::GetPixelValue(int x,int y,float *value)
{
	if(x<0||x>=m_width||y<0||y>=m_height)
	{
		value[0] = value[1] = value[2] =0.0;
		return;
	}
	for(int i=0;i<3;i++)
		value[i] = m_data[i+3*x+3*y*m_width];
}

void CPFMAccess::GetPixelValue(int x, int y, float* r, float* g, float* b)
{
	if(x<0||x>=m_width||y<0||y>=m_height)
	{
		*r = *g = *b =0.0;
		return;
	}
	*r = m_data[0+3*x+3*y*m_width];
	*g = m_data[1+3*x+3*y*m_width];
	*b = m_data[2+3*x+3*y*m_width];
}

void CPFMAccess::GetPixelRValue(int x, int y, float *value)
{
	if(x<0||x>=m_width||y<0||y>=m_height)
	{
		*value	= 0.0;
		return;
	}
	*value	= m_data[0+3*x+3*y*m_width];
}

void CPFMAccess::GetPixelGValue(int x, int y, float *value)
{
	if(x<0||x>=m_width||y<0||y>=m_height)
	{
		*value	= 0.0;
		return;
	}
	*value	= m_data[1+3*x+3*y*m_width];
}

void CPFMAccess::GetPixelBValue(int x, int y, float *value)
{
	if(x<0||x>=m_width||y<0||y>=m_height)
	{
		*value	= 0.0;
		return;
	}
	*value	= m_data[2+3*x+3*y*m_width];
}

void CPFMAccess::GetPixelComponentValue(int x, int y, int component_id, float *value)
{
	if(x<0||x>=m_width||y<0||y>=m_height)
	{
		*value	= 0.0;
		return;
	}
	if(component_id<0 || component_id>2)
	{
		*value = 0.0;
		return;
	}
	*value	= m_data[component_id+3*x+3*y*m_width];
}



