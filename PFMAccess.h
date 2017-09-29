#pragma once
#include "stdio.h"
#include "stdlib.h"


class CPFMAccess
{
public:
	CPFMAccess(void);
	CPFMAccess(const CPFMAccess&) = delete;
	CPFMAccess operator = (const CPFMAccess&) = delete;
	~CPFMAccess(void);

	bool LoadFromFile(char* fn);
	bool SaveToFile(char* fn);
	float* GetData(){return m_data;}
	int GetWidth(){return m_width;}
	int GetHeight(){return m_height;}
	bool SetSize(int width,int height);
	bool SetData(float* data);
	
	void SetPixelValue(int x,int y,const float * value);
	void SetPixelValue(int x,int y,const float r, const float g, const float b);
	void SetPixelBValue(int x, int y, const float b);
	void SetPixelGValue(int x, int y, const float g);
	void SetPixelRValue(int x, int y, const float r);
	void SetPixelComponentValue(int x, int y, const int component_id, const float v);

	void GetPixelValue(int x, int y, float* r, float* g, float* b);
	void GetPixelValue(int x,int y,float *value);
	void GetPixelValue(float x,float y,float *value);
	void GetPixelRValue(int x, int y, float* value);
	void GetPixelGValue(int x, int y, float* value);
	void GetPixelBValue(int x, int y, float* value);
	void GetPixelComponentValue(int x, int y, int component_id, float* value);

    int ReadPFMHead(FILE * pFile, int* width,int* height);
public:
	float * m_data;
	int m_width;
	int m_height;

};
