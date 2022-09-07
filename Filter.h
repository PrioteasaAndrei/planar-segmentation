#pragma once
#include <opencv2/opencv.hpp>
#include <set>

struct Region {
	int id;
	cv::Vec4f medianNormal;
	cv::Vec4f medianPoint;
	int pxNo;
	int starting_offset_in_image;

	std::set<Region*> neighbours;

	bool operator<(const Region& t) const
	{
		return (this->id < t.id);
	}
};


class Filter {
public:
	Filter();
	~Filter();



	static void colorToGrayscale(cv::Mat colorImage);
	static void colorToGrayscale(cv::Vec4b* colorData, int width, int height);
	static void filterColorAverage(cv::Vec4b* colorData, cv::Vec4b* colorProcessedData, int width, int height);
	static void filterDepthGaussian(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height);
	static void filterGrayscaleGaussian(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height);
	static void filterGrayscaleSobel(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height);
	static void filterGrayscaleMedianFilter(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height);
	static void filterDepthByDistance(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, float* depthMeasureData, int width, int height);
	static void filterDepthPrewitt(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height);
	static void filterNormalByDot(cv::Vec4f* normalMeasure, cv::Vec4b* normalProcessedData, int width, int height);
	static float distance3D(cv::Vec4f a, cv::Vec4f b);
	static void computeNormals(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height);
	static void computeNormals5x5Vicinity(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height);
	static void transformNormalsToImage(cv::Vec4f* normalMeasureComputedData, cv::Vec4b* normalImageComputedData, int width, int height);
	static void planarSegmentation(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasure, cv::Vec4b* segmentedData, int width, int height);
	static float my_dot(cv::Vec4f a, cv::Vec4f b);
	static void regions_statistics(std::map<int, Region> regions);
	static void add_pixel_to_region(cv::Vec4f* normalMeasure, cv::Vec4f* pointCloudData, int** region_matrix, int offset_vecin, int offset_curent, Region* regiune_vecin);
	static void mergeRegions(int width, int height, std::map<int, Region> regions, int** region_matrix);
	static void mergeRegionsAux(Region* a, Region* b, std::map<int, Region> regions, int** region_matrix, int width, int height);

};