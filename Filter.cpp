#include "Filter.h"
#include "glm/glm.hpp"
#include <fstream>
#include <iostream>


Filter::Filter(){}

Filter::~Filter(){}


void Filter::colorToGrayscale(cv::Mat colorImage) {
	int width = colorImage.cols;
	int height = colorImage.rows;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec4b color = colorImage.at<cv::Vec4b>(y, x);
			uchar gray = (uchar)((float)color[0]*0.114 + (float)color[1]*0.587 + (float)color[2]*0.299);
			colorImage.at<cv::Vec4b>(y, x) = cv::Vec4b(gray, gray, gray, color[3]);
		}
	}
}


void Filter::colorToGrayscale(cv::Vec4b* colorData, int width, int height) {
	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec4b color = colorData[offset];
			uchar gray = (color[0] + color[1] + color[2]) / 3;
			colorData[offset] = cv::Vec4b(gray, gray, gray, color[3]);
			offset++;
		}
	}
}

void Filter::filterColorAverage(cv::Vec4b* colorData, cv::Vec4b* colorProcessedData, int width, int height){
	int offset, offset_neighbor;
	for (int y = 3; y < height - 3; y++)
	{
		for (int x = 3; x < width - 3; x++)
		{
			cv::Vec4f color = cv::Vec4f(0, 0, 0, 0);
			for (int k = -3; k <= 3; k++)
			{
				for (int l = -3; l <= 3; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					cv::Vec4b color_neighbor = colorData[offset_neighbor];
					color += (cv::Vec4f)color_neighbor;
				}
			}

			color /= 49;
			offset = y * width + x;
			colorProcessedData[offset] = cv::Vec4b(color[0], color[1], color[2], color[3]);
			
		}
	}

}

void Filter::filterDepthGaussian(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
}

void Filter::filterGrayscaleGaussian(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	int offset, offset_neighbor;
	float mask[9] = { 1,2,1,2,4,2,1,2,1 };
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float grayscale = 0;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					uchar grayscale_neighbor = grayscaleData[offset_neighbor];
					grayscale += (float)grayscale_neighbor * mask[(k+1)*3 + (l+1)];
				}
			}

			grayscale /= 16;
			offset = y * width + x;
			grayscaleProcessedData[offset] = (uchar)grayscale;
		}
	}
}

void Filter::filterDepthByDistance(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, float* depthMeasureData,int width, int height){
	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float depthInfo = depthMeasureData[offset];
			if (depthInfo < 2)
				depthProcessedData[offset] = cv::Vec4b(255,255,255,0);
			else
				if (depthInfo < 3)
					depthProcessedData[offset] = cv::Vec4b(200, 200, 200, 1);
				else if (depthInfo < 4)
					depthProcessedData[offset] = cv::Vec4b(127, 127, 127, 1);
				else if (depthInfo < 5)
					depthProcessedData[offset] = cv::Vec4b(70,70, 70, 1);
				else
					depthProcessedData[offset] = cv::Vec4b(0, 0, 0, 1);
			offset++;
		}
	}
}

void Filter::filterGrayscaleMedianFilter(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
}


void Filter::filterGrayscaleSobel(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	int offset, offset_neighbor;
	float mask_x[9] = { -1,0,1,-2,0,2,-1,0,1 };
	float mask_y[9] = { -1,-2,-1,0,0,0,1,2,1 };

	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float Gx = 0; 
			float Gy = 0;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					uchar grayscale_neighbor = grayscaleData[offset_neighbor];
					Gx += (float)grayscale_neighbor * mask_x[(k + 1) * 3 + (l + 1)];
					Gy += (float)grayscale_neighbor * mask_y[(k + 1) * 3 + (l + 1)];
				}
			}

			float G = sqrt(Gx * Gx + Gy * Gy);
			//float G = abs(Gx) + abs(Gy);
			offset = y * width + x;
			grayscaleProcessedData[offset] = (uchar)G;
		}
	}
}

void Filter::filterDepthPrewitt(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
	int offset, offset_neighbor;
	float mask_x[9] = { -1,0,1,-1,0,1,-1,0,1 };
	float mask_y[9] = { -1,-1,-1,0,0,0,1,1,1 };

	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float Gx = 0;
			float Gy = 0;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					uchar grayscale_neighbor = depthData[offset_neighbor][0];
					Gx += (float)grayscale_neighbor * mask_x[(k + 1) * 3 + (l + 1)];
					Gy += (float)grayscale_neighbor * mask_y[(k + 1) * 3 + (l + 1)];
				}
			}

			float G = sqrt(Gx * Gx + Gy * Gy);
			//float G = abs(Gx) + abs(Gy);
			offset = y * width + x;
			depthProcessedData[offset] = cv::Vec4b(G, G, G, 0);
		}
	}
}



void Filter::filterNormalByDot(cv::Vec4f* normalMeasure, cv::Vec4b* normalProcessedData, int width, int height) {
	//TODO
	int offset, offset_neighbor_s, offset_neighbor_d, offset_neighbor_ss, offset_neighbor_ds, offset_neighbor_sj, offset_neighbor_dj;
	
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float Gx = 0;
			float Gy = 0;
			
			offset_neighbor_s = y * width + (x - 1);
			//.....
			cv::Vec4f normal_neighbor_s = normalMeasure[offset_neighbor_s];
			cv::Vec4f normal_neighbor_d, normal_neighbor_ss, normal_neighbor_ds, normal_neighbor_sj, normal_neighbor_dj;
				
			//Gx =  (1 - dot(normal_neighbor_ss, normal_neighbor_ds)) + 2 * (1 - dot(normal_neighbor_s, normal_neighbor_d))
			//	+ (1 - dot(normal_neighbor_sj, normal_neighbor_dj));
			//float G = sqrt(Gx * Gx + Gy * Gy);
			float G = (abs(Gx) + abs(Gy))*50;
			offset = y * width + x;
			normalProcessedData[offset] = cv::Vec4b(G, G, G, 0);
		}
	}
}


void Filter::computeNormals(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height)
{
	glm::vec3 p_left_vec, p_right_vec, p_up_vec, p_down_vec;
	cv::Vec4f p_left, p_right, p_up, p_down;
	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;

	int offset;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			offset = y * width + x;
			p_left = pointCloudData[offset - 1];
			p_right = pointCloudData[offset + 1];
			p_up = pointCloudData[offset - width];
			p_down = pointCloudData[offset + width];
			p_left_vec = glm::vec3(p_left[0], p_left[1], p_left[2]);
			p_right_vec = glm::vec3(p_right[0], p_right[1], p_right[2]);
			p_up_vec = glm::vec3(p_up[0], p_up[1], p_up[2]);
			p_down_vec = glm::vec3(p_down[0], p_down[1], p_down[2]);
			vec_horiz = p_right_vec - p_left_vec;
			vec_vert = p_up_vec - p_down_vec;
			normal = glm::cross(vec_horiz, vec_vert);
			if (glm::length(normal) > 0.0001)
				normal = glm::normalize(normal);
			normalMeasureComputedData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}
}


/*
* Functie care ia in considerare pentru media vectorilor
* doar vectorii care sunt vecini
* 
* Vectorii sunt vecini doar daca | depth1 - depth2 | < 0.5
* 
* In final, nu am mai folosit-o pentru ca mergea foarte incet
*/
glm::vec3 correspondingMean(glm::vec3* array, int n) {
	bool* bitmap = (bool*)malloc(n * sizeof(bool));
	
	glm::vec3 mean = glm::vec3(0, 0, 0);
	int counter = 1;
	for (int i = 0; i < n; ++i) {
		if (array[i][2] < 0.5) {
			mean *= counter;
			mean += array[i];
			counter++;
			mean /= counter;
		}
	}

	return mean;
}


float Filter::distance3D(cv::Vec4f a, cv::Vec4f b) {
	return sqrt(std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2) + std::pow(a[2] - b[2], 2));
}


// merge pe normale ca au ||v||=1
float Filter::my_dot(cv::Vec4f a, cv::Vec4f b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void Filter::add_pixel_to_region(cv::Vec4f* normalMeasure, cv::Vec4f* pointCloudData, int* region_matrix, int offset_vecin, int offset_curent, Region *regiune_vecin) {
	cv::Vec4f _3dpoint_curent = pointCloudData[offset_curent];
	cv::Vec4f normala_curenta = normalMeasure[offset_curent];

	region_matrix[offset_curent] = region_matrix[offset_vecin];

	// account for the median 3d point
	regiune_vecin->medianPoint[0] *= regiune_vecin->pxNo;
	regiune_vecin->medianPoint[1] *= regiune_vecin->pxNo;
	regiune_vecin->medianPoint[2] *= regiune_vecin->pxNo;

	// account for the normal
	regiune_vecin->medianNormal[0] *= regiune_vecin->pxNo;
	regiune_vecin->medianNormal[1] *= regiune_vecin->pxNo;
	regiune_vecin->medianNormal[2] *= regiune_vecin->pxNo;


	regiune_vecin->medianPoint[0] += _3dpoint_curent[0];
	regiune_vecin->medianPoint[1] += _3dpoint_curent[1];
	regiune_vecin->medianPoint[2] += _3dpoint_curent[2];


	regiune_vecin->medianNormal[0] += normala_curenta[0];
	regiune_vecin->medianNormal[1] += normala_curenta[1];
	regiune_vecin->medianNormal[2] += normala_curenta[2];

	regiune_vecin->pxNo++;

	regiune_vecin->medianPoint[0] /= regiune_vecin->pxNo;
	regiune_vecin->medianPoint[1] /= regiune_vecin->pxNo;
	regiune_vecin->medianPoint[2] /= regiune_vecin->pxNo;

	regiune_vecin->medianNormal[0] /= regiune_vecin->pxNo;
	regiune_vecin->medianNormal[1] /= regiune_vecin->pxNo;
	regiune_vecin->medianNormal[2] /= regiune_vecin->pxNo;


}

void Filter::regions_statistics(std::map<int, Region> regions) {
	int max_id = -1;
	Region max_region;
	for (auto const &p : regions)
	{
		// printf("\t\t\t%d\n", p.second.pxNo);
		// id boop
		if (p.first > max_id) {
			max_id = p.first;
		}
		
		if (p.second.pxNo > max_region.pxNo) {
			max_region = p.second;
		}
	}

	printf("Number of regions: %d\n", max_id);
	printf("Biggest region has %d\n", max_region.pxNo);
}


void Filter::planarSegmentation(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasure,cv::Vec4b* segmentedData, int width, int height) {
	
	printf("%f\n", normalMeasure[0][0]);

	// 0.9
	float treshold = 0.9;
	int* region_matrix = new int[width * height];

	// safe 
	auto createRegion = [](int id, cv::Vec4f medianNormal, cv::Vec4f medianPoint) {
		Region reg;
		reg.id = id;
		reg.medianNormal = medianNormal;
		reg.medianPoint = medianPoint;
		reg.pxNo = 1;

		return reg;
	};
	

	// safe 
	auto evaluate_cost = [normalMeasure,pointCloudData](int offset_vecin, int offset_curent, Region regiune_vecin) {
		// 1.3
		float pondere1 = 1.3;
		// 0.3
		float pondere2 = 0.3;
		
		float cost = pondere1 * (1 - my_dot(normalMeasure[offset_curent], regiune_vecin.medianNormal)) + pondere2 * distance3D(regiune_vecin.medianPoint, pointCloudData[offset_curent]);
		//printf("Cost: %f\n", cost);
		return cost;
	};

	
	int offset, offset_neighbor_s, offset_neighbor_d, offset_neighbor_ss, offset_neighbor_ds, offset_neighbor_sj, offset_neighbor_dj;

	std::map<int, Region> regions;
	int noRegions = 0;

	regions[noRegions++] = createRegion(noRegions - 1,normalMeasure[0],pointCloudData[0]);
	region_matrix[0] = 0;

	
	for (int i = 1; i < width; ++i) {

		if (isnan(normalMeasure[i][0]) || isnan(normalMeasure[i][1]) || isnan(normalMeasure[i][2]) || isnan(normalMeasure[i][3])) {
			add_pixel_to_region(normalMeasure, pointCloudData, region_matrix, 0, i, &regions[0]);
			continue;
		}

		Region regiune_vecin = regions[region_matrix[i - 1]];
		auto cost = evaluate_cost(i - 1, i, regiune_vecin);
		// printf("Costul este: %f\n", cost);
		
		if (cost < treshold) {
			// add pixel to region

;			add_pixel_to_region(normalMeasure,pointCloudData,region_matrix,i - 1, i, &regiune_vecin);

		}
		else {
			// create new region
			regions[noRegions++] = createRegion(noRegions - 1, normalMeasure[i], pointCloudData[i]);
			region_matrix[i] = noRegions - 1;
			// vecinul din spate
			regions[noRegions - 1].neighbours.push_back(&regiune_vecin);
		}
	}

	for (int i = 1; i < height; ++i) {
		if (isnan(normalMeasure[i][0]) || isnan(normalMeasure[i][1]) || isnan(normalMeasure[i][2]) || isnan(normalMeasure[i][3])) {
			add_pixel_to_region(normalMeasure, pointCloudData, region_matrix, 0, i, &regions[0]);
			continue;
		}

		Region regiune_vecin = regions[region_matrix[(i - 1) * width]];
		auto cost = evaluate_cost((i - 1)*width, i*width, regiune_vecin);
		if (cost < treshold) {
			// add pixel to region

			add_pixel_to_region(normalMeasure, pointCloudData, region_matrix, (i - 1)*width, i*width, &regiune_vecin);
	
		}
		else {
			// create new region
			regions[noRegions++] = createRegion(noRegions - 1, normalMeasure[i*width], pointCloudData[i*width]);
			region_matrix[i*width] = noRegions - 1;
			regions[noRegions - 1].neighbours.push_back(&regiune_vecin);
		}
	}
	
	for (int y = 1; y < height-1; y++)
	{
		for (int x = 1; x < width-1; x++)
		{

			offset = y * width + x;

			if (isnan(normalMeasure[offset][0]) || isnan(normalMeasure[offset][1]) || isnan(normalMeasure[offset][2]) || isnan(normalMeasure[offset][3]) ){
				add_pixel_to_region(normalMeasure, pointCloudData, region_matrix, 0, offset, &regions[0]);
				continue;
			}

			// normalProcessedData[offset] = cv::Vec4b(G, G, G, 0);
			int left = offset - 1;
			int up = offset - width;
			int up_left = offset - width - 1;
			int up_right = offset - width + 1;

			// map < id_regiune , regiune >
			// key : region_matrix[left]
			auto cost_left = evaluate_cost(left, offset, regions[region_matrix[left]]);
			auto cost_up = evaluate_cost(up, offset, regions[region_matrix[up]]);
			auto cost_up_left = evaluate_cost(up_left, offset, regions[region_matrix[up_left]]);
			auto cost_up_right = evaluate_cost(up_right, offset, regions[region_matrix[up_right]]);

			Region region_left = regions[region_matrix[left]];
			Region region_up = regions[region_matrix[up]];
			Region region_up_left = regions[region_matrix[up_left]];
			Region region_up_right = regions[region_matrix[up_right]];

			auto min_cost = cost_left;
			auto corresponding_region = regions[region_matrix[left]];
			auto corresponding_offset = left;

			if (cost_up < min_cost) {
				min_cost = cost_up;
				corresponding_region = regions[region_matrix[up]];
				corresponding_offset = up;
			}

			if (cost_up_left < min_cost) {
				min_cost = cost_up_left;
				corresponding_region = regions[region_matrix[up_left]];
				corresponding_offset = up_left;
			}

			if (cost_up_right < min_cost) {
				min_cost = cost_up_right;
				corresponding_region = regions[region_matrix[up_right]];
				corresponding_offset = up_right;
			}

			if (min_cost < treshold) {
				// adaug la regiunea corespunzatoare
				// printf("expanding existing region %d\n", regions[region_matrix[corresponding_offset]].id);
				// region matrix de ce?
				add_pixel_to_region(normalMeasure, pointCloudData, region_matrix, corresponding_offset, offset, &regions[region_matrix[corresponding_offset]]);
	
			}
			else {
				regions[noRegions++] = createRegion(noRegions - 1, normalMeasure[offset], pointCloudData[offset]);
				region_matrix[offset] = noRegions - 1;
				// adauga vecin stanga sus stanga sus sus sus dreapta
				
				// vecinii trebuie sa fie reciproci
				// trebuie set 
				regions[noRegions - 1].neighbours.push_back(&region_left);
				regions[noRegions - 1].neighbours.push_back(&region_up);
				regions[noRegions - 1].neighbours.push_back(&region_up_left);
				regions[noRegions - 1].neighbours.push_back(&region_up_right);
				
				
			}

		}

	}

	// Try to find out why left wall and floor are the same color

	//mergeRegions(width, height, regions, region_matrix);

	
	for (int y = 1; y < height - 1; y++)
	{
		//printf("\n");
		
		for (int x = 1; x < width - 1; x++)
		{
			offset = y * width + x;
			uchar r = (region_matrix[offset] * 5) % 255;
			uchar g = (region_matrix[offset] * 4) % 255;
			uchar b = (region_matrix[offset] * 9) % 255;
			segmentedData[offset] = cv::Vec4b(r, g, b, 1);
			//printf("%d ", region_matrix[offset]);

		}
	}

	// printf("No regions in image: %d --- No pixels in image: %d ---- Percent of regions: %f %% \n\n\n", noRegions,width*height,noRegions * 1.0 / (width*height) * 100);
	// regions_statistics(regions);
	
}


cv::Vec4f dif(cv::Vec4f a, cv::Vec4f b) {
	cv::Vec4f tbr;
	tbr[0] = a[0] - b[0];
	tbr[1] = a[1] - b[1];
	tbr[2] = a[2] - b[2];
	tbr[3] = a[3] - b[3];

	return tbr;
}

// not safe
// point in plan, normal la plan -> ecuatia planului
cv::Vec4f planeEquationFromPointAndNormal(cv::Vec4f point, cv::Vec4f normal) {
	cv::Vec4f tbr;
	tbr[0] = normal[0];
	tbr[1] = normal[1];
	tbr[2] = normal[2];
	tbr[3] = -1 * (normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2]);

	return tbr;
}


// distance from origin (0,0,0)

float pointToPlaneDistance(cv::Vec4f plane) {
	return abs(plane[3]) / sqrt(pow(plane[0], 2) + pow(plane[1], 2) + pow(plane[2], 2));
}


// merge region b into a

void Filter::mergeRegionsAux(Region* a, Region* b, std::map<int,Region> regions, int *region_matrix, int width, int height) {
	// move everything into a
		
	a->medianNormal = (a->medianNormal * a->pxNo + b->medianNormal * b->pxNo) / (a->pxNo + b->pxNo);
	a->medianPoint = (a->medianPoint * a->pxNo + b->medianPoint * b->pxNo) / (a->pxNo + b->pxNo);
	a->pxNo += b->pxNo;

	// update region mask

	for (int i = 0; i < width * height; ++i) {
		if (region_matrix[i] == b->id) {
			region_matrix[i] = a->id;
		}
	}

	// remove b from map

	regions.erase(b->id);

}

// deque 
// fiecare pixel are un pointer la un int
// cand schimb culorea schimb doar ce e la valoarea pointerului ala
// nu trebuie sa fac flood fill

// posibil sa nu modifice regiunile din cauza pointerilor
// dar macar la final ar trebui sa am un region matrix bun
void Filter::mergeRegions(int width, int height, std::map<int, Region> regions, int* region_matrix) {
	
	float treshold = 100;

	// 1.3
	// 0.4
	// 0.655
	auto evaluate_cost = [](Region a, Region b) {
		float pondere1 = 0.5;
		float pondere2 = 0.5;


		// maybe abs diff
		return pondere1 * (1 - my_dot(a.medianNormal, b.medianNormal))
			+ pondere2 * (pointToPlaneDistance(planeEquationFromPointAndNormal(a.medianPoint, a.medianNormal)) 
				- pointToPlaneDistance(planeEquationFromPointAndNormal(b.medianPoint, b.medianNormal)));
	};
	
	for (auto const& x : regions) {
		// get neighbours
		
		int len = x.second.neighbours.size();
		for (int i = 0; i < len; ++i) {
			float cost = evaluate_cost(x.second, *(x.second.neighbours[i]));
			if (abs(cost - treshold) < 0.001) {
				Region current_region = regions[x.first];
				Region neigh_region = regions[x.second.neighbours[i]->id];
				mergeRegionsAux(&current_region,&neigh_region,regions, region_matrix, width, height);
			}
		}
	}
}


void Filter::computeNormals5x5Vicinity(cv::Vec4f* pointCloudData, cv::Vec4f* myNormalMeasureData, int width, int height)
{
	//TODO compute Normals based on 5x5 vicinity
	//using 5 horizontal vectors and 5 vertical vectors
	//check if the neighbors are close to each other (distance < 30 cm)

	glm::vec3 p_left_vec, p_right_vec, p_up_vec, p_down_vec;

	cv::Vec4f p_left, p_right, p_up, p_down;
	cv::Vec4f p_left_sus_1, p_right_sus_1, p_up_1, p_down_1;
	cv::Vec4f p_left_sus_2, p_right_sus_2, p_up_2, p_down_2;
	cv::Vec4f p_left_jos_1, p_right_jos_1, p_up_1j, p_down_1j;
	cv::Vec4f p_left_jos_2, p_right_jos_2, p_up_2j, p_down_2j;

	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;

	int offset;
	for (int y = 2; y < height - 2; y++)
	{
		for (int x = 2; x < width - 2; x++)
		{
			offset = y * width + x;

			p_left = pointCloudData[offset - 2];
			p_right = pointCloudData[offset + 2];

			p_left_sus_1 = pointCloudData[offset - width - 2];
			p_right_sus_1 = pointCloudData[offset - width + 2];

			p_left_sus_2 = pointCloudData[offset - 2*width - 2];
			p_right_sus_2 = pointCloudData[offset - 2*width + 2];

			p_left_jos_1 = pointCloudData[offset + width - 2];
			p_right_jos_1 = pointCloudData[offset + width + 2];

			p_left_jos_2 = pointCloudData[offset + 2 * width - 2];
			p_right_jos_2 = pointCloudData[offset + 2 * width + 2];

			vec_horiz = glm::vec3(p_right[0] - p_left[0], p_right[1] - p_left[1], p_right[2] - p_left[2]);
			glm::vec3 vec_horiz_sus_1 = glm::vec3(p_right_sus_1[0] - p_left_sus_1[0], p_right_sus_1[1] - p_left_sus_1[1], p_right_sus_1[2] - p_left_sus_1[2]);
			glm::vec3 vec_horiz_sus_2 = glm::vec3(p_right_sus_2[0] - p_left_sus_2[0], p_right_sus_2[1] - p_left_sus_2[1], p_right_sus_2[2] - p_left_sus_2[2]);
			glm::vec3 vec_horiz_jos_1 = glm::vec3(p_right_jos_1[0] - p_left_jos_1[0], p_right_jos_1[1] - p_left_jos_1[1], p_right_jos_1[2] - p_left_jos_1[2]);
			glm::vec3 vec_horiz_jos_2 = glm::vec3(p_right_jos_2[0] - p_left_jos_2[0], p_right_jos_2[1] - p_left_jos_2[1], p_right_jos_2[2] - p_left_jos_2[2]);

			glm::vec3 mean_vec_horiz = glm::vec3((vec_horiz[0] + vec_horiz_sus_1[0] + vec_horiz_sus_2[0] + vec_horiz_jos_1[0] + vec_horiz_jos_2[0]) / 5, (vec_horiz[1] + vec_horiz_sus_1[1] + vec_horiz_sus_2[1] + vec_horiz_jos_1[1] + vec_horiz_jos_2[1]) / 5, (vec_horiz[2] + vec_horiz_sus_1[2] + vec_horiz_sus_2[2] + vec_horiz_jos_1[2] + vec_horiz_jos_2[2]) / 5);
			
			/*glm::vec3* arr_horiz = (glm::vec3*)malloc(5 * sizeof(glm::vec3));
			arr_horiz[0] = vec_horiz;
			arr_horiz[1] = vec_horiz_sus_1;
			arr_horiz[2] = vec_horiz_sus_2;
			arr_horiz[3] = vec_horiz_jos_1;
			arr_horiz[3] = vec_horiz_jos_2;
			*/
			//glm::vec3 mean_vec_horiz = correspondingMean(
			p_up = pointCloudData[offset - 2*width];
			p_down = pointCloudData[offset + 2*width];

			cv::Vec4f p_up_left_1 = pointCloudData[offset - 2*width - 1];
			cv::Vec4f p_down_left_1 = pointCloudData[offset + 2*width - 1];

			cv::Vec4f p_up_left_2 = pointCloudData[offset - 2 * width - 2];
			cv::Vec4f p_down_left_2 = pointCloudData[offset + 2 * width - 2];

			cv::Vec4f p_up_right_1 = pointCloudData[offset - 2 * width + 1];
			cv::Vec4f p_down_right_1 = pointCloudData[offset + 2 * width + 1];

			cv::Vec4f p_up_right_2 = pointCloudData[offset - 2 * width + 2];
			cv::Vec4f p_down_right_2 = pointCloudData[offset + 2 * width + 2];

			vec_vert = glm::vec3(p_up[0] - p_down[0], p_up[1] - p_down[1], p_up[2] - p_down[2]);
			glm::vec3 vec_vert_stanga_1 = glm::vec3(p_up_left_1[0] - p_down_left_1[0], p_up_left_1[1] - p_down_left_1[1], p_up_left_1[2] - p_down_left_1[2]);
			glm::vec3 vec_vert_stanga_2 = glm::vec3(p_up_left_2[0] - p_down_left_2[0], p_up_left_2[1] - p_down_left_2[1], p_up_left_2[2] - p_down_left_2[2]);
			glm::vec3 vec_vert_dreapta_1 = glm::vec3(p_up_right_1[0] - p_down_right_1[0], p_up_right_1[1] - p_down_right_1[1], p_up_right_1[2] - p_down_right_1[2]);
			glm::vec3 vec_vert_dreapta_2 = glm::vec3(p_up_right_2[0] - p_down_right_2[0], p_up_right_2[1] - p_down_right_2[1], p_up_right_2[2] - p_down_right_2[2]);

			glm::vec3 mean_vec_vert = glm::vec3((vec_vert[0] + vec_vert_stanga_1[0] + vec_vert_stanga_2[0] + vec_vert_dreapta_1[0] + vec_vert_dreapta_2[0]) / 5, (vec_vert[1] + vec_vert_stanga_1[1] + vec_vert_stanga_2[1] + vec_vert_dreapta_1[1] + vec_vert_dreapta_2[1]) / 5, (vec_vert[2] + vec_vert_stanga_1[2] + vec_vert_stanga_2[2] + vec_vert_dreapta_1[2] + vec_vert_dreapta_2[2]) / 5);

			normal = glm::cross(mean_vec_horiz, mean_vec_vert);
			if (glm::length(normal) > 0.0001)
				normal = glm::normalize(normal);
			myNormalMeasureData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}
}


void Filter::transformNormalsToImage(cv::Vec4f* normalMeasureComputedData, cv::Vec4b* normalImageComputedData, int width, int height)
{

	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			normalImageComputedData[offset] = cv::Vec4b((abs(normalMeasureComputedData[offset][2]) + 1) / 2 * 255,
				(abs(normalMeasureComputedData[offset][1]) + 1) / 2 * 255,
				(abs(normalMeasureComputedData[offset][0]) + 1) / 2 * 255, 0);
			
			
			/*normalImageComputedData[offset] = cv::Vec4b((normalMeasureComputedData[offset][2] + 1.f) / 2.f * 255,
				(normalMeasureComputedData[offset][1] + 1.f) / 2.f * 255,
				(normalMeasureComputedData[offset][0] + 1.f) / 2.f * 255, 0);*/

			offset++;
		}
	}

}