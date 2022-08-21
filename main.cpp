///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2022, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/************************************************************
** This sample demonstrates how to read a SVO video file. **
** We use OpenCV to display the video.					   **
*************************************************************/

// ZED include
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "Filter.h"


// Using namespace
using namespace sl;
using namespace std;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");

int main(int argc, char **argv) {

    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    
    static sl::RuntimeParameters runParameters;


    init_parameters.input.setFromSVOFile("C:\\Users\\priot\\Documents\\An 3\\practica vara\\2&3D Cameras\\videos\\zed2indoorhead.svo");
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_units = UNIT::METER;
    runParameters.measure3D_reference_frame = sl::REFERENCE_FRAME::CAMERA;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
   
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    auto resolution = zed.getCameraInformation().camera_configuration.resolution;
    
 
    Mat colorImage(resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat colorImage_ocv = slMat2cvMat(colorImage);
    cv::Vec4b* colorData = (cv::Vec4b*)colorImage_ocv.data;
    cv::Mat colorImageProcessed_ocv = cv::Mat(resolution.height, resolution.width,CV_8UC4);
    cv::Vec4b* colorProcessedData = (cv::Vec4b*)colorImageProcessed_ocv.data;

    Mat depthImage(resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat depthImage_ocv = slMat2cvMat(depthImage);
    cv::Vec4b* depthData = (cv::Vec4b*)depthImage_ocv.data;
    cv::Mat depthImageProcessed_ocv = cv::Mat(resolution.height, resolution.width, CV_8UC4);
    cv::Vec4b* depthProcessedData = (cv::Vec4b*)depthImageProcessed_ocv.data;

    Mat grayscaleImage(resolution, MAT_TYPE::U8_C1, MEM::CPU);
    cv::Mat grayscaleImage_ocv = slMat2cvMat(grayscaleImage);
    uchar* grayscaleData = (uchar*)grayscaleImage_ocv.data;
    cv::Mat grayscaleImageProcessed_ocv = cv::Mat(resolution.height, resolution.width, CV_8UC1);
    uchar* grayscaleProcessedData = (uchar*)grayscaleImageProcessed_ocv.data;

    Mat normalImage(resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat normalImage_ocv = slMat2cvMat(normalImage);
    uchar* normalData = (uchar*)normalImage_ocv.data;
    cv::Mat normalImageProcessed_ocv = cv::Mat(resolution.height, resolution.width, CV_8UC1);
    uchar* normalProcessedData = (uchar*)normalImageProcessed_ocv.data;
    cv::Mat normalImageComputed_ocv = cv::Mat(resolution.height, resolution.width, CV_8UC4);
    cv::Vec4b* normalImageComputedData = (cv::Vec4b*)normalImageComputed_ocv.data;

    Mat depthMeasure(resolution, MAT_TYPE::F32_C1, MEM::CPU);
    cv::Mat depthMeasure_ocv = slMat2cvMat(depthMeasure);
    float* depthMeasureData = (float*)depthMeasure_ocv.data;

    Mat normalMeasure(resolution, MAT_TYPE::F32_C4, MEM::CPU);
    cv::Mat normalMeasure_ocv = slMat2cvMat(normalMeasure);
    cv::Vec4f* normalMeasureData = (cv::Vec4f*)normalMeasure_ocv.data;
    cv::Mat normalMeasureComputed_ocv = cv::Mat(resolution.height, resolution.width, CV_32FC4);
    cv::Vec4f* normalMeasureComputedData = (cv::Vec4f*)normalMeasureComputed_ocv.data;

    Mat pointCloud(resolution, MAT_TYPE::F32_C4, MEM::CPU);
    cv::Mat pointCloud_ocv = slMat2cvMat(pointCloud);
    cv::Vec4f* pointCloudData = (cv::Vec4f*)pointCloud_ocv.data;

    // Setup key, images, times
    char key = ' ';
    cout << " Press 's' to save SVO image as a PNG" << endl;
    cout << " Press 'f' to jump forward in the video" << endl;
    cout << " Press 'b' to jump backward in the video" << endl;
    cout << " Press 'q' to exit..." << endl;

    int svo_frame_rate = zed.getInitParameters().camera_fps;
    int nb_frames = zed.getSVONumberOfFrames();
    print("[Info] SVO contains " +to_string(nb_frames)+" frames");

    // Start SVO playback

     while (key != 'q') {
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {

            // Get the side by side image
            zed.retrieveImage(depthImage, VIEW::DEPTH, MEM::CPU, resolution);
            zed.retrieveImage(colorImage, VIEW::LEFT, MEM::CPU, resolution);
            zed.retrieveImage(grayscaleImage, VIEW::LEFT_GRAY, MEM::CPU, resolution);
            zed.retrieveImage(normalImage, VIEW::NORMALS, MEM::CPU, resolution);
            zed.retrieveMeasure(depthMeasure, MEASURE::DEPTH, MEM::CPU, resolution);
            zed.retrieveMeasure(normalMeasure, MEASURE::NORMALS, MEM::CPU, resolution);
            zed.retrieveMeasure(pointCloud, MEASURE::XYZ, MEM::CPU, resolution);

            //Filter::colorToGrayscale(colorData,(int)resolution.width, (int)resolution.height);
            //Filter::filterColorAverage(colorData, colorProcessedData, (int)resolution.width, (int)resolution.height);
           // Filter::filterDepthByDistance(depthData, depthProcessedData, depthMeasureData, (int)resolution.width, (int)resolution.height);
            //Filter::filterGrayscaleGaussian(grayscaleData, grayscaleProcessedData, (int)resolution.width, (int)resolution.height);
            //Filter::filterGrayscaleSobel(grayscaleData, grayscaleProcessedData, (int)resolution.width, (int)resolution.height);
            //Filter::filterDepthPrewitt(depthData, depthProcessedData, (int)resolution.width, (int)resolution.height);
            //Filter::computeNormals(pointCloudData, normalMeasureComputedData, (int)resolution.width, (int)resolution.height);
            Filter::computeNormals5x5Vicinity(pointCloudData, normalMeasureComputedData, (int)resolution.width, (int)resolution.height);
            //Filter::transformNormalsToImage(normalMeasureComputedData, normalImageComputedData, (int)resolution.width, (int)resolution.height);
            

            // TODO : normal matrix is empty ????????/
            for (int i = 0; i < 100; ++i) {
                printf("x:%f y:%f z:%f\n", normalMeasureComputedData[i][0], normalMeasureComputedData[i][1], normalMeasureComputedData[i][2]);
            }
            
            
            Filter::planarSegmentation(pointCloudData, normalMeasureComputedData, normalImageComputedData, (int)resolution.width, (int)resolution.height);
            int svo_position = zed.getSVOPosition();

            // Display the frame
            //cv::imshow("Depth", depthImage_ocv);
            //cv::imshow("DepthProcessed", depthImageProcessed_ocv);
            cv::imshow("Color", colorImage_ocv);
            //cv::imshow("ColorProcessed", colorImageProcessed_ocv);
            //cv::imshow("Grayscale", grayscaleImage_ocv);
            //cv::imshow("GrayscaleProcessed", grayscaleImageProcessed_ocv);
            //cv::imshow("Normals", normalImage_ocv);
            //cv::imshow("NormalsComputed", normalImageComputed_ocv);
            //cv::imshow("NormalMeasure", normalMeasure_ocv);
            cv::imshow("Planar segmentation map", normalImageComputed_ocv);
            key = cv::waitKey(10);
           

            ProgressBar((float)(svo_position / (float)nb_frames), 30);
        }
        else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            print("SVO end has been reached. Looping back to 0\n");
            zed.setSVOPosition(0);
        }
        else {
            print("Grab ZED : ", returned_state);
            break;
        }
     } 
    zed.close();
    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout <<"[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
