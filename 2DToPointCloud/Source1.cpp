
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <filesystem>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>


#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace cv;
using namespace viz;
using namespace pcl;
using namespace std;

namespace fs = std::filesystem;

//shared_ptr<visualization::PCLVisualizer> viewer(
//    new visualization::PCLVisualizer("3D Viewer"));

struct StereoImgs //struct for stereo image file pairs
{
    vector <Mat> StereoImg;
    vector <string> Paths;

};

struct CalibrationResults
{
    Mat cameraMatrix[2], distCoeffs[2];
    Mat R, R1, R2, P1, P2, T, E, F, Q;
    double rms = 0;
    double err = 0;
    bool calibSucces = false;
};

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

StereoImgs LoadStereoImages(fs::path parentPath, Size size) {
    StereoImgs imgs;

    int i = 0, j = 0, k = 0;

    set<fs::path> sorted_by_name;

    cout << "iterating through files in " << parentPath.string() << endl;
    for (auto& entry : fs::directory_iterator(parentPath)) {
        sorted_by_name.insert(entry.path());
    }

    for (auto& filename : sorted_by_name)
    {
        Mat resize_img;
        cout << "loading file " << filename.string() << "...";

        auto cur_image = imread(filename.string(), IMREAD_GRAYSCALE);

        if (cur_image.empty()) {
            cout << "Could not read the image: " << filename.string() << ", skipping image" << endl;
            cin.get(); //wait for any key press
            continue;
        }
        resize(cur_image, resize_img, size);

        cout << " success! " << endl;
        imgs.StereoImg.push_back(resize_img);
        imgs.Paths.push_back(filename.string());
        
    }

    return imgs;
}

int main(int argc, char* argv[])
{
    Mat Q, img1r, img2r, disp, vdisp, floatdisp;
   
	Size imageSize(400, 600);
    string fullTestFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/Session2";
    string smallTestFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/TestSmall";

    Mat m_map[2][2];

    String filenamein = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/3D Modelling of Lunar Surfaces/Calibration_Results.xml";
    String filenameout = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/3D Modelling of Lunar Surfaces/PointCloud.xml";

    FileStorage filein(filenamein, FileStorage::READ);

    filein["Q"] >> Q;

    filein["map00"] >> m_map[0][0];
    filein["map01"] >> m_map[0][1];
    filein["map10"] >> m_map[1][0];
    filein["map11"] >> m_map[1][1];

    filein.release();
    int i, k, j, x, y;

    Mat pair;
    pair.create(imageSize.height, imageSize.width * 2, CV_8UC3);

	Ptr<StereoSGBM> stereo = StereoSGBM::create(
		-5, 50, 5, 0, 0,                           //minDisparity, numDisparities, blocksize, P1, P2
		32, 0, 5, 2000, 32,                        //disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange
		StereoSGBM::MODE_HH                         //mode
	);

    auto imgs = LoadStereoImages(smallTestFolder, imageSize);

    //namedWindow(imgs.Paths[1].substr(131, 20), WINDOW_AUTOSIZE);
    //namedWindow(imgs.Paths[0].substr(131, 20), WINDOW_AUTOSIZE);

    //cv::imshow(imgs.Paths[1].substr(131, 20), imgs.StereoImg[1]);
    //cv::imshow(imgs.Paths[0].substr(131, 20), imgs.StereoImg[0]);

    //k = waitKey(0);


    //for (i = 0; i < 43; i++){
        remap(imgs.StereoImg[1], img1r, m_map[0][0], m_map[0][1], INTER_LINEAR);
        remap(imgs.StereoImg[0], img2r, m_map[1][0], m_map[1][1], INTER_LINEAR);

        //namedWindow(imgs.Paths[31].substr(131, 20), WINDOW_AUTOSIZE);
        //namedWindow(imgs.Paths[30].substr(131, 20), WINDOW_AUTOSIZE);
        //cv::imshow(imgs.Paths[31].substr(131, 20), img1r);
        //cv::imshow(imgs.Paths[30].substr(131, 20), img2r);
        //k = waitKey(0);

        stereo->compute(img1r, img2r, disp);
        normalize(disp, vdisp, 0, 256, NORM_MINMAX, CV_8U);
        disp.convertTo(floatdisp, CV_32F, 1. / 16);
        
        //namedWindow("disparity", WINDOW_AUTOSIZE);
        //cv::imshow("disparity", vdisp);
        //k = waitKey();

        Mat part = pair.colRange(0, imageSize.width);
        cvtColor(img1r, part, COLOR_GRAY2BGR);
        part = pair.colRange(imageSize.width, imageSize.width * 2);
        cvtColor(img2r, part, COLOR_GRAY2BGR);

        for (j = 0; j < imageSize.height; j += 16)
            cv::line(
                pair,
                Point(0, j),
                Point(imageSize.width * 2, j),
                Scalar(0, 255, 0)
            );

        //namedWindow("rectified", WINDOW_AUTOSIZE);
        //cv::imshow("rectified", pair);
        //k = waitKey();

        auto type1 = type2str(disp.type());
        cout << "disp Mat is of type: " << type1 << endl;
        auto type2 = type2str(vdisp.type());
        cout << "vdisp Mat is of type: " << type2 << endl;
        auto type3 = type2str(floatdisp.type());
        cout << "floatdisp Mat is of type: " << type3 << endl;

        Mat_<Vec3f> xyz(floatdisp.rows, floatdisp.cols);
        Mat_<double> vec_tmp(4, 1);

        auto type4 = type2str(vec_tmp.type());
        cout << "vec_temp Mat is of type: " << type4 << endl;
        auto type5 = type2str(Q.type());
        cout << "Q Mat is of type: " << type5 << endl;

        cout << Q << endl;

        for (y = 0; y < floatdisp.rows; ++y) {
            for (x = 0; x < floatdisp.cols; ++x) {
                vec_tmp(0) = x; vec_tmp(1) = y; vec_tmp(2) = floatdisp.at<float>(y, x); vec_tmp(3) = 1;
                vec_tmp = Q * vec_tmp;
                vec_tmp /= vec_tmp(3);
                Vec3f& point = xyz.at<Vec3f>(y, x);
                point[0] = vec_tmp(0);
                point[1] = vec_tmp(1);
                point[2] = vec_tmp(2);
            }
        }

        //reprojectImageTo3D(floatdisp, xyz, Q, true, CV_32F);
        
        auto type6 = type2str(xyz.type());
        cout << "Point cloud Mat is of type: "<< type6 << endl;

        FileStorage fileout(filenameout, FileStorage::WRITE);

        fileout << "PointCloud" << xyz;

        fileout.release();

        PointCloud<PointXYZ> point_cloud;
        //point_cloud.width = 416409;
        point_cloud.width = 451587;
        point_cloud.height = 1;
        point_cloud.is_dense = false;
        point_cloud.points.resize(xyz.rows * xyz.cols);
        for (i = 0; i < xyz.rows; i++)
        {
            for (j = 0; j < xyz.cols; j++)
           {
                // TODO remove points that have no interest
                if (xyz.at<Point3f>(i, j).z > 1000) continue;
                PointXYZ point;
                point.x = xyz.at<Point3f>(i, j).x;
                point.y = xyz.at<Point3f>(i, j).y;
                point.z = xyz.at<Point3f>(i, j).z;
                point_cloud.points.push_back(point);
            }
        }

        io::savePCDFileASCII("test_pcd.pcd", point_cloud);
        //PointCloud<PointXYZ>::Ptr point_cloud_ptr(
        //    new PointCloud<PointXYZ>());  
        //point_cloud_ptr->width = 10000;
        //point_cloud_ptr->height = 1;
        //point_cloud_ptr->is_dense = false;
        //point_cloud_ptr->points.resize(xyz.rows * xyz.cols);
        //for (i = 0; i < xyz.rows; i++)
        //{
        //    for (j = 0; j < xyz.cols; j++)
        //   {
        //        // TODO remove points that have no interest
        //        if (xyz.at<Point3f>(i, j).z > 1000) continue;
        //        PointXYZ point;
        //        point.x = xyz.at<Point3f>(i, j).x;
        //        point.y = xyz.at<Point3f>(i, j).y;
        //        point.z = xyz.at<Point3f>(i, j).z;
        //        point_cloud_ptr->points.push_back(point);
        //    }
        //}
        //viewer->setBackgroundColor(0, 0, 0);
        //viewer->addPointCloud<PointXYZ>(point_cloud_ptr, "sample_cloud");
        //viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample_cloud");
        //viewer->addCoordinateSystem(1.0);
        //viewer->initCameraParameters();


        /*Viz3d window;
        window.showWidget("3D Point Cloud", WCoordinateSystem());
        window.showWidget("points", WCloud(xyz, Color::white()));
        window.spin();*/
    //}
    
}