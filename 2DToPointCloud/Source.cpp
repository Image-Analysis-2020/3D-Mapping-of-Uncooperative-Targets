#include <iostream>
#include <string>
#include <filesystem>
#include <set>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>



using namespace cv;
using namespace pcl;
using namespace std;
using namespace samples;

namespace fs = std::filesystem;

int ndisparities = 96;
int SADWindowSize = 7;
Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

Mat m_map[2][2];

shared_ptr<visualization::PCLVisualizer> viewer(
    new visualization::PCLVisualizer("3D Viewer"));

int i = 0, j = 0, k = 0, key = 0;

struct StereoImgs //struct for stereo image file pairs
{
    vector <Mat> StereoA, StereoB;
    vector <string> PathsA, PathsB;

};

struct CalibrationResults 
{
    Mat cameraMatrix[2], distCoeffs[2];
    Mat R, R1, R2, P1, P2, T, E, F, Q;
    double rms = 0;
    double err = 0;
    bool calibSucces = false;
};

StereoImgs LoadStereoImages(fs::path parentPath, Size size) {
    StereoImgs imgs;
    
    string StereoA("StereoA"), StereoB("StereoB");

    int i = 0, j = 0, k = 0;

    set<fs::path> sorted_by_name;

    cout << "iterating through files in " << parentPath.string() << endl;
    for (auto &entry : fs::directory_iterator(parentPath)) {
        sorted_by_name.insert(entry.path());
    }

    for (auto &filename : sorted_by_name)
    {
        Mat resize_img;
        cout << "loading file " << filename.string() << "...";

        auto cur_image = imread(filename.string(), IMREAD_COLOR);

        if (cur_image.empty()) {
            cout << "Could not read the image: " << filename.string() << ", skipping image" << endl;
            cin.get(); //wait for any key press
            continue;
        }
        resize(cur_image, resize_img, size);

        if (filename.string().find(StereoA) != string::npos) {
            cout << " success! " << endl;
            imgs.StereoA.push_back(resize_img);
            imgs.PathsA.push_back(filename.string());
            //namedWindow(imgs.PathsA[i], WINDOW_AUTOSIZE); 

            //imshow(imgs.PathsA[i], imgs.StereoA[i]); 

            //k = waitKey(0);
            //++i;
        }
        else
        {
            cout << " success! " << endl;
            imgs.StereoB.push_back(resize_img);
            imgs.PathsB.push_back(filename.string());

 /*           namedWindow(imgs.PathsB[j], WINDOW_AUTOSIZE); 

            imshow(imgs.PathsB[j], imgs.StereoB[j]); 

            k = waitKey(0);
            ++j;*/
        }
    }


        
    return imgs;
}

CalibrationResults CalibrateCameras(fs::path parentPath, Size CheesBoardSize, float SquareSize, Size ImageSize, int nrCalibrationSamples) {
    CalibrationResults R;
    int nrFrames = 0;

    auto imgs = LoadStereoImages(parentPath, ImageSize);
    //destroyAllWindows();

    if (imgs.StereoA.size() != imgs.StereoB.size())
    {
        cerr << "Un-even number of images, please load Stereo Pairs only and try again..." << endl;
        R.calibSucces = false;
        return R;
    }

    vector<Point2f> corners[30];

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;


    for (i = 0; i < imgs.StereoA.size(); ++i)
    {
        bool found[2] = { false, false };

        cout << "Finding chessboard corners in StereoA(" << i << ") and StereoB(" << i << ")...";

        found[0] = findChessboardCorners(imgs.StereoA[i], CheesBoardSize, corners[0],
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);


        found[1] = findChessboardCorners(imgs.StereoB[i], CheesBoardSize, corners[1],
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);


        if (found[0] && found[1])
        {
            Mat viewGray[2];
            cout << "Corners Found!";
            cvtColor(imgs.StereoA[i], viewGray[0], COLOR_BGR2GRAY);
            cornerSubPix(viewGray[0], corners[0], Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            cout << " Drawing chessboard corners StereoA " << "...";
            drawChessboardCorners(imgs.StereoA[i], CheesBoardSize, Mat(corners[0]), found[0]);
            imagePoints[0].push_back(corners[0]);

            cvtColor(imgs.StereoB[i], viewGray[1], COLOR_BGR2GRAY);
            cornerSubPix(viewGray[1], corners[1], Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            cout << "Drawing chessboard corners StereoB " << "...";
            drawChessboardCorners(imgs.StereoB[i], CheesBoardSize, Mat(corners[1]), found[1]);
            cout << "Succes!" << endl;
            imagePoints[1].push_back(corners[1]);

            namedWindow(imgs.PathsA[i].substr(131, 20), WINDOW_AUTOSIZE);
            namedWindow(imgs.PathsB[i].substr(131, 20), WINDOW_AUTOSIZE);

            imshow(imgs.PathsA[i].substr(131, 20), imgs.StereoA[i]);
            imshow(imgs.PathsB[i].substr(131, 20), imgs.StereoB[i]);

            k = waitKey(0);

            ++nrFrames;
        }
        else
        {
            cout << "Corners not found, disregarding pair and moving to next image..." << endl;
        }

        if (nrFrames == nrCalibrationSamples)
        {
            cout << "Sufficient calibration samples captured!" << endl;
            break;
        }
    }

    if (nrFrames < nrCalibrationSamples)
    {
        cerr << "Did no reach required number of calibration frames, exiting calibration program..." << endl;
        R.calibSucces = false;
        return R;
    }

    cout << nrCalibrationSamples << " samples have been succesfully detected!" << endl;

    cout << "Calibrating...";
    objectPoints.resize(nrCalibrationSamples);
    for (i = 0; i < nrCalibrationSamples; i++)
    {
        for (j = 0; j < CheesBoardSize.height; j++)
            for (k = 0; k < CheesBoardSize.width; k++)
                objectPoints[i].push_back(Point3f(j * SquareSize, k * SquareSize, 0));
    }

    R.cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    R.cameraMatrix[1] = Mat::eye(3, 3, CV_64F);

    R.rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        R.cameraMatrix[0], R.distCoeffs[0],
        R.cameraMatrix[1], R.distCoeffs[1],
        ImageSize, R.R, R.T, R.E, R.F,
        CALIB_FIX_ASPECT_RATIO +
        CALIB_ZERO_TANGENT_DIST +
        CALIB_SAME_FOCAL_LENGTH +
        CALIB_RATIONAL_MODEL +
        CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

    cout << "done with RMS error=" << R.rms << endl;

    cout << "Calibration quality check...";

    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for (i = 0; i < nrCalibrationSamples; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];

            imgpt[0] = Mat(imagePoints[0][i]);
            undistortPoints(imgpt[0], imgpt[0], R.cameraMatrix[0], R.distCoeffs[0], Mat(), R.cameraMatrix[0]);
            computeCorrespondEpilines(imgpt[0], 1, R.F, lines[0]);

            imgpt[1] = Mat(imagePoints[1][i]);
            undistortPoints(imgpt[1], imgpt[1], R.cameraMatrix[1], R.distCoeffs[1], Mat(), R.cameraMatrix[1]);
            computeCorrespondEpilines(imgpt[1], 2, R.F, lines[1]);

        for (j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x * lines[1][j][0] +
                imagePoints[0][i][j].y * lines[1][j][1] + lines[1][j][2]) +
                fabs(imagePoints[1][i][j].x * lines[0][j][0] +
                    imagePoints[1][i][j].y * lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " << err / npoints << endl;

    R.err = err / npoints;
    R.calibSucces = true;

    Rect roi1, roi2;

    stereoRectify(R.cameraMatrix[0], R.distCoeffs[0], R.cameraMatrix[1], R.distCoeffs[1],
        ImageSize, R.R, R.T, R.R1, R.R2, R.P1, R.P2, R.Q,
        CALIB_ZERO_DISPARITY, -1, ImageSize, &roi1, &roi2);

    bool isVerticalStereo = fabs(R.P2.at<double>(1, 3)) > fabs(R.P2.at<double>(0, 3));

    initUndistortRectifyMap(R.cameraMatrix[0], R.distCoeffs[0], R.R1, R.P1, ImageSize, CV_16SC2,
        m_map[0][0], m_map[0][1]);

    initUndistortRectifyMap(R.cameraMatrix[1], R.distCoeffs[1], R.R2, R.P2, ImageSize, CV_16SC2,
        m_map[1][0], m_map[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(ImageSize.width, ImageSize.height);
        w = cvRound(ImageSize.width * sf);
        h = cvRound(ImageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(ImageSize.width, ImageSize.height);
        w = cvRound(ImageSize.width * sf);
        h = cvRound(ImageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    for (i = 0; i < 13; i++)
    {
        Mat rimgA, rimgB, cimgA, cimgB;
        remap(imgs.StereoA[i], rimgA, m_map[0][0], m_map[0][1], INTER_LINEAR);

        //cvtColor(rimgA, cimgA, COLOR_GRAY2BGR);

        remap(imgs.StereoB[i], rimgB, m_map[1][0], m_map[1][1], INTER_LINEAR);
        //cvtColor(rimgB, cimgB, COLOR_GRAY2BGR);

        Mat canvasPartA = !isVerticalStereo ? canvas(Rect(w * k, 0, w, h)) : canvas(Rect(0, h * k, w, h));
        Mat canvasPartB = !isVerticalStereo ? canvas(Rect(w * k, 0, w, h)) : canvas(Rect(0, h * k, w, h));

        resize(rimgB, canvasPartA, canvasPartA.size(), 0, 0, INTER_AREA);
        resize(rimgB, canvasPartB, canvasPartB.size(), 0, 0, INTER_AREA);

        Rect vroi1(cvRound(roi1.x * sf), cvRound(roi1.y * sf),
            cvRound(roi1.width * sf), cvRound(roi1.height * sf));
        rectangle(canvasPartA, vroi1, Scalar(0, 0, 255), 3, 8);
        
        Rect vroi2(cvRound(roi2.x * sf), cvRound(roi2.y * sf),
            cvRound(roi2.width * sf), cvRound(roi2.height * sf));
        rectangle(canvasPartB, vroi2, Scalar(0, 0, 255), 3, 8);

        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }

    return R;
}

void create_point_cloud(Mat StereoA_Img, Mat StereoB_Img, CalibrationResults CalibrateData)  {
    cout << "In create_point_cloud" << endl;
    Mat disp, disp8, aux[2], rgb_img, xyz;
    double minVal, maxVal;

    cvtColor(StereoA_Img, aux[0], COLOR_BGR2GRAY,4);
    cvtColor(StereoB_Img, aux[1], COLOR_BGR2GRAY,4);

    cout << "Computing disparity...";
    sbm->compute(aux[0], aux[1], disp);
    cout << "Succes!" << endl;

    rgb_img = StereoA_Img.clone();

    minMaxLoc(disp, &minVal, &maxVal);
    cout << "Min disp: " << minVal << " Max value: " << maxVal << endl;

    disp.convertTo(disp8, CV_8U, 255 / (maxVal - minVal));
    namedWindow("disparity", WINDOW_AUTOSIZE);
    imshow("disparity", disp8);

    key = waitKey();

    reprojectImageTo3D(disp, xyz, CalibrateData.Q, true);

    PointCloud<PointXYZRGB>::Ptr point_cloud_ptr(
        new PointCloud<PointXYZRGB>);

    cout << "Starting Point Cloud...";
    for (i = 0; i < xyz.rows; i++)
    {
        for (j = 0; j < xyz.cols; j++)
        {

            // TODO remove points that have no interest
            if (xyz.at<cv::Point3f>(i, j).z > 1000) continue;
            pcl::PointXYZRGB point;
            point.x = xyz.at<cv::Point3f>(i, j).x;
            point.y = xyz.at<cv::Point3f>(i, j).y;
            point.z = xyz.at<cv::Point3f>(i, j).z;
            // OpenCV is BGR
            point.r = rgb_img.at<cv::Point3i>(i, j).z;
            point.g = rgb_img.at<cv::Point3i>(i, j).y;
            point.b = rgb_img.at<cv::Point3i>(i, j).x;
            point_cloud_ptr->points.push_back(point);
        }
    }

    point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;
    cout << "Succes!" << endl;

    cout << "Showing Point Cloud" << endl;

    visualization::PCLVisualizer::Ptr rgbVis(PointCloud<PointXYZRGB>::ConstPtr point_cloud_ptr);

    visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(point_cloud_ptr);
    viewer->addPointCloud(point_cloud_ptr, "reconstruction");

    key = waitKey();
}

int main(int argc, char* argv[])
{
    Size CboardSize = Size(6, 8);
    const float squareSize = 45.f;

    Size imgsSize = Size(6000/8, 4000/8);
    

    string fullCalibFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/StereoCalib";
    string smallCalibFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/calibrationsmall";

    string smallTestFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/TestSmall";
    
    int fullCalibSmples = 22;
    int smallCalibSmples = 2;

    auto CalibrateData = CalibrateCameras(fullCalibFolder, CboardSize, squareSize, imgsSize, fullCalibSmples);

    auto imgs = LoadStereoImages(smallTestFolder, imgsSize);
    if (imgs.StereoA.size() != imgs.StereoB.size())
    {
        cerr << "Un-even number of images, please load Stereo Pairs only and try again..." << endl;
        return 1;
    }

    for (i = 0; i < imgs.StereoA.size(); ++i)
    {
        Mat aux;
        remap(imgs.StereoA[i], aux, m_map[0][0], m_map[0][1], INTER_LINEAR);
        imgs.StereoA[i] = aux;
        remap(imgs.StereoB[i], aux, m_map[1][0], m_map[1][1], INTER_LINEAR);
        imgs.StereoB[i] = aux;

        namedWindow(imgs.PathsA[i], WINDOW_AUTOSIZE);
        namedWindow(imgs.PathsB[i], WINDOW_AUTOSIZE);

        imshow(imgs.PathsA[i], imgs.StereoA[i]);
        imshow(imgs.PathsB[i], imgs.StereoB[i]);
        k = waitKey(0);
    }

    


    create_point_cloud(imgs.StereoA[0], imgs.StereoB[0], CalibrateData);

    return 0;
}