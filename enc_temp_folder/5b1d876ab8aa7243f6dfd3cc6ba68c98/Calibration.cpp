#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <filesystem>
#include <set>

using namespace cv;
using namespace std;
using namespace samples;

namespace fs = std::filesystem;
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
    Mat m_map[2][2];
};

StereoImgs LoadStereoImages(fs::path parentPath, Size size) {
    StereoImgs imgs;

    string StereoA("StereoA"), StereoB("StereoB");

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

            //namedWindow(imgs.PathsA[i].substr(131, 20), WINDOW_AUTOSIZE);
            //namedWindow(imgs.PathsB[i].substr(131, 20), WINDOW_AUTOSIZE);

            //imshow(imgs.PathsA[i].substr(131, 20), imgs.StereoA[i]);
            //imshow(imgs.PathsB[i].substr(131, 20), imgs.StereoB[i]);

            //k = waitKey(0);

            ++nrFrames;
        }
        else
        {
            cout << "Corners not found, disregarding pair and moving to next image..." << endl;
        }

        if (nrFrames == nrCalibrationSamples)
        {
            cout << "Sufficient calibration samples captured!" << endl;
            destroyAllWindows();
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

    cout << "Calibrating..."<<endl;
    //We iniciate the vector of vectors objectPoints to contain the coordinates of the points in the physical calibration board, meassured in mm
    //This will force the extrinsic parameters to be in mm too
    objectPoints.resize(nrCalibrationSamples);
    for (i = 0; i < nrCalibrationSamples; i++)
    {
        for (j = 0; j < CheesBoardSize.height; j++)
            for (k = 0; k < CheesBoardSize.width; k++)
                objectPoints[i].push_back(Point3f(j * SquareSize, k * SquareSize, 0));
    }


    //Calibration
    //It is good practice to inicialice the matrices with the known focal leght of the camera
    R.cameraMatrix[0] = (Mat_<float>(3,3) << 491.55, 0, 297.41, 0, 491.18, 201.56, 0, 0, 1);
    R.cameraMatrix[1] = (Mat_<float>(3, 3) << 491.55, 0, 297.41, 0, 491.18, 201.56, 0, 0, 1);

    Mat rot[2];
    Mat trans[2];

    auto rms_l = calibrateCamera(objectPoints, imagePoints[0], ImageSize,
        R.cameraMatrix[0], R.distCoeffs[0], rot[0], trans[0], 
        CALIB_USE_INTRINSIC_GUESS +
        CALIB_FIX_ASPECT_RATIO +
        CALIB_ZERO_TANGENT_DIST +
        CALIB_FIX_K3, 
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6));

    cout << "Values from calibration:" << endl;
    cout << "-Camera 1-" << endl;
    cout << "Camera Matrix:" << endl;
    cout << R.cameraMatrix[0] << endl;

    cout << "Distortion array:" << endl;
    cout << R.distCoeffs[0] << endl;

    cout << "Rotation Matrix:" << endl;
    cout << rot[0] << endl;

    cout << "Translation Matrix:" << endl;
    cout << trans[0] << endl;

    cout << "Camera 1 RMS error =" << rms_l<<endl;

    auto rms_r = calibrateCamera(objectPoints, imagePoints[1], ImageSize,
        R.cameraMatrix[1], R.distCoeffs[1], rot[1], trans[1],
        CALIB_USE_INTRINSIC_GUESS +
        CALIB_FIX_ASPECT_RATIO +
        CALIB_ZERO_TANGENT_DIST +
        CALIB_FIX_K3, 
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6));
    
    cout << "-Camera 2-" << endl;
    cout << "Camera Matrix:" << endl;
    cout << R.cameraMatrix[1] << endl;

    cout << "Distortion array:" << endl;
    cout << R.distCoeffs[1] << endl;

    cout << "Rotation Matrix:" << endl;
    cout << rot[1] << endl;

    cout << "Translation Matrix:" << endl;
    cout << trans[1] << endl;

    cout << "Camera 2 RMS error =" << rms_r << endl;

    R.rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        R.cameraMatrix[0], R.distCoeffs[0],
        R.cameraMatrix[1], R.distCoeffs[1],
        ImageSize, R.R, R.T, R.E, R.F,
        CALIB_USE_INTRINSIC_GUESS + //Use the precalculated camera matrix to make the initial guesses on how to compute the parameters
        CALIB_FIX_ASPECT_RATIO +    //Optimization procedure will vary only fx and fy together keeping the ratio fixed to the value in the input camera matrix
        CALIB_ZERO_TANGENT_DIST +   //Only for cameras with very little tangential distortion, tangential distortion parameters p1 and p2 = 0
        CALIB_FIX_FOCAL_LENGTH +   // Just use the fx fy passed in camera matrix
        CALIB_FIX_K3,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6)); 
    //reprojection error is the sum of the squares of the distances between the computed(projected) locations of the three dimensional points onto the image plane and the actual location of the corresponding points on the original image

    cout << "-Stereo-" << endl;
    cout << "Camera Matrix A:" << endl;
    cout << R.cameraMatrix[0] << endl;
    cout << "Camera Matrix B:" << endl;
    cout << R.cameraMatrix[1] << endl;

    cout << "Distortion array A:" << endl;
    cout << R.distCoeffs[0] << endl;
    cout << "Distortion array B:" << endl;
    cout << R.distCoeffs[1] << endl;

    cout << "Rotation Matrix:" << endl;
    cout << R.R << endl;

    cout << "Translation Matrix:" << endl;
    cout << R.T << endl;

    cout << "Extrinsic Matrix:" << endl;
    cout << R.E << endl;

    cout << "Fundamental Matrix:" << endl;
    cout << R.F << endl;

    cout << "Stereo pair RMS error=" << R.rms << endl;

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
    cout << "Mean Stereo Reprojection Error = " << err / npoints << endl;

    R.err = err / npoints;
    R.calibSucces = true;

    Rect roi1, roi2;

    stereoRectify(R.cameraMatrix[0], R.distCoeffs[0], R.cameraMatrix[1], R.distCoeffs[1],
        ImageSize, R.R, R.T, R.R1, R.R2, R.P1, R.P2, R.Q,
        CALIB_ZERO_DISPARITY, -1, ImageSize, &roi1, &roi2);

    cout << "Values from stereoRectify:" << endl;
    cout << "Rectification xform Camera 1:" << endl;
    cout << R.R1 << endl;
    cout << "Rectification xform Camera 2:" << endl;
    cout << R.R2 << endl;

    cout << "New projection matrix Camera 1:" << endl;
    cout << R.P1 << endl;

    cout << "New projection matrix Camera 2:" << endl;
    cout << R.P2 << endl;

    cout << "Disparity fo depth mapping matrix:" << endl;
    cout << R.Q << endl;

    cout << "Guaranteed valid pixels Camera 1:" << endl;
    cout << roi1 << endl;

    cout << "Guaranteed valid pixels Camera 2:" << endl;
    cout << roi2 << endl;

    bool isVerticalStereo = fabs(R.P2.at<double>(1, 3)) > fabs(R.P2.at<double>(0, 3));

    initUndistortRectifyMap(R.cameraMatrix[0], R.distCoeffs[0], R.R1, R.P1, ImageSize, CV_16SC2,
        R.m_map[0][0], R.m_map[0][1]);

    initUndistortRectifyMap(R.cameraMatrix[1], R.distCoeffs[1], R.R2, R.P2, ImageSize, CV_16SC2,
        R.m_map[1][0], R.m_map[1][1]);

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
        sf = 400. / MAX(ImageSize.width, ImageSize.height);
        w = cvRound(ImageSize.width * sf);
        h = cvRound(ImageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    for (i = 0; i < nrCalibrationSamples; i++)
    {
        Mat rimgA, rimgB, cimgA, cimgB;
        remap(imgs.StereoA[i], rimgA, R.m_map[0][0], R.m_map[0][1], INTER_LINEAR);

        remap(imgs.StereoB[i], rimgB, R.m_map[1][0], R.m_map[1][1], INTER_LINEAR);

        //namedWindow(imgs.PathsA[i].substr(131, 20), WINDOW_AUTOSIZE);
        //namedWindow(imgs.PathsB[i].substr(131, 20), WINDOW_AUTOSIZE);

        //imshow(imgs.PathsA[i].substr(131, 20), rimgA);
        //imshow(imgs.PathsB[i].substr(131, 20), rimgB);

        //k = waitKey(0);

        Mat canvasPartA = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
        Mat canvasPartB = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));

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

        //imshow("rectified", canvas);

        char c = (char)waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }

    return R;
}

int main(int argc, char* argv[])
{
    //Matx31f m31f;
    //Matx33f m33f;
    //m33f = Matx33f::eye();
    //m31f = m33f.diag();

    //cout << m31f.row(0).val[0];
    cv::Size ss;
	Size CboardSize = Size(6, 8);
	const float squareSize = 45.f;

	Size imgsSize = Size(400 , 600);

	string fullCalibFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/StereoCalib";
	string smallCalibFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/calibrationsmall";

	string smallTestFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/Test_photos/TestSmall";

	int fullCalibSmples = 22;
	int smallCalibSmples = 2;

	auto CalibrateData = CalibrateCameras(fullCalibFolder, CboardSize, squareSize, imgsSize, fullCalibSmples);

    String filename = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image_Analysis_with_Microcomputer/Special_Assigment/3D Modelling of Lunar Surfaces/Calibration_Results.xml";

    FileStorage file(filename, FileStorage::WRITE);

    file << "CameraMatrix1" << CalibrateData.cameraMatrix[0];
    file << "Disparity1" << CalibrateData.distCoeffs[0];

    file << "CameraMatrix2" << CalibrateData.cameraMatrix[1];
    file << "Disparity2" << CalibrateData.distCoeffs[1];

    file << "rms" << CalibrateData.rms;
    file << "Mean_Reprojection_Error" << CalibrateData.err;

    file << "T" << CalibrateData.T;
    file << "R" << CalibrateData.R;
    file << "R1" << CalibrateData.R1;
    file << "R2" << CalibrateData.R2;
    file << "P1" << CalibrateData.P1;
    file << "P2" << CalibrateData.P2;
    file << "E" << CalibrateData.E;
    file << "F" << CalibrateData.F;

    file << "Q" << CalibrateData.Q;

    file << "map00" << CalibrateData.m_map[0][0];
    file << "map01" << CalibrateData.m_map[0][1];
    file << "map10" << CalibrateData.m_map[1][0];
    file << "map11" << CalibrateData.m_map[1][1];

    file.release();

    cout << "Write Done." << endl;

}

