#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <set>



using namespace cv;
using namespace std;
using namespace samples;

namespace fs = std::filesystem;

struct StereoImgs //struct for stereo image file pairs
{
    vector <Mat> StereoA, StereoB;
    vector <string> PathsA, PathsB;

};

struct CalibrationResults 
{
    Mat cameraMatrix[2], distCoeffs[2];
    Mat R, R1, R2, P1, P2, T, E, F, Q;
    double rms;
    double err;
    bool calibSucces;
};

StereoImgs LoadCalibrationImages(fs::path parentPath, Size size) {
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

            //namedWindow(imgs.PathsB[j], WINDOW_AUTOSIZE); 

            //imshow(imgs.PathsB[j], imgs.StereoB[j]); 

            //k = waitKey(0);
            //++j;
        }
    }


        
    return imgs;
}

CalibrationResults CalibrateCameras(fs::path parentPath, Size CheesBoardSize, float SquareSize, Size ImageSize, int nrCalibrationSamples) {
    CalibrationResults R;
    int i, j = 0, k = 0, nrFrames = 0, key;

    auto imgs = LoadCalibrationImages(parentPath, ImageSize);
    //destroyAllWindows();

    if (imgs.StereoA.size() != imgs.StereoB.size())
    {
        cerr << "Un-even number of images, please load Stereo Pairs only and try again..." << endl;
        R.calibSucces = false;
        return R;
    }

    vector<Point2f> corners[30];
    bool found[2] = { false, false };
    Mat viewGray[2];

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;


    for (i = 0; i < imgs.StereoA.size(); ++i)
    {
        cout << "Finding chessboard corners in StereoA(" << i << ") and StereoB(" << i << ")...";

        found[0] = findChessboardCorners(imgs.StereoA[i], CheesBoardSize, corners[0],
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);


        found[1] = findChessboardCorners(imgs.StereoB[i], CheesBoardSize, corners[1],
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);


        if (found[0] && found[1])
        {
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

    return R;
}

int main(int argc, char* argv[])
{
    int i, j, k, key;
    Size CboardSize = Size(9, 6);
    const float squareSize = 24.f;

    Size imgsSize = Size(640, 480);
    

    string fullTestFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/calibration";
    string smallTestFolder = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/calibrationsmall";

    int fullTestSmples = 13;
    int smallTestSmples = 3;

    auto CalibrateData = CalibrateCameras(fullTestFolder, CboardSize, squareSize, imgsSize, fullTestSmples);
    key = waitKey();

    return 0;
}