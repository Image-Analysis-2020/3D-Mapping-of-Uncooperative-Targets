#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>
#include <filesystem>



using namespace cv;
using namespace std;
using namespace samples;

namespace fs = std::filesystem;

vector<Mat> ImageFileGenerator(fs::path parentPath, Size size) {
    vector<Mat> images;
    Mat resize_img;

    cout << "iterating through files in " << parentPath.string();
    for (const auto& entry : fs::directory_iterator(parentPath)) {
        auto cur_path = entry.path().string();

        cout << "loading file " << cur_path << "...";
        auto cur_image = imread(cur_path, IMREAD_GRAYSCALE);

        if (cur_image.empty()) {
            cout << "Could not read the image: " << cur_path << ", skipping image" << endl;
            cin.get(); //wait for any key press
            continue;
        }
        resize(cur_image, resize_img, size);

        cout << " success! " << endl;
        images.push_back(resize_img);
    }

    return images;
}

auto ComputeKeypointsORB(vector<Mat> imgs, int size, int threshold) {
    struct retVects {
        vector <vector<KeyPoint>> KeypointVector;
        vector<Mat> DescriptorVector;
    };

    vector <vector<KeyPoint>> KeypointVector;
    vector<Mat> DescriptorVector;

    KeypointVector.resize(size);
    DescriptorVector.resize(size);

    Ptr<Feature2D> orb = ORB::create();
    
    for (int i = 0; i < size; i++) {
        cout << "Image size :" << imgs[i].rows << " " << imgs[i].cols << "\n";
        orb->detectAndCompute(
            imgs[i], 
            Mat(),
            KeypointVector[i], 
            DescriptorVector[i]);
        //KeyPointsFilter::retainBest(KeypointVector[i], threshold);
    }

    return retVects{ KeypointVector, DescriptorVector};
}

//void ShowKeyPoints(vector<Mat> imgs, vector <vector<KeyPoint>> KeypointVector) {
//    for (int i = 0; i < KeypointVector.size(); i++) {
//        drawKeypoints(imgs[i], KeypointVector[i], imgs[i], Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//        namedWindow(KeypointVector., WINDOW_NORMAL);
//        imshow("keypoints1", imgs[i]);
//    }
//}


int main()
{

    string  image_path1 = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/Session2/test1.jpg";
    string  image_path2 = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/Session2/test2.jpg";

    Size size(1082, 1629);
    //auto imgs = ImageFileGenerator("C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/Session2", size);

    vector<Mat> imgs;
    imgs.resize(2);
    vector<Mat> imgsRe;
    imgsRe.resize(2);

    imgsRe[0] = imread(image_path1, IMREAD_GRAYSCALE);
    imgsRe[1] = imread(image_path2, IMREAD_GRAYSCALE);
    resize(imgsRe[0], imgs[0], size);
    resize(imgsRe[1], imgs[1], size);

    //if (img1.empty())
    //{
    //    cout << "Could not read the image: " << image_path1 << endl;
    //    cin.get(); //wait for any key press
    //    return -1;
    //}
    //if (img2.empty())
    //{
    //    cout << "Could not read the image: " << image_path2 << endl;
    //    cin.get(); //wait for any key press
    //    return -1;
    //}

    //String windowName = "Lunar Surface"; //Name of the window

    //namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window

    namedWindow("luna1", WINDOW_NORMAL);
    imshow("luna1", imgs[0]); // Show our image inside the created window.
    namedWindow("luna2", WINDOW_NORMAL);
    imshow("luna2", imgs[1]); // Show our image inside the created window.

    int k = waitKey(0); // Wait for any keystroke in the window

    auto [KeypointVector, DescriptorVector] = ComputeKeypointsORB(imgs, 2, 3500);
    
    drawKeypoints(imgs[0], KeypointVector[0], imgs[0],Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("keypoints1", WINDOW_NORMAL);
    imshow("keypoints1", imgs[0]);

    drawKeypoints(imgs[1], KeypointVector[1], imgs[1], Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("keypoints2", WINDOW_NORMAL);
    imshow("keypoints2", imgs[1]);

    k = waitKey(0);


    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    vector< DMatch > matches;
    matcher->match(DescriptorVector[0], DescriptorVector[1], matches);

    Mat img_matches;
    drawMatches(imgs[0], KeypointVector[0], imgs[1], KeypointVector[1], matches, img_matches);

    imshow("Matches", img_matches);

    k = waitKey(0);

    double max_dist = 0; 
    double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector< DMatch > good_matches;
    vector<Point2f>imgpts1, imgpts2;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance <= max(6 * min_dist, 0.02)) {
            good_matches.push_back(matches[i]);
            imgpts1.push_back(KeypointVector[0][matches[i].queryIdx].pt);
            imgpts2.push_back(KeypointVector[1][matches[i].trainIdx].pt);
        }

    }

    //Mat img_good_matches;
    //drawMatches(imgs[0], KeypointVector[0], imgs[1], KeypointVector[1], good_matches, img_good_matches);

    //imshow("Good Matches", img_matches);

    //k = waitKey(0);

    cout << "finding fundamental Matrix " << "...";
    Mat F = findFundamentalMat(imgpts1, imgpts2, RANSAC, 3., 0.99);   //FM_RANSAC
    cout << " success! " << endl;

    namedWindow("Fundamental Matrix", WINDOW_NORMAL);
    imshow("Fundamental Matrix", F); 
    k = waitKey(0);

    cout << "Rectify uncalibrated points " << "...";
    Mat H1, H2;
    stereoRectifyUncalibrated(imgpts1, imgpts2, F, imgs[0].size(), H1, H2);
    cout << " success! " << endl;

    cout << "Warp image 1 " << "...";
    Mat rectified1(imgs[0].size(), imgs[0].type());
    warpPerspective(imgs[0], rectified1, H1, imgs[0].size());
    cout << " success! " << endl;

    namedWindow("Rectified Image1", WINDOW_NORMAL);
    imshow("Rectified Image1", rectified1);
    k = waitKey(0);

    cout << "Warp image 2 " << "...";
    Mat rectified2(imgs[1].size(), imgs[1].type());
    warpPerspective(imgs[1], rectified2, H2, imgs[1].size());
    cout << " success! " << endl;

    namedWindow("Rectified Image2", WINDOW_NORMAL);
    imshow("Rectified Image2", rectified2);
    k = waitKey(0);

    cout << "Crete StereoSGBM " << "...";
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,    //int minDisparity
        96,     //int numDisparities
        5,      //int SADWindowSize
        600,    //int P1 = 0
        2400,   //int P2 = 0
        20,     //int disp12MaxDiff = 0
        16,     //int preFilterCap = 0
        1,      //int uniquenessRatio = 0
        100,    //int speckleWindowSize = 0
        20,     //int speckleRange = 0
        true);  //bool fullDP = false
    cout << " success! " << endl;

    cout << "Compute StereoSGBM " << "...";
    Mat disp;
    sgbm->compute(rectified1, rectified2, disp);
    cout << " success! " << endl;

    String windowName = "Disparity Map"; //Name of the window

    namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window

    namedWindow(windowName, WINDOW_NORMAL);
    imshow(windowName, disp); // Show our image inside the created window.

    k = waitKey(0);

    destroyAllWindows(); //destroy the created window
    return 0;
}