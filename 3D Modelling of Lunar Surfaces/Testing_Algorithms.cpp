#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
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
        KeyPointsFilter::retainBest(KeypointVector[i], threshold);
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

   /* string  image_path1 = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/Session1/DSC_0007-2.jpg";
    Mat img1 = imread(image_path1, IMREAD_GRAYSCALE);
    string  image_path2 = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/Session1/DSC_0008-2.jpg";
    Mat img2 = imread(image_path2, IMREAD_GRAYSCALE);*/

    Size size(1082, 1629);
    auto imgs = ImageFileGenerator("C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/Session2", size);
    
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

    //namedWindow("luna1", WINDOW_NORMAL);
    //imshow("luna1", imgs[1]); // Show our image inside the created window.
    //namedWindow("luna2", WINDOW_NORMAL);
    //imshow("luna2", imgs[2]); // Show our image inside the created window.

    //int k = waitKey(0); // Wait for any keystroke in the window
    //if (k == 's')
    //{
    //    imwrite("starry_night.png", img);
    //}

    auto [KeypointVector, DescriptorVector] = ComputeKeypointsORB(imgs, 2, 2500);
    
    drawKeypoints(imgs[0], KeypointVector[0], imgs[0],Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("keypoints1", WINDOW_NORMAL);
    imshow("keypoints1", imgs[0]);

    drawKeypoints(imgs[1], KeypointVector[1], imgs[1], Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("keypoints2", WINDOW_NORMAL);
    imshow("keypoints2", imgs[1]);

    int k = waitKey(0);

    //Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create();

    //Ptr<DescriptorExtractor> descriptor = ORB::create;
    //Mat descriptors1, descriptors2;
    //descriptor->compute(imgs[0], keypointVector[0], descriptors1);
    //descriptor->compute(imgs[1], keypointVector[1], descriptors2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    vector< DMatch > matches;
    matcher->match(DescriptorVector[0], DescriptorVector[1], matches);

    Mat img_matches;
    drawMatches(imgs[0], KeypointVector[0], imgs[1], KeypointVector[1], matches, img_matches);

    imshow("Matches", img_matches);

    k = waitKey(0);
    
    destroyAllWindows; //destroy the created window
    return 0;
}