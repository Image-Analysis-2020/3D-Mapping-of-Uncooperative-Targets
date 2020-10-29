#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace samples;

vector<KeyPoint> DetectKeypointsFAST(Mat img) {
    
    cout << "Image size :" << img.rows << " " << img.cols << "\n";
    vector<KeyPoint> keypointsD;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();

    detector->detect(img, keypointsD, Mat());
    KeyPointsFilter::retainBest(keypointsD, 2500);
    return keypointsD;
}

int main()
{
    string  image_path1 = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/IMG_4706.jpg";
    Mat img1 = imread(image_path1, IMREAD_GRAYSCALE);
    string  image_path2 = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/IMG_4707.jpg";
    Mat img2 = imread(image_path2, IMREAD_GRAYSCALE);
    
    if (img1.empty())
    {
        cout << "Could not read the image: " << image_path1 << endl;
        cin.get(); //wait for any key press
        return -1;
    }
    if (img2.empty())
    {
        cout << "Could not read the image: " << image_path2 << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    //String windowName = "Lunar Surface"; //Name of the window

    //namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window

    namedWindow("luna1", WINDOW_NORMAL);
    imshow("luna1", img1); // Show our image inside the created window.
    namedWindow("luna2", WINDOW_NORMAL);
    imshow("luna2", img2); // Show our image inside the created window.

    int k = waitKey(0); // Wait for any keystroke in the window
    //if (k == 's')
    //{
    //    imwrite("starry_night.png", img);
    //}

    vector<KeyPoint> keypointsD1 = DetectKeypointsFAST(img1);
    drawKeypoints(img1, keypointsD1, img1,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("keypoints1", WINDOW_NORMAL);
    imshow("keypoints1", img1);

    vector<KeyPoint> keypointsD2 = DetectKeypointsFAST(img2);
    drawKeypoints(img2, keypointsD2, img2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("keypoints2", WINDOW_NORMAL);
    imshow("keypoints2", img2);

    k = waitKey(0);

    //Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create();

    auto Descriptor = DescriptorExtractor::Algorithm();
    
    destroyAllWindows; //destroy the created window
    return 0;
}