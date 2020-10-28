#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace samples;


int main()
{
	string image_path = "C:/Users/gabri/OneDrive/University/Masters/Autumn_2020/Image Analysis with Microcomputer/Special Assigment/Test photos/IMG_4706_a.jpg";
	Mat img = imread(image_path, IMREAD_GRAYSCALE);
    //Mat imgS;
    //resize(img, imgS, Size(1000, 500));
    
    if (img.empty())
    {
        cout << "Could not read the image: " << image_path << endl;
        cin.get(); //wait for any key press
        return -1;
    }
    String windowName = "Lunar Surface"; //Name of the window

    namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window

    imshow(windowName, img); // Show our image inside the created window.

    int k = waitKey(0); // Wait for any keystroke in the window
    //if (k == 's')
    //{
    //    imwrite("starry_night.png", img);
    //}

    cout << "Image size :" << img.rows << " " << img.cols << "\n";

    vector<KeyPoint> keypointsD;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();

    

    detector->detect(img, keypointsD, Mat());
    KeyPointsFilter::retainBest(keypointsD, 2500);
    drawKeypoints(img, keypointsD, img,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints", img);

    k = waitKey(0);

    destroyWindow(windowName); //destroy the created window
    return 0;
}