#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

#define DESCRIPTER_SIZE 16
#define USE_ORIENTATION true

vector<KeyPoint>        detect_key_point        (const Mat &image);
Mat                     add_blank_both_side     (Mat &image, int expand_size);
vector<KeyPoint>        move_key_point_x_axis   (vector<KeyPoint> &key_point, int move);
Mat                     comput_descriptor       (const Mat &image, vector<KeyPoint> &key_point);
vector<vector<DMatch>>  match_two_image         (const Mat &descriptor1, const Mat &descriptor2);
vector<DMatch>          find_good_match         (vector<vector<DMatch>> &matches);
Mat                     find_homography         (vector<KeyPoint> &key_point1, vector<KeyPoint> &key_point2, vector<DMatch> &good_matches);
Mat                     warp_with_homography    (Mat &im1, Mat &im2, Mat &homogrpy);

vector<KeyPoint> detect_key_point(const Mat &image)
{
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create();

    vector<KeyPoint> key_point;
    detector->detect(image, key_point);

    return key_point;
}

Mat add_blank_both_side(Mat &image, int expand_size)
{
    Mat expaned_image;
    copyMakeBorder(image, expaned_image, 0, 0, expand_size, expand_size, BORDER_CONSTANT, Scalar(0));
    return expaned_image;
}

vector<KeyPoint> move_key_point_x_axis(vector<KeyPoint> &key_point, int move)
{
    vector<KeyPoint> key_point_moved(key_point);

    for(size_t i = 0; i < key_point.size(); i++)
    {
        key_point_moved[i].pt.x = key_point_moved[i].pt.x + move;
    }

    return key_point_moved;
}

Mat comput_descriptor(const Mat &image, vector<KeyPoint> &key_point)
{
    //Ptr<xfeatures2d::BriefDescriptorExtractor> descriptor_extractor = xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTER_SIZE, USE_ORIENTATION);
    Ptr<xfeatures2d::SURF> descriptor_extractor = xfeatures2d::SURF::create();

    Mat decriptors;
    descriptor_extractor->compute(image, key_point, decriptors);

    return decriptors;
}

vector<vector<DMatch>> match_two_image(const Mat &descriptor1, const Mat &descriptor2)
{
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

    return knn_matches;
}

vector<DMatch> find_good_match(vector<vector<DMatch>> &matches)
{
    const float ratio_thresh = 0.4f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    return good_matches;
}

Mat find_homography(vector<KeyPoint> &key_point1, vector<KeyPoint> &key_point2, vector<DMatch> &good_matches)
{
    Mat homogrpy;
    
    std::vector<Point2f> points1, points2;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        points1.push_back(key_point1[good_matches[i].queryIdx].pt);
        points2.push_back(key_point2[good_matches[i].trainIdx].pt);
    }
    
    homogrpy = findHomography(points1, points2, RANSAC);

    return homogrpy;
}

Mat warp_with_homography(Mat &im1, Mat &im2, Mat &homogrpy)
{
    Mat im1_warped;
    im1_warped.copyTo(im1_warped);
    
    Mat homogrpy_cpy;
    homogrpy.copyTo(homogrpy_cpy);
    //homogrpy_cpy.at<double>(0, 2) = homogrpy_cpy.at<double>(0, 2) + im2.cols;

    warpPerspective(im1, im1_warped, homogrpy_cpy, Size(im2.cols, im2.rows));

    return im1_warped;
}

Mat copy_warped_image(Mat &left, Mat &middle, Mat &right)
{
    Mat copied_image;

    Mat left_tmp = left(Rect(0, 0, left.cols/3, left.rows));
    Mat middle_tmp = middle(Rect(middle.cols/3, 0, middle.cols/3, middle.rows));
    Mat right_tmp = right(Rect(right.cols*2/3, 0, middle.cols/3, middle.rows));

    Mat left_middle;
    hconcat(left_tmp, middle_tmp, left_middle);
    hconcat(left_middle, right_tmp, copied_image);

    return copied_image;
}

int main( int argc, char ** argv )
{
    cout << "Read images" << endl;
    Mat im1 = imread("001.jpg", IMREAD_GRAYSCALE);
    Mat im2 = imread("002.jpg", IMREAD_GRAYSCALE);
    Mat im3 = imread("003.jpg", IMREAD_GRAYSCALE);

    cout << "Detect key point" << endl;
    vector<KeyPoint> key_point1 = detect_key_point(im1);
    vector<KeyPoint> key_point2 = detect_key_point(im2);
    vector<KeyPoint> key_point3 = detect_key_point(im3);

    cout << "Expand middle image" << endl;
    Mat im2_expaned = add_blank_both_side(im2, im2.cols);

    cout << "Move matched feature point for expanded middle image" << endl;
    vector<KeyPoint> key_point2_moved = move_key_point_x_axis(key_point2, im2.cols);

    cout << "Calculate descriptor" << endl;
    Mat descriptor1 = comput_descriptor(im1, key_point1);
    Mat descriptor2 = comput_descriptor(im2_expaned, key_point2_moved);
    Mat descriptor3 = comput_descriptor(im3, key_point3);

    cout << "Matching with key point and descriptor" << endl;
    vector<vector<DMatch>> matches_left = match_two_image(descriptor1, descriptor2);
    vector<vector<DMatch>> matches_right = match_two_image(descriptor3, descriptor2);

    cout << "Finde good match point" << endl;
    vector<DMatch> good_matches_left = find_good_match(matches_left);
    vector<DMatch> good_matches_right = find_good_match(matches_right);

    cout << "Find homography" << endl;
    Mat homogrpy1 = find_homography(key_point1, key_point2_moved, good_matches_left);
    Mat homogrpy2 = find_homography(key_point3, key_point2_moved, good_matches_right);

    cout << "Warping with homography" << endl;
    Mat warped_img1 = warp_with_homography(im1, im2_expaned, homogrpy1);
    Mat warped_img2 = warp_with_homography(im3, im2_expaned, homogrpy2);

    cout << "Stitch warped image" << endl;
    Mat stitched_img = copy_warped_image(warped_img1, im2_expaned, warped_img2);
    imshow("stitched_img", stitched_img);

    cout << "Save stitched image" << endl;
    imwrite("output.jpg", stitched_img);

    waitKey(0);
}
