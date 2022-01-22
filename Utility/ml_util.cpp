#include <fstream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/full_object_detection.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/opencv.hpp>
#include "env_util.h"

namespace dlib
{
	class shape_predictor;
}

// In a vector of points, find the index of point closest to input point.
static int find_index(const std::vector<cv::Point2f>& points, const cv::Point2f& point)
{
    int min_index = 0;
    double min_distance = cv::norm(points[0] - point);

    for (int i = 1; i < points.size(); i++)
    {
        const double distance = cv::norm(points[i] - point);

        if (distance < min_distance)
        {
            min_index = i;
            min_distance = distance;
        }
    }

    return min_index;
}

void calculate_delaunay_triangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector<std::vector<int>>& delaunay_tri)
{
    // Create an instance of Subdiv2D
    cv::Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
    }

    // Get Delaunay triangulation
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    // Variable to store a triangle ( 3 points )
    std::vector<cv::Point2f> pt(3);

    // Variable to store a triangle as indices from list of points
    std::vector<int> ind(3);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        // The triangle returned by getTriangleList is
        // a list of 6 coordinates of the 3 points in
        // x1, y1, x2, y2, x3, y3 format.
        cv::Vec6f t = triangleList[i];

        // Store triangle as a vector of three points
        pt[0] = cv::Point2f(t[0], t[1]);
        pt[1] = cv::Point2f(t[2], t[3]);
        pt[2] = cv::Point2f(t[4], t[5]);


        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            // Find the index of each vertex in the points list
            for (int j = 0; j < 3; j++)
            {
                ind[j] = find_index(points, pt[j]);
            }
            // Store triangulation as a list of indices
            delaunay_tri.push_back(ind);
        }
    }
}

// Read points corresponding to the faces, stored in text files
std::vector<cv::Point2f> get_saved_points(const std::string& points_file_name)
{
    std::vector<cv::Point2f> points;
    std::ifstream ifs(points_file_name.c_str());

    float x;
    float y;

    if (!ifs)
    {
        std::cout << "Unable to open file\n";
    }

    while (ifs >> x >> y)
    {
        points.emplace_back(x, y);
    }

    return points;
}

std::string remove_extension(const std::string& file)
{
    //Remove extension of image name
    const size_t extension_marker = file.find_last_of(".");
    const std::string file_name = file.substr(0, extension_marker);

    std::string extensionless_name = file_name + ".txt";
    return extensionless_name;
}

bool landmark_file_found(const std::string& file)
{
    //Remove extension of image name
    const std::string file_name = remove_extension(file);
    const std::string landmark_file = file_name + ".txt";

    const std::fstream fs(landmark_file);
    return fs.good();
}

void landmarks_to_points(dlib::full_object_detection& landmarks, std::vector<cv::Point2f>& points)
{
    // Loop over all landmark points
    for (int i = 0; i < landmarks.num_parts(); i++)
    {
        cv::Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
        points.push_back(pt);
    }
}

// Compare dlib rectangle
static bool rect_area_comparator(const dlib::rectangle& r1, const dlib::rectangle& r2)
{
    return r1.area() < r2.area();
}

std::vector<cv::Point2f> get_landmarks(dlib::frontal_face_detector& face_detector, const dlib::shape_predictor& landmark_detector, const cv::Mat& img, const float FACE_DOWNSAMPLE_RATIO = 1)
{
    std::vector<cv::Point2f> points;

    cv::Mat img_small;
    cv::resize(img, img_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);

    // Convert OpenCV image format to Dlib's image format
    const dlib::cv_image<dlib::bgr_pixel> dlib_im(img);
    const dlib::cv_image<dlib::bgr_pixel> dlib_im_small(img_small);

    // Detect faces in the image
    std::vector<dlib::rectangle> face_rects = face_detector(dlib_im_small);

    if (!face_rects.empty())
    {
        // Pick the biggest face
        dlib::rectangle rect = *std::max_element(face_rects.begin(), face_rects.end(), rect_area_comparator);

        const dlib::rectangle scaled_rect
        (
            rect.left() * FACE_DOWNSAMPLE_RATIO,
            rect.top() * FACE_DOWNSAMPLE_RATIO,
            rect.right() * FACE_DOWNSAMPLE_RATIO,
            rect.bottom() * FACE_DOWNSAMPLE_RATIO
        );

        dlib::full_object_detection landmarks = landmark_detector(dlib_im, scaled_rect);
        landmarks_to_points(landmarks, points);
    }

    return points;
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void warp_triangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f> tri1, std::vector<cv::Point2f> tri2)
{
    // Find bounding rectangle for each triangle
    const cv::Rect r1 = cv::boundingRect(tri1);
    cv::Rect r2 = cv::boundingRect(tri2);

    // Crop the input image to the bounding box of input triangle
    cv::Mat img1_cropped;
    img1(r1).copyTo(img1_cropped);

    // Once the bounding boxes are cropped, 
    // the triangle coordinates need to be 
    // adjusted by an offset to reflect the 
    // fact that they are now in a cropped image. 
    // Offset points by left top corner of the respective rectangles
    std::vector<cv::Point2f> tri1_cropped, tri2_cropped;
    std::vector<cv::Point> tri2_cropped_int;

    for (int i = 0; i < 3; i++)
    {
        tri1_cropped.emplace_back(tri1[i].x - r1.x, tri1[i].y - r1.y);
        tri2_cropped.emplace_back(tri2[i].x - r2.x, tri2[i].y - r2.y);

        // fillConvexPoly needs a vector of Point and not Point2f
        tri2_cropped_int.emplace_back(tri2[i].x - r2.x, tri2[i].y - r2.y);
    }

    // Given a pair of triangles, find the affine transform.
    const cv::Mat warp_mat = getAffineTransform(tri1_cropped, tri2_cropped);

    // Apply the Affine Transform just found to the src image
    cv::Mat img2_cropped = cv::Mat::zeros(r2.height, r2.width, img1_cropped.type());
    cv::warpAffine(img1_cropped, img2_cropped, warp_mat, img2_cropped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

    // We are interested in the pixels inside 
    // the triangle and not the entire bounding box. 

    // So we create a triangular mask using fillConvexPoly.
    // This mask has values 1 ( in all three channels ) 
    // inside the triangle and 0 outside.   
    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
    cv::fillConvexPoly(mask, tri2_cropped_int, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

    // Copy triangular region of the rectangular patch to the output image
    cv::multiply(img2_cropped, mask, img2_cropped);
    cv::multiply(img2(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2_cropped;
}

std::vector<cv::Point2f> get_landmark_point_vector(const cv::Mat& img, const std::string& dir, const std::string& filename, dlib::frontal_face_detector fd, const dlib::shape_predictor& pd)
{
    if (landmark_file_found(filename))
    {
        return get_saved_points(util::get_data_path() + dir + remove_extension(filename) + ".txt");
    }
    else
    {
        return get_landmarks(fd, pd, img);
    }
}