#include "ml_util.h"

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

    std::string extensionless_name = file_name;
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

std::vector<cv::Point2f> get_landmarks(dlib::frontal_face_detector& face_detector, const dlib::shape_predictor& landmark_detector, const cv::Mat& img, const float face_downsample_ratio = 1)
{
    std::vector<cv::Point2f> points;

    cv::Mat img_small;
    cv::resize(img, img_small, cv::Size(), 1.0 / face_downsample_ratio, 1.0 / face_downsample_ratio);

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
            rect.left() * face_downsample_ratio,
            rect.top() * face_downsample_ratio,
            rect.right() * face_downsample_ratio,
            rect.bottom() * face_downsample_ratio
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

// Constrains points to be inside boundary
void constrain_point(cv::Point2f& p, const cv::Size& sz)
{
    p.x = std::min(std::max(static_cast<double>(p.x), 0.0), static_cast<double>(sz.width - 1));
    p.y = std::min(std::max(static_cast<double>(p.y), 0.0), static_cast<double>(sz.height - 1));
}

void warp_image(cv::Mat& img_in, cv::Mat& img_out, const std::vector<cv::Point2f>& points_in, const std::vector<cv::Point2f>& points_out, const std::vector<std::vector<int>>& delaunay_tri)
{
    // Specify the output image the same size and type as the input image.
    const cv::Size size = img_in.size();
    img_out = cv::Mat::zeros(size, img_in.type());

    // Warp each input triangle to output triangle.
    // The triangulation is specified by delaunayTri
    for (size_t j = 0; j < delaunay_tri.size(); j++)
    {
        // Input and output points corresponding to jth triangle
        std::vector<cv::Point2f> tin, tout;

        for (int k = 0; k < 3; k++)
        {
            // Extract a vertex of input triangle
            cv::Point2f p_in = points_in[delaunay_tri[j][k]];
            // Make sure the vertex is inside the image.
            constrain_point(p_in, size);

            // Extract a vertex of the output triangle
            cv::Point2f p_out = points_out[delaunay_tri[j][k]];
            // Make sure the vertex is inside the image.
            constrain_point(p_out, size);

            // Push the input vertex into input triangle
            tin.push_back(p_in);
            // Push the output vertex into output triangle
            tout.push_back(p_out);
        }

        // Warp pixels inside input triangle to output triangle.  
        warp_triangle(img_in, img_out, tin, tout);
    }
}

void get_eight_boundary_points(const cv::Size& size, std::vector<cv::Point2f>& boundary_pts)
{
    const int h = size.height;
    const int w = size.width;

    boundary_pts.emplace_back(0, 0);
    boundary_pts.emplace_back(w / 2, 0);
    boundary_pts.emplace_back(w - 1, 0);
    boundary_pts.emplace_back(w - 1, h / 2);
    boundary_pts.emplace_back(w - 1, h - 1);
    boundary_pts.emplace_back(w / 2, h - 1);
    boundary_pts.emplace_back(0, h - 1);
    boundary_pts.emplace_back(0, h / 2);
}

cv::Mat correct_colors(const cv::Mat& im1, cv::Mat im2, const std::vector<cv::Point2f>& points2)// lower number --> output is closer to webcam and vice-versa
{
    const cv::Point2f dist_between_eyes = points2[38] - points2[43];
    const float distance = cv::norm(dist_between_eyes);

    //using heuristics to calculate the amount of blur
    int blur_amount = static_cast<int>(0.5 * distance);

    if (blur_amount % 2 == 0)
    {
        blur_amount += 1;
    }

    cv::Mat im1_blur = im1.clone();
    cv::Mat im2_blur = im2.clone();

    cv::blur(im1_blur, im1_blur, cv::Size(blur_amount, blur_amount));
    cv::blur(im2_blur, im2_blur, cv::Size(blur_amount, blur_amount));
    // Avoid divide-by-zero errors.

    im2_blur += 2 * (im2_blur <= 1) / 255;
    im1_blur.convertTo(im1_blur, CV_32F);
    im2_blur.convertTo(im2_blur, CV_32F);
    im2.convertTo(im2, CV_32F);

    cv::Mat ret = im2.clone();
    ret = im2.mul(im1_blur).mul(1 / im2_blur);
    cv::threshold(ret, ret, 255, 255, cv::THRESH_TRUNC);
    ret.convertTo(ret, CV_8UC3);

    return ret;
}

#define GRID 30
#define	IMG_MAX_X	3000
#define	IMG_MAX_Y	2000
#define MAXPOINT	100

// Flag for Map Computation
static MlsMode calc_map = MlsMode::FAST;
// Map points for Projection used in MLSWarpImage
static cv::Point2f ptmap[IMG_MAX_Y / GRID + 1][IMG_MAX_X / GRID + 1];

int mls_projection_single(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, int x, int y, float& tx, float& ty)
{
    float w[MAXPOINT];	// Weights
    float wsum = 0.0;

    //Centroids
    cv::Point2f p_star;
    cv::Point2f q_star;
    cv::Point2f p_hat[MAXPOINT];
    cv::Point2f q_hat[MAXPOINT];

    // Transform Matrix
    cv::Mat a[MAXPOINT];

    // Intermediate matrices for computation
    cv::Mat p(2, 2, CV_32F, 0.0);
    cv::Mat v(2, 2, CV_32F, 0.0);
    cv::Mat vt(2, 2, CV_32F, 0.0);
    cv::Mat q(1, 2, CV_32F, 0.0);

    // calc weights
    for (int i = 0; i < src.size(); i++)
    {
        w[i] = 1.0 / (pow(x - src[i].x + 0.5, 2) + pow(y - src[i].y + 0.5, 2));
        wsum += w[i];
    }

    // calculate centroids of p,q w.r.t W --> p* and q*
    p_star.x = 0.0; p_star.y = 0.0;
    q_star.x = 0.0; q_star.y = 0.0;

    for (int j = 0; j < src.size(); j++)
    {
        p_star.x += (w[j] * src[j].x);
        p_star.y += (w[j] * src[j].y);
        q_star.x += (w[j] * dst[j].x);
        q_star.y += (w[j] * dst[j].y);
    }

    q_star /= wsum;
    p_star /= wsum;

    // calc phat and qhat -- p^ and q^
    for (int i = 0; i < src.size(); i++)
    {
        p_hat[i].x = src[i].x - p_star.x;
        p_hat[i].y = src[i].y - p_star.y;
        q_hat[i].x = dst[i].x - q_star.x;
        q_hat[i].y = dst[i].y - q_star.y;
    }

    // calc Ai
    for (int i = 0; i < src.size(); i++)
    {
        p.at<float>(0, 0) = p_hat[i].x;
        p.at<float>(0, 1) = p_hat[i].y;
        p.at<float>(1, 0) = p_hat[i].y;
        p.at<float>(1, 1) = -p_hat[i].x;

        v.at<float>(0, 0) = x - p_star.x;
        v.at<float>(0, 1) = y - p_star.y;
        v.at<float>(1, 0) = y - p_star.y;
        v.at<float>(1, 1) = -(x - p_star.x);

        cv::transpose(v, vt);

        a[i] = w[i] * p * vt;
    }

    cv::Mat fr(1, 2, CV_32F, 0.0);
    cv::Mat temp_fr(1, 2, CV_32F, 0.0);

    float len_fr;
    float dist;

    // Calc Fr and |Fr|
    for (int i = 0; i < src.size(); i++)
    {
        q.at<float>(0, 0) = q_hat[i].x;
        q.at<float>(0, 1) = q_hat[i].y;

        temp_fr = q * a[i];

        fr += temp_fr;
    }

    len_fr = sqrt(powf(fr.at<float>(0, 0), 2) + powf(fr.at<float>(0, 1), 2));

    fr /= len_fr;

    // Calc |V - p*|
    dist = sqrt((x - p_star.x) * (x - p_star.x) + (y - p_star.y) * (y - p_star.y));

    tx = dist * fr.at<float>(0, 0) + q_star.x;
    ty = dist * fr.at<float>(0, 1) + q_star.y;

    return 1;
}

int calc_mls(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, const int x_size, const int y_size)
{
    // Create Map for Projection
    if (x_size > IMG_MAX_X || y_size > IMG_MAX_Y)
    {
        printf("x_size or y_size is larger than maximum size (%d,%d)/(%d,%d)\n", x_size, y_size, IMG_MAX_X, IMG_MAX_Y);
        return 0;
    }

    float tx;
    float ty;

    // Project GRID Points
    for (int y = 0; y < y_size / GRID + 2; y++) 
    {
        for (int x = 0; x < x_size / GRID + 2; x++) 
        {
            mls_projection_single(src, dst, x * GRID, y * GRID, tx, ty);
            ptmap[y][x].x = tx;
            ptmap[y][x].y = ty;
        }
    }

    calc_map = MlsMode::SINGLE;

    return 1;
}

int calc_mls(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst)
{
    return calc_mls(src, dst, IMG_MAX_X, IMG_MAX_Y);
}

//
//  Fast estimate MLS Projection Point using Precomputed Map
//  calcMLS() must be called before use.
//
int mls_projection_fast(const int x, const int y, float& tx, float& ty)
{
    if (calc_map == MlsMode::FAST)
    {
        printf("calcMLS() must be called before MLSProjectionFast()\n");
        return 0;
    }

    //unit square
    const cv::Point2f f00 = ptmap[y / GRID][x / GRID];
    const cv::Point2f f01 = ptmap[y / GRID][x / GRID + 1];
    const cv::Point2f f10 = ptmap[y / GRID + 1][x / GRID];
    const cv::Point2f f11 = ptmap[y / GRID + 1][x / GRID + 1];

    //bi-linear interpolation
    const float dx = static_cast<float>(x - GRID * (x / GRID)) / static_cast<float>(GRID);
    const float dy = static_cast<float>(y - GRID * (y / GRID)) / static_cast<float>(GRID);

    tx = (f00.x * (1.0 - dy) + f10.x * dy) * (1.0 - dx) + (f01.x * (1.0 - dy) + f11.x * dy) * dx;
    ty = (f00.y * (1.0 - dy) + f10.y * dy) * (1.0 - dx) + (f01.y * (1.0 - dy) + f11.y * dy) * dx;

    return 1;
}

void mls_warp_image(cv::Mat& src, std::vector<cv::Point2f>& spts, cv::Mat& dst, std::vector<cv::Point2f>& dpts, const MlsMode mode)
{
	float tx;
	float ty;

	//Precompute Map for dpts --> spts
    if (mode == MlsMode::FAST) 
    {
        calc_mls(dpts, spts, dst.cols, dst.rows);
    }

    // Warp Image using MLS + bi-linear interpolation
    for (int y = 0; y < dst.rows; y++)
    {
        for (int x = 0; x < dst.cols; x++)
        {
            if (mode == MlsMode::FAST)
            {
                mls_projection_fast(x, y, tx, ty);
            }
            else //Rarely used
            {
                mls_projection_single(dpts, spts, x, y, tx, ty);
            }

            if (tx < 0 || tx > src.cols - 1 || ty < 0 || ty > src.rows - 1)
            {
                //Out of bounds
                dst.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
            else
            {
	            const int xcoord = static_cast<int>(tx + 0.5);
                const int ycoord = static_cast<int>(ty + 0.5);

                if (xcoord > src.cols - 1 || xcoord < 0 || ycoord > src.rows - 1 || ycoord < 0)
                {
                    dst.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
                else
                {
                    dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(ycoord, xcoord);
                }
            }
        }
    }
}

//LAndmark writer
void write_landmarks_to_file(dlib::full_object_detection& landmarks, const std::string& filename)
{
    // Open file
    std::ofstream ofs;
    ofs.open(filename);

    // Loop over all landmark points
    for (int i = 0; i < landmarks.num_parts(); i++)
    {
        // Print x and y coordinates to file
        ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << '\n';
    }

    // Close file
    ofs.close();
}

void draw_polyline
(
    cv::Mat& img,
    const dlib::full_object_detection& landmarks,
    const int start,
    const int end,
    const bool is_closed = false
)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.emplace_back(landmarks.part(i).x(), landmarks.part(i).y());
    }
    cv::polylines(img, points, is_closed, cv::Scalar(255, 200, 0), 2, 16);

}

// Draw face for the 68-point model.
void render_face(cv::Mat& img, const dlib::full_object_detection& landmarks)
{
    draw_polyline(img, landmarks, 0, 16);           // Jaw line
    draw_polyline(img, landmarks, 17, 21);          // Left eyebrow
    draw_polyline(img, landmarks, 22, 26);          // Right eyebrow
    draw_polyline(img, landmarks, 27, 30);          // Nose bridge
    draw_polyline(img, landmarks, 30, 35, true);    // Lower nose
    draw_polyline(img, landmarks, 36, 41, true);    // Left eye
    draw_polyline(img, landmarks, 42, 47, true);    // Right Eye
    draw_polyline(img, landmarks, 48, 59, true);    // Outer lip
    draw_polyline(img, landmarks, 60, 67, true);    // Inner lip
}

void render_face(
    cv::Mat& img, // Image to draw the points on
    const std::vector<cv::Point2f>& points, // Vector of points
    const cv::Scalar& color, // color points
    const int radius = 3) // Radius of points.
{
    for (int i = 0; i < points.size(); i++)
    {
        cv::circle(img, points[i], radius, color, -1);
    }
}

void face_landmark_writer(const cv::Mat& src, const std::string& res_file)
{
    std::string model_path = util::get_model_path();
    std::string data_path = util::get_data_path();

    // Get the face detector
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

    // The landmark detector is implemented in the shape_predictor class
    dlib::shape_predictor landmark_detector;
    std::string predictor_path(model_path + "dlib_models/shape_predictor_68_face_landmarks.dat");

    //Load image to render
    cv::Mat img = src.clone();

    // Load the landmark model
    dlib::deserialize(predictor_path) >> landmark_detector;

    // landmarks will be stored in results/family_0.txt
    std::string landmarks_basename(util::get_data_path() + "images/landmark_results/data/" + remove_extension(res_file));

    // Convert OpenCV image format to Dlib's image format
    dlib::cv_image<dlib::bgr_pixel> dlib_img(img);

    // Detect faces in the image
    std::vector<dlib::rectangle> face_rects = face_detector(dlib_img);
    std::cout << "Number of faces detected: " << face_rects.size() << '\n';

    // Vector to store landmarks of all detected faces
    std::vector<dlib::full_object_detection> landmarks_all;

    // Loop over all detected face rectangles
    for (int i = 0; i < face_rects.size(); i++)
    {
        // For every face rectangle, run landmarkDetector
        dlib::full_object_detection landmarks = landmark_detector(dlib_img, face_rects[i]);

        // Print number of landmarks
        if (i == 0)
        {
            std::cout << "Number of landmarks : " << landmarks.num_parts() << '\n';
        }

        // Store landmarks for current face
        landmarks_all.push_back(landmarks);

        // Next, we render the outline of the face using detected landmarks.
        // Draw landmarks on face
        render_face(img, landmarks); //Internal function

        // The code below saves the landmarks to results/family_0.txt … results/family_4.txt
        // Write landmarks to disk
        std::stringstream landmarks_filename;
        landmarks_filename << landmarks_basename << "_" << i << ".txt";
        std::cout << "Saving landmarks to " << landmarks_filename.str() << '\n';
        write_landmarks_to_file(landmarks, landmarks_filename.str());
    }

    // Save image
    std::string output_filename(util::get_data_path() + "images/landmark_results/face/" + res_file);
    std::cout << "Saving output image to " << output_filename << '\n';
    cv::imwrite(output_filename, img);
    cv::imshow("Image", img);
    cv::waitKey(5000);
}