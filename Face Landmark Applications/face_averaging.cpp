#include <filesystem>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

#include "triangle_warp.h"
#include "env_util.h"
#include "delaunay.h"

#define M_PI 3.1415928

// Compute similarity transform given two sets of two points.
// OpenCV requires 3 pairs of corresponding points.
// We are faking the third one.
void similarity_transform(const std::vector<cv::Point2f>& in_points, const std::vector<cv::Point2f>& out_points, cv::Mat& tform)
{
	const double s60 = sin(60 * M_PI / 180.0);
	const double c60 = cos(60 * M_PI / 180.0);

	std::vector<cv::Point2f> in_pts = in_points;
	std::vector<cv::Point2f> out_pts = out_points;

	// Placeholder for the third point.
	in_pts.emplace_back(0, 0);
	out_pts.emplace_back(0, 0);

	// The third point is calculated so that the three points make an equilateral triangle
	in_pts[2].x = c60 * (in_pts[0].x - in_pts[1].x) - s60 * (in_pts[0].y - in_pts[1].y) + in_pts[1].x;
	in_pts[2].y = s60 * (in_pts[0].x - in_pts[1].x) + c60 * (in_pts[0].y - in_pts[1].y) + in_pts[1].y;

	out_pts[2].x = c60 * (out_pts[0].x - out_pts[1].x) - s60 * (out_pts[0].y - out_pts[1].y) + out_pts[1].x;
	out_pts[2].y = s60 * (out_pts[0].x - out_pts[1].x) + c60 * (out_pts[0].y - out_pts[1].y) + out_pts[1].y;

	// Now we can use estimateAffinePartial2D for calculating the similarity transform.
	tform = cv::estimateAffinePartial2D(in_pts, out_pts);
}

void normalize_images_and_landmarks(const cv::Size& out_size, const cv::Mat& img_in, cv::Mat& img_out, const std::vector<cv::Point2f>& points_in, std::vector<cv::Point2f>& points_out)
{
	const int h = out_size.height;
	const int w = out_size.width;
	
	std::vector<cv::Point2f> eyecorner_src;
	// Get the locations of the left corner of left eye
	eyecorner_src.push_back(points_in[36]);
	// Get the locations of the right corner of right eye
	eyecorner_src.push_back(points_in[45]);


	std::vector<cv::Point2f> eyecorner_dst;
	// Location of the left corner of left eye in normalized image.
	eyecorner_dst.emplace_back(0.3 * w, h / 3);
	// Location of the right corner of right eye in normalized image.
	eyecorner_dst.emplace_back(0.7 * w, h / 3);

	// Calculate similarity transform
	cv::Mat tform;
	similarity_transform(eyecorner_src, eyecorner_dst, tform);

	// Apply similarity transform to input image
	img_out = cv::Mat::zeros(h, w, CV_32FC3);
	cv::warpAffine(img_in, img_out, tform, img_out.size());

	// Apply similarity transform to landmarks
	cv::transform(points_in, points_out, tform);
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
	cv::Size size = img_in.size();
	img_out = cv::Mat::zeros(size, img_in.type());

	// Warp each input triangle to output triangle.
	// The triangulation is specified by delaunay_tri
	for (size_t j = 0; j < delaunay_tri.size(); j++)
	{
		// Input and output points corresponding to jth triangle
		std::vector<cv::Point2f> tin, tout;

		for (int k = 0; k < 3; k++)
		{
			// Extract a vertex of input triangle
			cv::Point2f pIn = points_in[delaunay_tri[j][k]];
			// Make sure the vertex is inside the image.
			constrain_point(pIn, size);

			// Extract a vertex of the output triangle
			cv::Point2f pOut = points_out[delaunay_tri[j][k]];
			// Make sure the vertex is inside the image.
			constrain_point(pOut, size);

			// Push the input vertex into input triangle
			tin.push_back(pIn);
			// Push the output vertex into output triangle
			tout.push_back(pOut);
		}
		// Warp pixels inside input triangle to output triangle.  
		warp_triangle(img_in, img_out, tin, tout);
	}
}

void read_file_names(std::string string, std::vector<std::string>& file_vector)
{
	//Get all filenames in directory
	for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(string))
	{
		file_vector.push_back(entry.path().string());
	}
}

// Read landmark points stored in text files
std::vector<cv::Point2f> get_saved_points(const std::string& points_file_name)
{
	std::vector<cv::Point2f> points;
	std::ifstream ifs(points_file_name.c_str());
	float x;
	float y;

	if (!ifs)
	{
		std::cout << "Unable to open file: " << points_file_name <<'\n';
	}

	while (ifs >> x >> y)
	{
		points.emplace_back(x, y);
	}

	return points;
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

// Compare dlib rectangle
bool rect_area_comparator(const dlib::rectangle& r1, const dlib::rectangle& r2)
{
	return r1.area() < r2.area();
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

std::vector<cv::Point2f> getLandmarks(dlib::frontal_face_detector& face_detector, const dlib::shape_predictor& landmark_detector, const cv::Mat& img, const float FACE_DOWNSAMPLE_RATIO = 1)
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

void face_averaging_main()
{
	// Get the face detector
	dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

	// The landmark detector is implemented in the shape_predictor class
	dlib::shape_predictor landmark_detector;

	// Load the landmark model
	dlib::deserialize(util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat") >> landmark_detector;
	// Directory containing images.
	std::string dir_name = util::get_data_path() + "images/people";

	// Add slash to directory name if missing
	if (!dir_name.empty() && dir_name.back() != '/')
	{
		dir_name += '/';
	}

	// Read images in the directory
	std::vector<std::string> image_names;
	std::vector<std::string> pts_names;
	read_file_names(dir_name, image_names);

	// Exit program if no images are found or if the number of image files does not match with the number of point files
	if (image_names.empty())
	{
		std::cout << "No images found with extension jpg or jpeg\n";
		return;
	}

	// Vector of vector of points for all image landmarks.
	std::vector<std::vector<cv::Point2f>> all_points;

	// Read images and perform landmark detection.
	std::vector<cv::Mat> images;
	for (size_t i = 0; i < image_names.size(); i++)
	{
		cv::Mat img = cv::imread(image_names[i]);
		if (!img.data)
		{
			std::cout << "image " << image_names[i] << " not read properly\n";
		}
		else
		{
			std::vector<cv::Point2f> points = getLandmarks(face_detector, landmark_detector, img);
			//std::vector<cv::Point2f> points = get_saved_points(image_names[i].substr(0, image_names[i].length() - 4) + ".txt");

			if (points.size() > 0)
			{
				all_points.push_back(points);
				img.convertTo(img, CV_32FC3, 1 / 255.0);
				images.push_back(img);
			}
		}
	}

	if (images.empty())
	{
		std::cout << "No images found\n";
	}

	int num_images = images.size();

	// Space for normalized images and points.
	std::vector<cv::Mat> images_norm;
	std::vector<std::vector<cv::Point2f>> points_norm;

	// Space for average landmark points
	std::vector<cv::Point2f> points_avg(all_points[0].size());

	// Dimensions of output image
	cv::Size size(600, 600);

	// 8 Boundary points for Delaunay Triangulation
	std::vector<cv::Point2f> boundary_pts;
	get_eight_boundary_points(size, boundary_pts);

	// Warp images and transform landmarks to output coordinate system,
	// and find average of transformed landmarks.

	for (size_t i = 0; i < images.size(); i++)
	{
		std::vector<cv::Point2f> points = all_points[i];

		cv::Mat img;
		normalize_images_and_landmarks(size, images[i], img, points, points);

		// Calculate average landmark locations
		for (size_t j = 0; j < points.size(); j++)
		{
			points_avg[j] += points[j] * (1.0 / num_images);
		}

		// Append boundary points. Will be used in Delaunay Triangulation
		for (size_t j = 0; j < boundary_pts.size(); j++)
		{
			points.push_back(boundary_pts[j]);
		}

		points_norm.push_back(points);
		images_norm.push_back(img);
	}

	// Append boundary points to average points.
	for (size_t j = 0; j < boundary_pts.size(); j++)
	{
		points_avg.push_back(boundary_pts[j]);
	}

	// Calculate Delaunay triangles
	cv::Rect rect(0, 0, size.width, size.height);
	std::vector<std::vector<int>> dt;
	calculate_delaunay_triangles(rect, points_avg, dt);

	// Space for output image
	cv::Mat output = cv::Mat::zeros(size, CV_32FC3);

	// Warp input images to average image landmarks
	for (size_t i = 0; i < num_images; i++)
	{
		cv::Mat img;
		warp_image(images_norm[i], img, points_norm[i], points_avg, dt);

		// Add image intensities for averaging
		output = output + img;
	}

	// Divide by numImages to get average
	output = output / num_images;

	cv::imshow("Output", output);
	cv::waitKey(5000);
}