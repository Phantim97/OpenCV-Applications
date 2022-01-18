#include <filesystem>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

#include "triangle_warp.h"
#include "env_util.h"
#include "delaunay.h"

constexpr double M_PI = 3.1415928;

// Compute similarity transform given two sets of two points.
// OpenCV requires 3 pairs of corresponding points.
// We are faking the third one.
static void similarity_transform(const std::vector<cv::Point2f>& in_points, const std::vector<cv::Point2f>& out_points, cv::Mat& tform)
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

static void normalize_images_and_landmarks(const cv::Size& out_size, const cv::Mat& img_in, cv::Mat& img_out, const std::vector<cv::Point2f>& points_in, std::vector<cv::Point2f>& points_out)
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
static void constrain_point(cv::Point2f& p, const cv::Size& sz)
{
	p.x = std::min(std::max(static_cast<double>(p.x), 0.0), static_cast<double>(sz.width - 1));
	p.y = std::min(std::max(static_cast<double>(p.y), 0.0), static_cast<double>(sz.height - 1));
}

static void warp_image(cv::Mat& img_in, cv::Mat& img_out, const std::vector<cv::Point2f>& points_in, const std::vector<cv::Point2f>& points_out, const std::vector<std::vector<int>>& delaunay_tri)
{
	// Specify the output image the same size and type as the input image.
	const cv::Size size = img_in.size();
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

static void read_file_names(const std::string& string, std::vector<std::string>& file_vector)
{
	//Get all filenames in directory
	for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(string))
	{
		file_vector.push_back(entry.path().string());
	}
}

// Read landmark points stored in text files
static std::vector<cv::Point2f> get_saved_points(const std::string& points_file_name)
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

static void get_eight_boundary_points(const cv::Size& size, std::vector<cv::Point2f>& boundary_pts)
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
static bool rect_area_comparator(const dlib::rectangle& r1, const dlib::rectangle& r2)
{
	return r1.area() < r2.area();
}

static void landmarks_to_points(dlib::full_object_detection& landmarks, std::vector<cv::Point2f>& points)
{
	// Loop over all landmark points
	for (int i = 0; i < landmarks.num_parts(); i++)
	{
		cv::Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
		points.push_back(pt);
	}
}

static std::vector<cv::Point2f> get_landmarks(dlib::frontal_face_detector& face_detector, const dlib::shape_predictor& landmark_detector, const cv::Mat& img, const float FACE_DOWNSAMPLE_RATIO = 1)
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

static void list_files_in_directory(const std::string& dir_name)
{
	std::vector<std::string> files;
	read_file_names(dir_name, files);

	for (int i = 0; i < files.size(); i++)
	{
		std::cout << files[i] << '\n';
	}
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
	std::string dir_name = util::get_data_path() + "images/people/group_d";

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
			std::vector<cv::Point2f> points = get_landmarks(face_detector, landmark_detector, img);
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

static std::string remove_extension(const std::string& file)
{
	//Remove extension of image name
	const size_t extension_marker = file.find_last_of(".");
	const std::string file_name = file.substr(0, extension_marker);

	std::string extensionless_name = file_name + ".txt";
	return extensionless_name;
}

static bool landmark_file_found(const std::string& file)
{
	//Remove extension of image name
	const std::string file_name = remove_extension(file);
	const std::string landmark_file = file_name + ".txt";

	const std::fstream fs(landmark_file);
	return fs.good();
}

void face_morph_main()
{
	// Get the face detector
	dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

	// The landmark detector is implemented in the shape_predictor class
	dlib::shape_predictor landmark_detector;

	// Load the landmark model
	dlib::deserialize(util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat") >> landmark_detector;

	std::string s1;
	std::string s2;

	std::cout << "Enter Face Image 1: ";
	std::cin >> s1;
	std::cout << "Enter Face Image 2: ";
	std::cin >> s2;

	//Read two images
	cv::Mat img1 = cv::imread(util::get_data_path() + "images/people/" + s1);
	cv::Mat img2 = cv::imread(util::get_data_path() + "images/people/" + s2);

	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2;

	// Detect landmarks in both images.
	if (landmark_file_found(s1))
	{
		points1 = get_saved_points(util::get_data_path() + "images/people" +  remove_extension(s1) + ".txt");
	}
	else
	{
		points1 = get_landmarks(face_detector, landmark_detector, img1);
	}

	if (landmark_file_found(s2))
	{
		 points2 = get_saved_points(util::get_data_path() + "images/people/" + remove_extension(s2) + ".txt");
	}
	else
	{
		points2 = get_landmarks(face_detector, landmark_detector, img2);
	}

	// Convert image to floating point in the range 0 to 1
	img1.convertTo(img1, CV_32FC3, 1 / 255.0);
	img2.convertTo(img2, CV_32FC3, 1 / 255.0);
	// Dimensions of output image
	cv::Size size(600, 600);

	// Variables for storing normalized images.
	cv::Mat img_norm1;
	cv::Mat img_norm2;

	// Normalize image to output coordinates.
	normalize_images_and_landmarks(size, img1, img_norm1, points1, points1);
	normalize_images_and_landmarks(size, img2, img_norm2, points2, points2);

	// Calculate average points. Will be used for Delaunay triangulation.
	std::vector<cv::Point2f> points_avg;

	for (int i = 0; i < points1.size(); i++)
	{
		points_avg.push_back((points1[i] + points2[i]) / 2);
	}

	// 8 Boundary points for Delaunay Triangulation
	std::vector<cv::Point2f> boundary_pts;
	get_eight_boundary_points(size, boundary_pts);

	for (int i = 0; i < boundary_pts.size(); i++)
	{
		points_avg.push_back(boundary_pts[i]);
		points1.push_back(boundary_pts[i]);
		points2.push_back(boundary_pts[i]);
	}

	// Calculate Delaunay triangulation.
	std::vector<std::vector<int>> delaunay_tri;
	calculate_delaunay_triangles(cv::Rect(0, 0, size.width, size.height), points_avg, delaunay_tri);

	// Start animation.
	double alpha = 0;
	bool increase_alpha = true;
	int display_count = 1;

	cv::imshow("Image 1", img1);
	cv::imshow("Image 2", img2);
	cv::waitKey(2500);

	while (true)
	{
		// Compute landmark points based on morphing parameter alpha
		std::vector<cv::Point2f> points;
		for (int i = 0; i < points1.size(); i++)
		{
			cv::Point2f point_morph = (1 - alpha) * points1[i] + alpha * points2[i];
			points.push_back(point_morph);
		}

		// Warp images such that normalized points line up with morphed points.
		cv::Mat img_out1;
		cv::Mat img_out2;
		warp_image(img_norm1, img_out1, points1, points, delaunay_tri);
		warp_image(img_norm2, img_out2, points2, points, delaunay_tri);

		// Blend warped images based on morphing parameter alpha
		cv::Mat img_morph = (1 - alpha) * img_out1 + alpha * img_out2;

		// Keep animating by ensuring alpha stays between 0 and 1.
		if (alpha <= 0 && !increase_alpha)
		{
			increase_alpha = true;
		}

		if (alpha >= 1 && increase_alpha)
		{
			increase_alpha = false;
			break;
		}

		if (increase_alpha)
		{
			alpha += 0.075;
		}
		else
		{
			alpha -= 0.075;
		}

		// First subplot
		cv::imshow("Morph", img_morph);
		cv::waitKey(5000);
		display_count++;
	}
}