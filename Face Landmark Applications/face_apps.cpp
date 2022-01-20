#include <filesystem>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/gui_widgets.h>
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

//eyes mod portion
cv::Mat barrel(const cv::Mat& src, const float k)
{
	const int w = src.cols;
	const int h = src.rows;

	// Meshgrid of destiation image
	cv::Mat xd = cv::Mat::zeros(src.size(), CV_32F);
	cv::Mat yd = cv::Mat::zeros(src.size(), CV_32F);

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			// Normalize x and y
			const float xu = (static_cast<float>(x) / w) - 0.5;
			const float yu = (static_cast<float>(y) / h) - 0.5;

			// Radial distance from center
			const float r = sqrt(xu * xu + yu * yu);

			// Implementing the following equation
			// dr = k * r * cos(pi*r)
			float dr = k * r * cos(M_PI * r);

			// Outside the maximum radius dr is set to 0
			if (r > 0.5)
			{
				dr = 0;
			}

			// Remember we need to provide inverse mapping to remap
			// Hence the negative sign before dr
			const float rn = r - dr;

			// Applying the distortion on the grid
			// Back to un-normalized coordinates  
			xd.at<float>(y, x) = w * (rn * xu / r + 0.5);
			yd.at<float>(y, x) = h * (rn * yu / r + 0.5);

		}
	}

	// Interpolation of points
	cv::Mat dst;
	cv::remap(src, dst, xd, yd, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	return dst;
}


void bug_eyes()
{
	std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize(model_path) >> pose_model;
	int radius = 30;
	float bulge_amount = 0.5;

	std::string filename = util::get_data_path() + "images/tim.jpg";
	cv::Mat src = cv::imread(filename);
	dlib::cv_image<dlib::bgr_pixel> cimg(src);
	std::vector<dlib::rectangle> faces;

	faces = detector(cimg);

	// Find the pose of each face.
	dlib::full_object_detection landmarks;

	// Find the landmark points using DLIB Facial landmarks detector
	landmarks = pose_model(cimg, faces[0]);
	//std::vector<cv::Point2f> landmark_pts = get_saved_points(util::get_data_path() + "images/tim.txt");

	// Find the roi for left and right Eye
	cv::Rect roi_eye_right ( (landmarks.part(43).x()-radius)
	                       , (landmarks.part(43).y()-radius)
	                       , ( landmarks.part(46).x() - landmarks.part(43).x() + 2*radius )
	                       , ( landmarks.part(47).y() - landmarks.part(43).y() + 2*radius ) );
	cv::Rect roi_eye_left ( (landmarks.part(37).x()-radius)
	                      , (landmarks.part(37).y()-radius)
	                      , ( landmarks.part(40).x() - landmarks.part(37).x() + 2*radius )
	                      , ( landmarks.part(41).y() - landmarks.part(37).y() + 2*radius ) );
	/*cv::Rect roiEyeRight((landmark_pts[43].x - radius)
	                     , (landmark_pts[43].y - radius)
	                     , (landmark_pts[46].x - landmark_pts[43].x + 2 * radius)
	                     , (landmark_pts[47].y - landmark_pts[43].y + 2 * radius));
	cv::Rect roiEyeLeft((landmark_pts[37].x - radius)
	                    , (landmark_pts[37].y - radius)
	                    , (landmark_pts[40].x - landmark_pts[37].x + 2 * radius)
	                    , (landmark_pts[41].y - landmark_pts[37].y + 2 * radius));*/
	// Find the patch and apply the transformation
	cv::Mat eye_region;
	cv::Mat output;
	output = src.clone();

	bulge_amount = 0.5;
	src(roi_eye_left).copyTo(eye_region);
	eye_region = barrel(eye_region, bulge_amount);
	eye_region.copyTo(output(roi_eye_left));
	src(roi_eye_right).copyTo(eye_region);
	eye_region = barrel(eye_region, bulge_amount);
	eye_region.copyTo(output(roi_eye_right));

	cv::imshow("Output", output);
	cv::waitKey(5000);
}

//Pose estimation
#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 10
#define OPENCV_FACE_RENDER

// 3D Model Points of selected landmarks in an arbitrary frame of reference
std::vector<cv::Point3d> get3d_model_points()
{
	std::vector<cv::Point3d> model_points;

	model_points.emplace_back(0.0f, 0.0f, 0.0f); //The first must be (0,0,0) while using POSIT
	model_points.emplace_back(0.0f, -330.0f, -65.0f);
	model_points.emplace_back(-225.0f, 170.0f, -135.0f);
	model_points.emplace_back(225.0f, 170.0f, -135.0f);
	model_points.emplace_back(-150.0f, -150.0f, -125.0f);
	model_points.emplace_back(150.0f, -150.0f, -125.0f);

	return model_points;
}

// 2D landmark points from all landmarks
std::vector<cv::Point2d> get2d_image_points(dlib::full_object_detection& d)
{
	std::vector<cv::Point2d> image_points;
	image_points.emplace_back(d.part(30).x(), d.part(30).y());    // Nose tip
	image_points.emplace_back(d.part(8).x(), d.part(8).y());      // Chin
	image_points.emplace_back(d.part(36).x(), d.part(36).y());    // Left eye left corner
	image_points.emplace_back(d.part(45).x(), d.part(45).y());    // Right eye right corner
	image_points.emplace_back(d.part(48).x(), d.part(48).y());    // Left Mouth corner
	image_points.emplace_back(d.part(54).x(), d.part(54).y());    // Right mouth corner
	return image_points;
}

// Camera Matrix from focal length and focal center
cv::Mat get_camera_matrix(const float focal_length, const cv::Point2d& center)
{
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	return camera_matrix;
}

#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>


// Draw an open or closed polygon between
// start and end indices of full_object_detection
void draw_polyline(cv::Mat& img, const dlib::full_object_detection& landmarks, const int start, const int end, const bool is_closed = false)
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

// Draw points on an image.
// Works for any number of points.
void render_face
(
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

void pose_estimation_main()
{
	try
	{
		// Create a VideoCapture object
		cv::VideoCapture cap(0);
		// Check if OpenCV is able to read feed from camera
		if (!cap.isOpened())
		{
			std::cerr << "Unable to connect to camera\n";
			return;
		}

		// Just a place holder. Actual value calculated after 100 frames.
		double fps = 30.0;
		cv::Mat im;

		// Get first frame and allocate memory.
		cap >> im;
		cv::Mat im_small;
		cv::Mat im_display;

		// Resize image to reduce computations
		cv::resize(im, im_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);
		cv::resize(im, im_display, cv::Size(), 0.5, 0.5);

		cv::Size size = im.size();

		// Load face detection and pose estimation models.
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		dlib::shape_predictor predictor;
		dlib::deserialize("../../common/shape_predictor_68_face_landmarks.dat") >> predictor;

		// initiate the tickCounter
		int count = 0;
		double t = cv::getTickCount();

		// variable to store face rectangles
		std::vector<dlib::rectangle> faces;

		// Grab and process frames until the main window is closed by the user.
		while (true)
		{

			// start tick counter if count is zero
			if (count == 0)
			{
				t = cv::getTickCount();
			}

			// Grab a frame
			cap >> im;

			// Create imSmall by resizing image for face detection
			cv::resize(im, im_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);

			// Change to dlib's image format. No memory is copied.
			dlib::cv_image<dlib::bgr_pixel> cimg_small(im_small);
			dlib::cv_image<dlib::bgr_pixel> cimg(im);

			// Process frames at an interval of SKIP_FRAMES.
			// This value should be set depending on your system hardware
			// and camera fps.
			// To reduce computations, this value should be increased
			if (count % SKIP_FRAMES == 0)
			{
				// Detect faces
				faces = detector(cimg_small);
			}

			// Pose estimation
			std::vector<cv::Point3d> model_points = get3d_model_points();

			// Iterate over faces
			std::vector<dlib::full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
			{
				// Since we ran face detection on a resized image,
				// we will scale up coordinates of face rectangle
				dlib::rectangle r(
					(long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
					(long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
					(long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
					(long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
				);

				// Find face landmarks by providing reactangle for each face
				dlib::full_object_detection shape = predictor(cimg, r);
				shapes.push_back(shape);

				// Draw landmarks over face
				render_face(im, shape);

				// get 2D landmarks from Dlib's shape object
				std::vector<cv::Point2d> image_points = get2d_image_points(shape);

				// Camera parameters
				double focal_length = im.cols;
				cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(im.cols / 2, im.rows / 2));

				// Assume no lens distortion
				cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

				// calculate rotation and translation vector using solvePnP
				cv::Mat rotation_vector;
				cv::Mat translation_vector;
				cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

				// Project a 3D point (0, 0, 1000.0) onto the image plane.
				// We use this to draw a line sticking out of the nose
				std::vector<cv::Point3d> nose_end_point_3d;
				std::vector<cv::Point2d> nose_end_point_2d;
				nose_end_point_3d.emplace_back(0, 0, 1000.0);
				cv::projectPoints(nose_end_point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point_2d);

				// draw line between nose points in image and 3D nose points
				// projected to image plane
				cv::line(im, image_points[0], nose_end_point_2d[0], cv::Scalar(255, 0, 0), 2);

			}

			// Print actual FPS
			cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);

			// Display it all on the screen

			// Resize image for display
			im_display = im;
			cv::resize(im, im_display, cv::Size(), 0.5, 0.5);
			cv::imshow("webcam Head Pose", im_display);

			// WaitKey slows down the runtime quite a lot
			// So check every 15 frames
			if (count % 15 == 0)
			{
				int k = cv::waitKey(1);
				// Quit if 'q' or ESC is pressed
				if (k == 'q' || k == 27)
				{
					break;
				}
			}

			// Calculate actual fps
			// increment frame counter
			count++;
			// calculate fps at an interval of 100 frames
			if (count == 100)
			{
				t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
				fps = 100.0 / t;
				count = 0;
			}
		}
	}
	catch (dlib::serialization_error& e)
	{
		std::cout << "Shape predictor model file not found\n";
		std::cout << "Put shape_predictor_68_face_landmarks in models directory\n";
		std::cout << '\n' << e.what() << '\n';
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
	}
}