#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <opencv2/core/types_c.h>

#include "dirent.h"
#include "env_util.h"
#include "ml_util.h"

#define faceWidth 64
#define faceHeight 64

#define PI 3.14159265
#define RESIZE_HEIGHT 480

#define VIDEO 0

// Specify which model to use
// 'l' = LBPH
// 'f' = Fisher
// 'e' = Eigen
#define MODEL 'l'

cv::Mat get_cropped_face_region(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks, cv::Rect& selected_region)
{
	const int x1_limit = landmarks[0].x - (landmarks[36].x - landmarks[0].x);
	const int x2_limit = landmarks[16].x + (landmarks[16].x - landmarks[45].x);
	const int y1_limit = landmarks[27].y - 3 * (landmarks[30].y - landmarks[27].y);
	const int y2_limit = landmarks[8].y + (landmarks[30].y - landmarks[29].y);

	const int im_width = image.cols;
	const int im_height = image.rows;
	const int x1 = std::max(x1_limit, 0);
	const int x2 = std::min(x2_limit, im_width);
	const int y1 = std::max(y1_limit, 0);
	const int y2 = std::min(y2_limit, im_height);

	selected_region = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat cropped = image(selected_region);
    return cropped;
}

void align_face(const cv::Mat& im_face, cv::Mat& aligned_im_face, const std::vector<cv::Point2f>& landmarks)
{
    const float l_x = landmarks[39].x;
    const float l_y = landmarks[39].y;
    const float r_x = landmarks[42].x;
    const float r_y = landmarks[42].y;

    const float dx = r_x - l_x;
    const float dy = r_y - l_y;
    const double angle = atan2(dy, dx) * 180 / PI;

    cv::Point2f eyes_center;
    eyes_center.x = (l_x + r_x) / 2.0;
    eyes_center.y = (l_y + r_y) / 2.0;

    const cv::Mat rot_matrix = cv::getRotationMatrix2D(eyes_center, angle, 1);
    cv::warpAffine(im_face, aligned_im_face, rot_matrix, im_face.size());
}

void get_file_names(const std::string& dir_name, std::vector<std::string>& image_fnames)
{
    DIR* dir;
    dirent* ent;
    int count = 0;

    //image extensions to be found
    const std::string img_ext1 = "pgm";
    const std::string img_ext2 = "jpg";

    if ((dir = opendir(dir_name.c_str())) != nullptr)
    {
	    std::vector<std::string> files;
	    while ((ent = readdir(dir)) != nullptr)
        {
            // Avoiding dummy names which are read by default
            if (strcmp(ent->d_name, ".") == 0 | strcmp(ent->d_name, "..") == 0)
            {
                continue;
            }

            std::string temp_name = ent->d_name;
            files.push_back(temp_name);
        }

        // Sort file names
        std::sort(files.begin(), files.end());
        for (int it = 0; it < files.size(); it++)
        {
	        std::string path = dir_name;
	        std::string fname = files[it];

            if (fname.find(img_ext1, (fname.length() - img_ext1.length())) != std::string::npos)
            {
                path.append(fname);
                image_fnames.push_back(path);
            }
            else if (fname.find(img_ext2, (fname.length() - img_ext2.length())) != std::string::npos)
            {
                path.append(fname);
                image_fnames.push_back(path);
            }
        }
        closedir(dir);
    }
}

static void read_label_name_map(const std::string& filename, std::vector<std::string>& names, std::vector<int>& labels, std::map<int, std::string>& label_name_map, const char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file)
    {
	    const std::string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

    std::string line;
    std::string name;
    std::string class_label;

    while (getline(file, line))
    {
        // std::cout << line << '\n';
        std::stringstream liness(line);
        std::getline(liness, name, separator);
        std::getline(liness, class_label);
        // std::cout << name << " " << classlabel << '\n';

        if (!name.empty() && !class_label.empty())
        {
            names.push_back(name);
            int label = atoi(class_label.c_str());
            labels.push_back(label);
            label_name_map[label] = name;
        }
    }
}

void face_rec()
{
    cv::VideoCapture cap;
    std::vector<std::string> test_files;
    int test_file_count = 0;

    if (VIDEO)
    {
        // Create a VideoCapture object
        cap.open(util::get_data_path() + "videos/face1.mp4");

        // Check if OpenCV is able to read feed from camera
        if (!cap.isOpened())
        {
	        std::cerr << "Unable to connect to camera\n";
            return;
        }
    }
    else
    {
	    const std::string test_dataset_folder = util::get_data_path() + "images/FaceRec/testFaces";
        get_file_names(test_dataset_folder, test_files);
        test_file_count = 0;
    }

    cv::Ptr<cv::face::FaceRecognizer> face_recognizer;
    if (MODEL == 'e')
    {
	    std::cout << "Using Eigen Faces\n";
        face_recognizer = cv::face::EigenFaceRecognizer::create();
        face_recognizer->read("face_model_eigen.yml");
    }
    else if (MODEL == 'f')
    {
	    std::cout << "Using Fisher Faces\n";
        face_recognizer = cv::face::FisherFaceRecognizer::create();
        face_recognizer->read("face_model_fisher.yml");
    }
    else if (MODEL == 'l')
    {
	    std::cout << "Using LBPH\n";
        face_recognizer = cv::face::LBPHFaceRecognizer::create();
        face_recognizer->read("face_model_lbph.yml");
        std::cout << "Face Recognizer Created.\n";
    }

    std::map<int, std::string> label_name_map;
    std::vector<std::string> names;
    std::vector<int> labels;
    const std::string label_file = "labels_map.txt";
    read_label_name_map(label_file, names, labels, label_name_map);

    // Load face detector
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

    // Load landmark detector.
    dlib::shape_predictor landmark_detector;
    dlib::deserialize(util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat") >> landmark_detector;

    float image_resize;
    cv::Mat im;
    cv::Mat im_gray;

    while (true)
    {
        // Capture frame-by-frame
        if (VIDEO)
        {
            cap >> im;
        }
        else
        {
            im = cv::imread(test_files[test_file_count]);
            test_file_count++;
        }

        // If the frame is empty, break immediately
        if (im.empty())
        {
            break;
        }

        image_resize = static_cast<float>(im.rows) / RESIZE_HEIGHT;
        cv::resize(im, im, cv::Size(), 1.0 / image_resize, 1.0 / image_resize);

        std::vector<cv::Point2f> landmarks = get_landmark_point_vector(im, "images/people/", "no_file", face_detector, landmark_detector);
        if (landmarks.size() < 68)
        {
	        std::cout << "Only " << landmarks.size() << " landmarks found, continuing with next frame\n";
            continue;
        }

        cv::cvtColor(im, im_gray, cv::COLOR_BGR2GRAY);

        cv::Rect face_region;
        cv::Mat im_face = get_cropped_face_region(im_gray, landmarks, face_region);

        cv::Mat aligned_im_face;
        align_face(im_face, aligned_im_face, landmarks);
        cv::resize(aligned_im_face, aligned_im_face, cv::Size(faceHeight, faceWidth));
        aligned_im_face.convertTo(aligned_im_face, CV_32F, 1.0 / 255);

        int predicted_label = -1;
        double score = 0.0;
        face_recognizer->predict(aligned_im_face, predicted_label, score);

        cv::Point2d center = cv::Point2d(face_region.x + face_region.width / 2.0,
                                         face_region.y + face_region.height / 2.0);
        int radius = static_cast<int>(face_region.height / 2.0);

        cv::circle(im, center, radius, cv::Scalar(0, 255, 0), 1, cv::LINE_8);

        cv::putText(im, label_name_map[predicted_label], cv::Point(10, 100),
			cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

        cv::imshow("Face Recognition demo", im);

        int k = 0;
        if (VIDEO)
        {
            k = cv::waitKey(10);
        }
        else
        {
            k = cv::waitKey(1000);
        }

        if (k == 27)
        {
            break;
        }
    }

    if (VIDEO)
    {
        cap.release();
    }
    cv::destroyAllWindows();
}
