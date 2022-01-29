#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/calib3d.hpp>

#include "dirent.h"
#include "env_util.h"

//OpenFace Deep Learning

// Reads files, folders and symbolic links in a directory
void list_dir(const std::string& dir_name, std::vector<std::string>& folder_names, std::vector<std::string>& file_names, std::vector<std::string>& symlink_names)
{
    DIR* dir;
    dirent* ent;

    if ((dir = opendir(dir_name.c_str())) != nullptr)
    {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != nullptr) 
        {
            // ignore . and ..
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) 
            {
                continue;
            }

        	std::string temp_name = ent->d_name;
            // Read more about file types identified by dirent.h here
            // https://www.gnu.org/software/libc/manual/html_node/Directory-Entries.html
            switch (ent->d_type) {
            case DT_REG:
                file_names.push_back(temp_name);
                break;
            case DT_DIR:
                folder_names.push_back(dir_name + "/" + temp_name);
                break;
            case DT_LNK:
                symlink_names.push_back(temp_name);
                break;
            default:
                break;
            }
            // std::cout << temp_name << '\n'';
        }
        // sort all the files
        std::sort(folder_names.begin(), folder_names.end());
        std::sort(file_names.begin(), file_names.end());
        std::sort(symlink_names.begin(), symlink_names.end());
        closedir(dir);
    }
}

// filter files having extension ext i.e. jpg
void filter_files(const std::string& dir_path, const std::vector<std::string>& file_names, std::vector<std::string>& filtered_file_paths, const std::string& ext, std::vector<int>& image_labels, const int index)
{
    for (int i = 0; i < file_names.size(); i++)
    {
	    std::string fname = file_names[i];
        if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos)
        {
            filtered_file_paths.push_back(dir_path + "/" + fname);
            image_labels.push_back(index);
        }
    }
}

template<typename T>
void printVector(std::vector<T>& vec)
{
    for (int i = 0; i < vec.size(); i++) 
    {
        std::cout << i << " " << vec[i] << "; ";
    }
    std::cout << '\n';
}

#define M_PI 3.14159

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

    // Now we can use estimateRigidTransform for calculating the similarity transform.
    tform = cv::estimateAffinePartial2D(in_pts, out_pts);
}

// Converts Dlib landmarks into a vector for Point2f
void dlib_landmarks_to_points(dlib::full_object_detection& landmarks, std::vector<cv::Point2f>& points)
{
    // Loop over all landmark points
    for (int i = 0; i < landmarks.num_parts(); i++)
    {
	    cv::Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
        points.push_back(pt);
    }
}

void align_face(cv::Mat& img_in, cv::Mat& img_out, const dlib::rectangle face_rect, const dlib::shape_predictor& landmark_detector, const cv::Size out_size)
{
	const int h = out_size.height;
	const int w = out_size.width;

    std::vector<cv::Point2f> pointsIn;

	const dlib::cv_image<dlib::bgr_pixel> dlib_im(img_in);
    dlib::full_object_detection landmarks = landmark_detector(dlib_im, face_rect);
    dlib_landmarks_to_points(landmarks, pointsIn);

    std::vector<cv::Point2f> eyecorner_src;
    // Get the locations of the left corner of left eye
    eyecorner_src.push_back(pointsIn[2]);
    // Get the locations of the right corner of right eye
    eyecorner_src.push_back(pointsIn[0]);

    std::vector<cv::Point2f> eyecorner_dst;
    // Location of the left corner of left eye in normalized image.
    eyecorner_dst.push_back(cv::Point2f(0.2 * w, h / 3));
    // Location of the right corner of right eye in normalized image.
    eyecorner_dst.push_back(cv::Point2f(0.8 * w, h / 3));

    // Calculate similarity transform
    cv::Mat tform;
    similarity_transform(eyecorner_src, eyecorner_dst, tform);

    // Apply similarity transform to input image
    img_in.convertTo(img_in, CV_32FC3, 1 / 255.0);
    img_out = cv::Mat::zeros(h, w, CV_32FC3);
    cv::warpAffine(img_in, img_out, tform, img_out.size());

    img_in.convertTo(img_in, CV_8UC3, 255);
    img_out.convertTo(img_out, CV_8UC3, 255);
}

void openface_rec_dl_train()
{
    // Initialize face detector and face recognize
    const std::string rec_model_path = util::get_model_path() + "models/openface.nn4.small2.v1.t7";
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    cv::dnn::Net rec_model = cv::dnn::readNetFromTorch(rec_model_path);
    dlib::shape_predictor landmark_detector;
    dlib::deserialize(util::get_model_path() + "dlib_models/shape_predictor_5_face_landmarks.dat") >> landmark_detector;

    // Now let's prepare our training data
	// data is organized assuming following structure
	// faces folder has subfolders.
	// each subfolder has images of a person
    std::string face_dataset_folder = util::get_dataset_path() + "faces";

    std::vector<std::string> subfolders;
    std::vector<std::string> file_names;
    std::vector<std::string> symlink_names;
    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
	list_dir(face_dataset_folder, subfolders, file_names, symlink_names);

    // names: vector containing names of subfolders i.e. persons
    // labels: integer labels assigned to persons
    // labelNameMap: dict containing (integer label, person name) pairs
    std::vector<std::string> names;
    std::vector<int> labels;
    std::map<int, std::string> label_name_map;
    // add -1 integer label for un-enrolled persons
    names.emplace_back("unknown");
    labels.push_back(-1);

    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<std::string> image_paths;
    std::vector<int> image_labels;
    // variable to hold any subfolders within person subFolders
    std::vector<std::string> folder_names;
    // iterate over all subFolders within faces folder
    for (int i = 0; i < subfolders.size(); i++)
    {
	    std::string person_folder_name = subfolders[i];
        // remove / or \\ from end of subFolder
        std::size_t found = person_folder_name.find_last_of("/\\");
	    std::string name = person_folder_name.substr(found + 1);
        // assign integer label to person subFolder
        int label = i;
        // add person name and label to vectors
        names.push_back(name);
        labels.push_back(label);
        // add (integer label, person name) pair to map
        label_name_map[label] = name;

        // read imagePaths from each person subFolder
        // clear vectors
        folder_names.clear();
        file_names.clear();
        symlink_names.clear();
        // folderNames and symlinkNames are useless here
        // as we are only looking for files here
        // read all files present in subFolder
        list_dir(subfolders[i], folder_names, file_names, symlink_names);
        // filter only jpg files
        filter_files(subfolders[i], file_names, image_paths, "jpg", image_labels, i);
    }

    // process training data
	// We will store face descriptors in vector faceDescriptors
	// and their corresponding labels in vector faceLabels
    std::vector<cv::Mat> face_descriptors;
	// std::vector<cv_image<bgr_pixel> > imagesFaceTrain;
    std::vector<int> face_labels;
    cv::Mat face_descriptor;
    // iterate over images
    for (int i = 0; i < image_paths.size(); i++)
    {
	    std::string image_path = image_paths[i];
        int image_label = image_labels[i];

        std::cout << "processing: " << image_path << '\n';

        // read image using OpenCV
        cv::Mat im = cv::imread(image_path);

        dlib::cv_image<dlib::bgr_pixel> im_dlib(im);
        std::vector<dlib::rectangle> faceRects = face_detector(im_dlib);
        std::cout << faceRects.size() << " Face(s) Found\n" ;
        // Now process each face we found
        for (int j = 0; j < faceRects.size(); j++) 
        {
	        cv::Mat aligned_face;
            align_face(im, aligned_face, faceRects[j], landmark_detector, cv::Size(96, 96));

            cv::Mat blob = cv::dnn::blobFromImage(aligned_face, 1.0 / 255, cv::Size(96, 96), cv::Scalar(0, 0, 0), false, false);
            rec_model.setInput(blob);
            face_descriptor = rec_model.forward();

            // add face descriptor and label for this face to
            // vectors faceDescriptors and faceLabels
            face_descriptors.push_back(face_descriptor.clone());

            // add label for this face to vector containing labels corresponding to
            // vector containing face descriptors
            face_labels.push_back(image_label);
        }
    }

    // write label name map to disk
    const std::string label_name_file = "label_name_openface.txt";
    std::ofstream of;
    of.open(label_name_file);

    for (int m = 0; m < names.size(); m++) 
    {
        of << names[m];
        of << ";";
        of << labels[m];
        of << "\n";
    }

    of.close();

    std::cout << "number of face descriptors " << face_descriptors.size() << '\n';
    std::cout << "number of face labels " << face_labels.size() << '\n';

    // write face labels and descriptor to disk
	// each row of file descriptors_openface.csv has:
	// 1st element as face label and
	// rest 128 as descriptor values
    const std::string descriptors_path = "descriptors_openface.csv";
    std::ofstream ofs;
    ofs.open(descriptors_path);
    // write descriptors
    for (int m = 0; m < face_descriptors.size(); m++) 
    {
	    cv::Mat face_descriptor_vec = face_descriptors[m];
        ofs << face_labels[m];
        ofs << ";";
        for (int n = 0; n < face_descriptor_vec.cols; n++)
        {
            ofs << std::fixed << std::setprecision(8) << face_descriptor_vec.at<float>(n);
            // std::cout << n << " " << faceDescriptorVec.at<float>(n) << '\n';
            if (n == face_descriptor_vec.cols - 1) 
            {
                ofs << "\n";  // add ; if not the last element of descriptor
            }
            else 
            {
                ofs << ";";  // add newline character if last element of descriptor
            }
        }
    }

    ofs.close();
}

//Testing

// read names and labels mapping from file
static void read_label_name_map(const std::string& filename, std::vector<std::string>& names, std::vector<int>& labels,
    std::map<int, std::string>& label_name_map, char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) 
    {
	    const std::string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

    std::string line;
    std::string name;
    std::string label_str;
    // read lines from file one by one
    while (std::getline(file, line))
    {
	    std::stringstream liness(line);
        // read first word which is person name
        std::getline(liness, name, separator);
        // read second word which is integer label
        std::getline(liness, label_str);

        if (!name.empty() && !label_str.empty()) 
        {
            names.push_back(name);
            // convert label from string format to integer
            int label = std::atoi(label_str.c_str());
            labels.push_back(label);
            // add (integer label, person name) pair to map
            label_name_map[label] = name;
        }
    }
}

// read descriptors saved on disk
static void read_descriptors(const std::string& filename, std::vector<int>& face_labels, std::vector<cv::Mat>& face_descriptors, char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file)
    {
	    std::string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    // each line has:
    // 1st element = face label
    // rest 128 elements = descriptor elements
    std::string line;
    std::string face_label;
    // valueStr = one element of descriptor in string format
    // value = one element of descriptor in float
    std::string value_str;
    float value;
    std::vector<float> face_descriptor_vec;
    // read lines from file one by one
    while (std::getline(file, line)) 
    {
	    std::stringstream liness(line);
        // read face label
        // read first word on a line till separator
        std::getline(liness, face_label, separator);

        if (!face_label.empty()) 
        {
            face_labels.push_back(std::atoi(face_label.c_str()));
        }

        face_descriptor_vec.clear();
        // read rest of the words one by one using separator
        while (std::getline(liness, value_str, separator))
        {
            if (!value_str.empty()) 
            {
                // convert descriptor element from string to float
                face_descriptor_vec.push_back(atof(value_str.c_str()));
            }
        }

        // convert face descriptor from vector of float to Dlib's matrix format
        cv::Mat face_descriptor(face_descriptor_vec);
        face_descriptors.push_back(face_descriptor.clone());
    }
}

constexpr float g_rec_threshold = 0.4f;

void nearest_neighbor(const cv::Mat& face_descriptor_query, const std::vector<cv::Mat>& face_descriptors, const std::vector<int>& face_labels, int& label, float& min_distance)
{
    int min_dist_index = 0;
    min_distance = 1.0;
    label = -1;
    // Calculate Euclidean distances between face descriptor calculated on face dectected
    // in current frame with all the face descriptors we calculated while enrolling faces
    // Calculate minimum distance and index of this face
    for (int i = 0; i < face_descriptors.size(); i++) 
    {
	    const double distance = cv::norm(face_descriptors[i].t() - face_descriptor_query);
        if (distance < min_distance) 
        {
            min_distance = distance;
            min_dist_index = i;
        }
    }
    // if minimum distance is greater than a threshold
    // assign integer label -1 i.e. unknown face
    if (min_distance > g_rec_threshold) 
    {
        label = -1;
    }
    else 
    {
        label = face_labels[min_dist_index];
    }
}

void openface_rec_dl_test()
{
    // Initialize face detector, facial landmarks detector and face recognizer
    const std::string rec_model_path = util::get_model_path() + "models/openface.nn4.small2.v1.t7";
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    cv::dnn::Net rec_model = cv::dnn::readNetFromTorch(rec_model_path);
    dlib::shape_predictor landmark_detector;
    dlib::deserialize(util::get_model_path() + "models/shape_predictor_5_face_landmarks.dat") >> landmark_detector;

    // read names, labels and labels-name-mapping from file
    std::map<int, std::string> label_name_map;
    std::vector<std::string> names;
    std::vector<int> labels;
    const std::string label_name_file = "label_name_openface.txt";
    read_label_name_map(label_name_file, names, labels, label_name_map);

    // read descriptors of enrolled faces from file
    const std::string face_descriptor_file = "descriptors_openface.csv";
    std::vector<int> face_labels;
    std::vector<cv::Mat> face_descriptors;
    read_descriptors(face_descriptor_file, face_labels, face_descriptors);

    // read query image
    std::string image_path;
    image_path = util::get_data_path() + "images/faces/tim_face.jpg";
    cv::Mat img = cv::imread(image_path);

    double t = cv::getTickCount();

    dlib::cv_image<dlib::bgr_pixel> im_dlib(img);

    // detect faces in image
    std::vector<dlib::rectangle> face_rects = face_detector(im_dlib);

    std::string name;
    // Now process each face we found
    for (int i = 0; i < face_rects.size(); i++) 
    {
	    std::cout << face_rects.size() << " Face(s) Found\n";

        cv::Mat aligned_face;
        align_face(img, aligned_face, face_rects[i], landmark_detector, cv::Size(96, 96));
        cv::Mat blob = cv::dnn::blobFromImage(aligned_face, 1.0 / 255, cv::Size(96, 96), cv::Scalar(0, 0, 0), false, false);
        rec_model.setInput(blob);
        cv::Mat face_descriptor_query = rec_model.forward();

        // Find closest face enrolled to face found in frame
        int label;
        float min_distance;
        nearest_neighbor(face_descriptor_query, face_descriptors, face_labels, label, min_distance);
        // Name of recognized person from map
        name = label_name_map[label];

        // Draw a rectangle for detected face
        cv::Point2d p1 = cv::Point2d(face_rects[i].left(), face_rects[i].top());
        cv::Point2d p2 = cv::Point2d(face_rects[i].right(), face_rects[i].bottom());
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), 1, cv::LINE_8);

        // Draw circle for face recognition
        cv::Point2d center = cv::Point((face_rects[i].left() + face_rects[i].right()) / 2.0,
			(face_rects[i].top() + face_rects[i].bottom()) / 2.0);
        int radius = static_cast<int> ((face_rects[i].bottom() - face_rects[i].top()) / 2.0);
        cv::circle(img, center, radius, cv::Scalar(0, 255, 0), 1, cv::LINE_8);

        // Write text on image specifying identified person and minimum distance
        std::stringstream stream;
        stream << name << " ";
        stream << std::fixed << std::setprecision(4) << min_distance;
        std::string text = stream.str(); // name + " " + std::to_string(minDistance);
        cv::putText(img, text, p1, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("Output", img);
    cv::waitKey(10000);
}

void openface_rec_dl()
{
    openface_rec_dl_train();
    openface_rec_dl_test();
}

//Dlib Deep Learning

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
	dlib::input_rgb_image_sized<150>
	>>>>>>>>>>>>;
// ----------------------------------------------------------------------------------------

void dlib_rec_train_dl()
{
    // Initialize face detector, facial landmarks detector and face recognizer
    const std::string predictor_path = util::get_model_path() + "models/shape_predictor_68_face_landmarks.dat";
    const std::string face_recognition_model_path = util::get_model_path() + "models/dlib_face_recognition_resnet_model_v1.dat";
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor landmark_detector;
    dlib::deserialize(predictor_path) >> landmark_detector;
    anet_type net;
    dlib::deserialize(face_recognition_model_path) >> net;

    // Now let's prepare our training data
	// data is organized assuming following structure
	// faces folder has subfolders.
	// each subfolder has images of a person
    std::string face_dataset_folder = "../data/images/faces";
    std::vector<std::string> subfolders;
    std::vector<std::string> file_names;
    std::vector<std::string> symlink_names;
    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
    list_dir(face_dataset_folder, subfolders, file_names, symlink_names);

    // names: vector containing names of subfolders i.e. persons
    // labels: integer labels assigned to persons
    // labelNameMap: dict containing (integer label, person name) pairs
    std::vector<std::string> names;
    std::vector<int> labels;
    std::map<int, std::string> label_name_map;
    // add -1 integer label for un-enrolled persons
    names.emplace_back("unknown");
    labels.push_back(-1);

    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<std::string> image_paths;
    std::vector<int> image_labels;

    // variable to hold any subfolders within person subFolders
    std::vector<std::string> folder_names;
    // iterate over all subFolders within faces folder
    for (int i = 0; i < subfolders.size(); i++) 
    {
	    std::string person_folder_name = subfolders[i];
        // remove / or \\ from end of subFolder
        std::size_t found = person_folder_name.find_last_of("/\\");
        std::string name = person_folder_name.substr(found + 1);
        // assign integer label to person subFolder
        int label = i;
        // add person name and label to vectors
        names.push_back(name);
        labels.push_back(label);
        // add (integer label, person name) pair to map
        label_name_map[label] = name;

        // read imagePaths from each person subFolder
        // clear vectors
        folder_names.clear();
        file_names.clear();
        symlink_names.clear();
        // folderNames and symlinkNames are useless here
        // as we are only looking for files here
        // read all files present in subFolder
        list_dir(subfolders[i], folder_names, file_names, symlink_names);
        // filter only jpg files
        filter_files(subfolders[i], file_names, image_paths, "jpg", image_labels, i);
    }

    // process training data
	// We will store face descriptors in vector faceDescriptors
	// and their corresponding labels in vector faceLabels
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
    // std::vector<cv_image<bgr_pixel> > imagesFaceTrain;
    std::vector<int> face_labels;
    // iterate over images
    for (int i = 0; i < image_paths.size(); i++) 
    {
	    std::string image_path = image_paths[i];
        int image_label = image_labels[i];

        std::cout << "processing: " << image_path << '\n';

        // read image using OpenCV
        cv::Mat im = cv::imread(image_path, cv::IMREAD_COLOR);

        // convert image from BGR to RGB
        // because Dlib used RGB format
        cv::Mat imRGB;
        cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);

        // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
        // Dlib's dnn module doesn't accept Dlib's cv_image template
        dlib::matrix<dlib::rgb_pixel> im_dlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

        // detect faces in image
        std::vector<dlib::rectangle> face_rects = face_detector(im_dlib);
        std::cout << face_rects.size() << " Face(s) Found" << '\n';
        // Now process each face we found
        for (int j = 0; j < face_rects.size(); j++) 
        {
            // Find facial landmarks for each detected face
            dlib::full_object_detection landmarks = landmark_detector(im_dlib, face_rects[j]);

            // object to hold preProcessed face rectangle cropped from image
            dlib::matrix<dlib::rgb_pixel> face_chip;

            // original face rectangle is warped to 150x150 patch.
            // Same pre-processing was also performed during training.
            extract_image_chip(im_dlib, dlib::get_face_chip_details(landmarks, 150, 0.25), face_chip);

            // Compute face descriptor using neural network defined in Dlib.
            // It is a 128D vector that describes the face in img identified by shape.
            dlib::matrix<float, 0, 1> face_descriptor = net(face_chip);

            // add face descriptor and label for this face to
            // vectors faceDescriptors and faceLabels dlib::
            face_descriptors.push_back(face_descriptor);
            // add label for this face to vector containing labels corresponding to
            // vector containing face descriptors
            face_labels.push_back(image_label);
        }
    }

    // write label name map to disk
    const std::string label_name_file = "label_name.txt";
    std::ofstream of;
    of.open(label_name_file);

	for (int m = 0; m < names.size(); m++)
    {
        of << names[m];
        of << ";";
        of << labels[m];
        of << "\n";
    }

    of.close();

    std::cout << "number of face descriptors " << face_descriptors.size() << '\n';
    std::cout << "number of face labels " << face_labels.size() << '\n';

    // write face labels and descriptor to disk
	// each row of file descriptors.csv has:
	// 1st element as face label and
	// rest 128 as descriptor values
    const std::string descriptors_path = "descriptors.csv";
    std::ofstream ofs;
    ofs.open(descriptors_path);
    // write descriptors
    for (int m = 0; m < face_descriptors.size(); m++)
    {
        dlib::matrix<float, 0, 1> faceDescriptor = face_descriptors[m];
        std::vector<float> faceDescriptorVec(faceDescriptor.begin(), faceDescriptor.end());
        // std::out << "Label " << faceLabels[m] << '\n';
        ofs << face_labels[m];
        ofs << ";";
        for (int n = 0; n < faceDescriptorVec.size(); n++) 
        {
            ofs << std::fixed << std::setprecision(8) << faceDescriptorVec[n];
            // std::cout << n << " " << faceDescriptorVec[n] << '\n';
            if (n == (faceDescriptorVec.size() - 1)) 
            {
                ofs << "\n";  // add ; if not the last element of descriptor
            }
            else 
            {
                ofs << ";";  // add newline character if last element of descriptor
            }
        }
    }
    ofs.close();
}

//Testing
// find nearest face descriptor from vector of enrolled faceDescriptor
// to a query face descriptor
void nearest_neighbor(const dlib::matrix<float, 0, 1>& face_descriptor_query,
	const std::vector<dlib::matrix<float, 0, 1>>& face_descriptors,
	const std::vector<int>& face_labels, int& label, float& min_distance)
{
    int min_dist_index = 0;
    min_distance = 1.0;
    label = -1;
    // Calculate Euclidean distances between face descriptor calculated on face dectected
    // in current frame with all the face descriptors we calculated while enrolling faces
    // Calculate minimum distance and index of this face
    for (int i = 0; i < face_descriptors.size(); i++) 
    {
	    const double distance = length(face_descriptors[i] - face_descriptor_query);
        if (distance < min_distance) 
        {
            min_distance = distance;
            min_dist_index = i;
        }
    }
    // Dlib specifies that in general, if two face descriptor vectors have a Euclidean
    // distance between them less than 0.6 then they are from the same
    // person, otherwise they are from different people.

    // This threshold will vary depending upon number of images enrolled
    // and various variations (illuminaton, camera quality) between
    // enrolled images and query image
    // We are using a threshold of 0.5
    // if minimum distance is greater than a threshold
    // assign integer label -1 i.e. unknown face
    if (min_distance > g_rec_threshold) 
    {
        label = -1;
    }
    else
    {
        label = face_labels[min_dist_index];
    }
}

// read descriptors saved on disk
static void read_descriptors(const std::string& filename, std::vector<int>& face_labels, std::vector<dlib::matrix<float, 0, 1>>& face_descriptors, char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) 
    {
	    std::string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    // each line has:
    // 1st element = face label
    // rest 128 elements = descriptor elements
    std::string line;
    std::string face_label;
    // valueStr = one element of descriptor in string format
    // value = one element of descriptor in float
    std::string value_str;
    float value;
    std::vector<float> faceDescriptorVec;
    // read lines from file one by one
    while (std::getline(file, line))
    {
	    std::stringstream liness(line);
        // read face label
        // read first word on a line till separator
        std::getline(liness, face_label, separator);
        if (!face_label.empty()) 
        {
            face_labels.push_back(std::atoi(face_label.c_str()));
        }

        faceDescriptorVec.clear();
        // read rest of the words one by one using separator
        while (std::getline(liness, value_str, separator)) 
        {
            if (!value_str.empty()) 
            {
                // convert descriptor element from string to float
                faceDescriptorVec.push_back(std::atof(value_str.c_str()));
            }
        }

        // convert face descriptor from vector of float to Dlib's matrix format
        dlib::matrix<float, 0, 1> face_descriptor = dlib::mat(faceDescriptorVec);
        face_descriptors.push_back(face_descriptor);
    }
}

void dlib_rec_test_dl()
{
    // Initialize face detector, facial landmarks detector and face recognizer

    const std::string predictor_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
    const std::string face_recognition_model_path = util::get_model_path() + "dlib_models/dlib_face_recognition_resnet_model_v1.dat";
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor landmark_detector;
    dlib::deserialize(predictor_path) >> landmark_detector;
    anet_type net;
    dlib::deserialize(face_recognition_model_path) >> net;

    // read names, labels and labels-name-mapping from file
    std::map<int, std::string> label_name_map;
    std::vector<std::string> names;
    std::vector<int> labels;
    const std::string label_name_file = "label_name.txt";
    read_label_name_map(label_name_file, names, labels, label_name_map);

    // read descriptors of enrolled faces from file
    const std::string face_descriptor_file = "descriptors.csv";
    std::vector<int> face_labels;
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
    read_descriptors(face_descriptor_file, face_labels, face_descriptors);
    // read query image
    std::string image_path;
    image_path = util::get_data_path() + "images/faces/tim_face.jpg";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    double t = cv::getTickCount();
    // convert image from BGR to RGB
    // because Dlib used RGB format
    cv::Mat im_rgb = img.clone();
    cv::cvtColor(img, im_rgb, cv::COLOR_BGR2RGB);
    // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
    // Dlib's dnn module doesn't accept Dlib's cv_image template
    dlib::matrix<dlib::rgb_pixel> im_dlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(im_rgb)));

    std::vector<dlib::rectangle> face_rects = face_detector(im_dlib);
    std::cout << face_rects.size() << " Faces Detected " << '\n';
    std::string name;
    // Now process each face we found
    for (int i = 0; i < face_rects.size(); i++) 
    {
        // Find facial landmarks for each detected face
        dlib::full_object_detection landmarks = landmark_detector(im_dlib, face_rects[i]);

        // object to hold preProcessed face rectangle cropped from image
        dlib::matrix<dlib::rgb_pixel> face_chip;

        // original face rectangle is warped to 150x150 patch.
        // Same pre-processing was also performed during training.
        extract_image_chip(im_dlib, dlib::get_face_chip_details(landmarks, 150, 0.25), face_chip);

        // Compute face descriptor using neural network defined in Dlib.
        // It is a 128D vector that describes the face in img identified by shape.
        dlib::matrix<float, 0, 1> face_descriptor_query = net(face_chip);

        // Find closest face enrolled to face found in frame
        int label;
        float min_distance;
        nearest_neighbor(face_descriptor_query, face_descriptors, face_labels, label, min_distance);
        // Name of recognized person from map
        name = label_name_map[label];

        // Draw a rectangle for detected face
        cv::Point2d p1 = cv::Point2d(face_rects[i].left(), face_rects[i].top());
        cv::Point2d p2 = cv::Point2d(face_rects[i].right(), face_rects[i].bottom());
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), 1, cv::LINE_8);

        // Draw circle for face recognition
        cv::Point2d center = cv::Point((face_rects[i].left() + face_rects[i].right()) / 2.0,
            (face_rects[i].top() + face_rects[i].bottom()) / 2.0);
        int radius = static_cast<int> ((face_rects[i].bottom() - face_rects[i].top()) / 2.0);
        cv::circle(img, center, radius, cv::Scalar(0, 255, 0), 1, cv::LINE_8);

        // Write text on image specifying identified person and minimum distance
        std::stringstream stream;
        stream << name << " ";
        stream << std::fixed << std::setprecision(4) << min_distance;
        std::string text = stream.str(); // name + " " + std::to_string(minDistance);
        cv::putText(img, text, p1, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }
}

void dlib_rec_dl()
{
    dlib_rec_train_dl();
    dlib_rec_test_dl();
}