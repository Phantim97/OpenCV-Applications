#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "env_util.h"

#define RESIZE_HEIGHT 240
#define FACE_DOWNSAMPLE_RATIO_DLIB 1



bool smile_detector(const dlib::cv_image<dlib::bgr_pixel>& cimg, const dlib::rectangle& face, const dlib::full_object_detection& landmarks)
{
    enum BodyPart
    {
        LEFT_JAW = 3,
        RIGHT_JAW = 15,
        LEFT_LIP = 49,
        RIGHT_LIP = 55,
    };

    //55 - 49
    const double lip_width = landmarks.part(RIGHT_LIP).x() - landmarks.part(LEFT_LIP).x();
    //15 - 3
    const double jaw_width = landmarks.part(RIGHT_JAW).x() - landmarks.part(LEFT_JAW).x();

    const double ratio = lip_width / jaw_width;

    const bool is_smiling = ratio > 0.302;

    return is_smiling;
}

void smile_detection()
{
    // initialize dlib's face detector (HOG-based) and then create
    // the facial landmark predictor
    std::cout << "[INFO] loading facial landmark predictor...\n";
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shape_predictor;
    // Load model
    dlib::deserialize(util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat") >> shape_predictor;

    // Initializing video capture object
    cv::VideoCapture capture(util::get_data_path() + "videos/smile.mp4");

    if (!capture.isOpened()) 
    {
        std::cerr << "[ERROR] Unable to connect to camera\n";
    }

    // Create a VideoWriter object
    cv::VideoWriter smile_detection_out("smileDetectionOutput.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        15,
        cv::Size(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT))));

    int frame_number = 0;
    std::vector<int> smile_frames;

    cv::Mat frame, frame_small;
    float image_resize;

    while (capture.read(frame)) 
    {
        if (frame.empty())
        {
            std::cout << "[ERROR] Unable to capture frame\n";
            break;
        }

        //std::cout << "Processing frame: " << frame_number << std::endl;

        image_resize = static_cast<float>(frame.rows) / RESIZE_HEIGHT;
        cv::resize(frame, frame, cv::Size(), 1.0 / image_resize, 1.0 / image_resize);
        cv::resize(frame, frame_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO_DLIB, 1.0 / FACE_DOWNSAMPLE_RATIO_DLIB);

        // Turn OpenCV's Mat into something dlib can deal with. Note that this just
        // wraps the Mat object, it doesn't copy anything. So cimg is only valid as
        // long as frame is valid.  Also don't do anything to frame that would cause it
        // to reallocate the memory which stores the image as that will make cimg
        // contain dangling pointers.  This basically means you shouldn't modify frame
        // while using cimg.
        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
        dlib::cv_image<dlib::bgr_pixel> cimg_small(frame_small);

        // Detect faces 
        std::vector<dlib::rectangle> faces = detector(cimg_small);

        // if # faces detected is zero
        if (faces.empty()) 
        {
            cv::putText(frame, "Unable to detect face, Please check proper lighting", cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
        else 
        {
            dlib::rectangle face(
                (long)(faces[0].left() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].top() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].right() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].bottom() * FACE_DOWNSAMPLE_RATIO_DLIB)
            );

            dlib::full_object_detection landmarks = shape_predictor(cimg, face);

            if (smile_detector(cimg, face, landmarks)) 
            {
                cv::putText(frame, cv::format("Smiling :)"), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                smile_frames.push_back(frame_number);
            }
        }

        if (frame_number % 50 == 0) 
        {
            std::cout << "\nProcessed " << frame_number << " frames\n";
            std::cout << "Smile detected in " << smile_frames.size() << " number of frames\n";
        }

        // Write to VideoWriter
        cv::resize(frame, frame, cv::Size(), image_resize, image_resize);
        smile_detection_out.write(frame);
        frame_number++;
    }

    capture.release();
    smile_detection_out.release();
}