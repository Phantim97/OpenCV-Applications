cmake_minimum_required (VERSION 3.8)

project(VisionApplications LANGUAGES CXX)

find_package(OpenCV REQUIRED)

# Since my CMAKE default installed to Program Files x86 I would need to start as admin
# Every time, by just routing to the .cmake in the build I can circumvent this
set(DLIB_DIR $ENV{LIBS_PATH}/dlib/build/dlib) 

# To let this work easily I did ALLBUILD and INSTALL for the project
# To adjust I might need to INSTALL for OpenCV and change the dir as well.
find_package(DLIB REQUIRED) 
include_directories(${DLIB_INCLUDE_DIRECTORY})
include_directories(${OpenCV_LIB_DIR})

# Add source to this project's executable.
add_executable(FaceDetect "main.cpp" "face_detect_dlib_.h" "face_detect_dlib.cpp" "face_alignment.h" "face_alignment.cpp" "custom_detector.cpp" "custom_detector.h")
target_link_libraries(FaceDetect ${OpenCV_LIBS})
target_link_libraries(FaceDetect ${DLIB_LIBRARIES} dlib::dlib)
target_link_libraries(FaceDetect ${CMAKE_SOURCE_DIR}/Utility EnvUtil)