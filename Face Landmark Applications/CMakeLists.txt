cmake_minimum_required (VERSION 3.8)

project(VisionApplications LANGUAGES CXX)

# Note LIBS_PATH is my env var to where all my libraries are located. also bin dlls
# are mapped to my PATH as well.

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_LIB_DIR})

# Add source to this project's executable.
add_executable(FaceApp "main.cpp")
target_link_libraries(FaceApp ${OpenCV_LIBS})