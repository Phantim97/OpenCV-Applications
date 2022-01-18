cmake_minimum_required (VERSION 3.8)

project(VisionApplications LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Note LIBS_PATH is my env var to where all my libraries are located. also bin dlls
# are mapped to my PATH as well.

set(DLIB_DIR $ENV{LIBS_PATH}/dlib/build/dlib) 

# To let this work easily I did ALLBUILD and INSTALL for the project
# To adjust I might need to INSTALL for OpenCV and change the dir as well.
find_package(DLIB REQUIRED) 
include_directories(${DLIB_INCLUDE_DIRECTORY})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_LIB_DIR})

# Add source to this project's executable.
add_executable(FaceApp "main.cpp" "alpha_blending.cpp" "alpha_blending.h" "env_util.cpp" "env_util.h" "triangle_warp.h" "traingle_warp.cpp" "delaunay.cpp" "delaunay.h" "face_apps.cpp" "face_apps.h"  )
target_link_libraries(FaceApp ${OpenCV_LIBS})
target_link_libraries(FaceApp ${DLIB_LIBRARIES} dlib::dlib)