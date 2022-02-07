cmake_minimum_required (VERSION 3.8)

project(EnvironmentUtility LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# Since my CMAKE default installed to Program Files x86 I would need to start as admin
# Every time, by just routing to the .cmake in the build I can circumvent this
set(DLIB_DIR $ENV{LIBS_PATH}/dlib/build/dlib) 

# To let this work easily I did ALLBUILD and INSTALL for the project
# To adjust I might need to INSTALL for OpenCV and change the dir as well.
find_package(DLIB REQUIRED)

include_directories(${DLIB_INCLUDE_DIRECTORY})
include_directories(${OpenCV_LIB_DIR})

add_library(EnvUtil STATIC "env_util.h" "env_util.cpp" "utils.cpp" "utils.h" "ml_util.h" "ml_util.cpp" "dirent.h" "CSVReader.h" "str_split.h")

target_link_libraries(EnvUtil ${OpenCV_LIBS})
target_link_libraries(EnvUtil ${DLIB_LIBRARIES} dlib::dlib)