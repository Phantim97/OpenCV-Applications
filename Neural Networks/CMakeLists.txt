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
include_directories(${CMAKE_SOURCE_DIR}/Utility)

include_directories($ENV{LIBS_PATH}/opencv_contrib-4.x/modules/face/include)

set(CUDNN_LIBRARY_PATH "$ENV{cuDNN}/lib/x64")
set(CUDNN_INCLUDE_PATH "$ENV{cuDNN}/include")
set(CMAKE_PREFIX_PATH "$ENV{LIBS_PATH}/LibTorch/Torch-1.10.0/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH "$ENV{PYTHON_39}")
set(Python_ADDITIONAL_VERSIONS 3.9)

# Find Python Interpreter
find_package(PythonInterp REQUIRED)

# Find Python Header and Library (module is FindPythonLibs.cmake)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)

include_directories("${TORCH_INCLUDE_DIRS}")
# Python Embedding Include Directory
include_directories("${PYTHON_INCLUDE_DIRS}")

add_executable(NeuralNet "main.cpp" "lbitorch_sample.cpp" "libtorch_sample.h"  "image_classification.cpp")
target_link_libraries(NeuralNet ${OpenCV_LIBS})
target_link_libraries(NeuralNet ${DLIB_LIBRARIES} dlib::dlib)
target_link_libraries(NeuralNet ${CMAKE_SOURCE_DIR}/Utility EnvUtil)

# Python Library Linking
target_link_libraries(NeuralNet "${PYTHON_LIBRARIES}")

# Link Torch Libraries
target_link_libraries(NeuralNet "${TORCH_LIBRARIES}")

set_property(TARGET NeuralNet PROPERTY CXX_STANDARD 14)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET NeuralNet
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:NeuralNet>)