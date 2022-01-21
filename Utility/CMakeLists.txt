cmake_minimum_required (VERSION 3.8)

project(EnvironmentUtility LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(EnvUtil STATIC "env_util.h" "env_util.cpp")