cmake_minimum_required(VERSION 3.22)
project(SelfOrganizingMaps)
set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} include)


add_executable(demo-linear src/demo_linear.cpp
        include/som.hpp
        include/vis.hpp)

target_link_libraries( demo-linear ${OpenCV_LIBS} )

add_executable(run-tests tests/som-tests.cpp
        include/som.hpp)