cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dn_lego_syn)
# change to appropriate path
set(CMAKE_PREFIX_PATH "/homes/zhan2597/scratch/test_libtorch/libtorch")
find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(dn_lego_syn dn_lego_syn.cpp)
target_link_libraries(dn_lego_syn "${TORCH_LIBRARIES}")
target_link_libraries(dn_lego_syn "${OpenCV_LIBS}")
set_property(TARGET dn_lego_syn PROPERTY CXX_STANDARD 11)