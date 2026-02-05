cmake_minimum_required(VERSION 3.20)
project(obs-shape-overlay)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(obs_shape_overlay_SOURCES
  src/obs-shape-overlay.cpp
  src/shape_overlay_filter.cpp
)

add_library(obs-shape-overlay MODULE ${obs_shape_overlay_SOURCES})

target_link_libraries(obs-shape-overlay libobs)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
target_include_directories(obs-shape-overlay PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(obs-shape-overlay ${OpenCV_LIBS})

install_obs_plugin_with_data(obs-shape-overlay data)
