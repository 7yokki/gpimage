cmake_minimum_required(VERSION 3.10)
project(GPIMAGE CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(gpimage_core STATIC src/gpimage.cpp)
target_include_directories(gpimage_core PUBLIC src)
target_link_libraries(gpimage_core PUBLIC Threads::Threads)

add_executable(gpimage src/main.cpp)
target_include_directories(gpimage PRIVATE third_party)
target_link_libraries(gpimage PRIVATE gpimage_core)

# Tek dosyalık paylaşılan kütüphane: gpimage_lib.dll / libgpimage_lib.so
# (çekirdek gömülü, harici gpimage dosyasına ihtiyaç yok)
option(GPIMAGE_BUILD_LIB "Build gpimage_lib shared library" ON)
if(GPIMAGE_BUILD_LIB)
  add_subdirectory(library)
endif()
