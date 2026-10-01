# GPIMAGE

CPU image upscaler: Nearest / Bilinear / Bicubic (Catmull-Rom) / Lanczos3 / **EASU** + optional **RCAS** sharpening.
Multithreaded, C++17, only dependency is the bundled stb_image / stb_image_write.

## Build
    cmake -S . -B build && cmake --build build --config Release
    # or without CMake:
    g++ -std=c++17 -O2 -Isrc -Ithird_party src/main.cpp src/gpimage.cpp -o gpimage -pthread

## Use
    gpimage photo.jpg out.png --scale 2
    gpimage photo.jpg out.png --size 3840x2160 --method easu --sharpen 0.8
    gpimage photo.png out.png --width 2000 --method bicubic --sharpen 0

Output format follows the extension (png, jpg, bmp, tga). Alpha channel is preserved.
`--sharpen` is RCAS strength 0..1 (0 = off). Downscaling is anti-aliased for the separable methods.
