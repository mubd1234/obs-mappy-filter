# OBS Shape Overlay Filter

A video filter plugin for OBS Studio that detects a template shape in a frame (PNG sample) and overlays a PNG on the detected location.

## How It Works
- Loads a template PNG and converts it to grayscale.
- Uses OpenCV template matching (`TM_CCOEFF_NORMED`) every N milliseconds.
- When the match score is above the threshold, it alpha-blends the overlay PNG at the detected top-left corner (plus optional offsets).

## Limitations
- Uses `filter_video`, which only runs on **asynchronous** video filters. Synchronous (GPU) sources will not call this filter. OBS documents that `filter_video` is only used with asynchronous video filters.
- Only BGRA/BGRX frames are supported. If a source outputs YUV or other formats, the filter will skip processing.
- No rotation or multi-scale matching (template must match at 1:1 scale unless you pre-scale the template).
- CPU-heavy on large frames; use a higher detection interval for performance.

## Build Notes
This repository follows the OBS plugin directory structure and CMake conventions documented by OBS. It assumes you are building with the OBS Studio build system or an OBS plugin template that provides the `libobs` target and the `install_obs_plugin_with_data` macro.

Typical flow:
1. Install OpenCV (core, imgproc, imgcodecs) and ensure CMake can find it.
2. Configure CMake so it can find `libobs` (by building inside the OBS tree or using an OBS plugin template).
3. Build and install the module.

Example (conceptual):
```powershell
cmake -S . -B build -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build build --config Release
cmake --install build --config Release
```

## Usage
1. Add the filter to a video source in OBS.
2. Set **Template PNG** to the sample shape.
3. Set **Overlay PNG** to the shape you want to draw on top.
4. Adjust **Match Threshold** and **Detection Interval** for performance.

