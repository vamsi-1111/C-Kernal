# C Kernel for K-Means Image Compression

## Overview
A high-performance image compression project leveraging a custom-built K-Means clustering kernel written in **C**. This kernel reduces 24-bit RGB images to 8-bit colormaps by clustering similar colors, offering significant performance gains over standard Python-based implementations. The project includes a Python pipeline for preprocessing, integration, benchmarking, and visualization.

## Features
- **C-Based K-Means Kernel**: Core clustering algorithm implemented in C for maximum performance and memory efficiency.
- **Python Integration**: Interfaced with Python using `ctypes` for seamless high-level control and automation.
- **End-to-End Pipeline**: Handles image loading, color compression, colormap generation, and reconstruction.
- **Performance Benchmarking**: Compared against NumPy implementation across 5 real-world images; achieved up to **36× speedup**.
- **Cross-Platform Execution**: Validated on macOS and Linux environments using shared libraries.

## Technologies Used
- **Core Language**: C  
- **Scripting & Integration**: Python  
- **Libraries**: NumPy, Pillow (PIL), ctypes  
- **Performance Tools**: GCC with `-O3` optimization, memory-efficient allocation (`calloc`/`free`)  
- **OS/Environment**: macOS (Apple M2), Linux

## How It Works
1. Python loads a 24-bit RGB image and flattens it to a pixel array.
2. C kernel initializes centroids and iteratively clusters pixels using Euclidean distance.
3. Pixel labels and new centroids are computed and returned to Python.
4. Python reconstructs the image using the 8-bit colormap and saves the output.
5. Performance times for both Python and C implementations are printed and logged.

## Benchmark Results (Apple M2, 8-core CPU)
| Image             | Pixels     | Python Time | C Time  | Speedup |
|-------------------|------------|-------------|---------|---------|
| `colors.jpg`      | 720,000    | 23.96s      | 0.66s   | ~36×    |
| `crayons.jpg`     | 360,000    | 9.02s       | 0.32s   | ~28×    |
| `starry-night.jpg`| 289,200    | 6.67s       | 0.24s   | ~27×    |
| `landscape.jpg`   | 163,450    | 3.17s       | 0.13s   | ~24×    |
| `painting.jpg`    | 640,920    | 21.96s      | 0.55s   | ~40×    |

## Setup & Usage
1. Compile the C kernel using:
   ```bash
   gcc -O3 -shared -o libkernel.so -fPIC kernel.c
