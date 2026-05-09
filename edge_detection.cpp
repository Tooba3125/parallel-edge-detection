#include "edge_detection.h"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

EdgeDetector::EdgeDetector(int threads) : num_threads(threads) {
    omp_set_num_threads(threads);
}

Image EdgeDetector::convertToGrayscale(const std::string& filename) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    
    if (!img_data) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return {0, 0, {}};
    }
    
    Image result;
    result.width = width;
    result.height = height;
    result.data.resize(width * height);
    
    // Parallel grayscale conversion
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            unsigned char gray = static_cast<unsigned char>(
                0.299 * img_data[idx] + 
                0.587 * img_data[idx + 1] + 
                0.114 * img_data[idx + 2]
            );
            result.data[i * width + j] = gray;
        }
    }
    
    stbi_image_free(img_data);
    return result;
}

Image EdgeDetector::gaussianBlur(const Image& img, int kernel_size) {
    std::vector<float> kernel(kernel_size);
    float sigma = kernel_size / 3.0f;
    float sum = 0.0f;
    int radius = kernel_size / 2;
    
    // Create Gaussian kernel
    for (int i = 0; i < kernel_size; i++) {
        int x = i - radius;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    for (float& k : kernel) k /= sum;
    
    Image temp, result;
    temp.width = img.width;
    temp.height = img.height;
    temp.data.resize(img.width * img.height);
    result.width = img.width;
    result.height = img.height;
    result.data.resize(img.width * img.height);
    
    // Horizontal pass
    #pragma omp parallel for
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            float val = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int jj = std::min(std::max(j + k, 0), img.width - 1);
                val += kernel[k + radius] * img.data[i * img.width + jj];
            }
            temp.data[i * img.width + j] = static_cast<unsigned char>(val);
        }
    }
    
    // Vertical pass
    #pragma omp parallel for
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            float val = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int ii = std::min(std::max(i + k, 0), img.height - 1);
                val += kernel[k + radius] * temp.data[ii * img.width + j];
            }
            result.data[i * img.width + j] = static_cast<unsigned char>(val);
        }
    }
    
    return result;
}

bool EdgeDetector::loadImage(const std::string& filename) {
    input_image = convertToGrayscale(filename);
    return input_image.data.size() > 0;
}

Image EdgeDetector::sobelEdgeDetection() {
    Image result;
    result.width = input_image.width;
    result.height = input_image.height;
    result.data.resize(input_image.width * input_image.height, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < input_image.height - 1; i++) {
        for (int j = 1; j < input_image.width - 1; j++) {
            int gx = 0, gy = 0;
            
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    unsigned char pixel = input_image.data[(i + ki) * input_image.width + (j + kj)];
                    gx += sobel_x[ki + 1][kj + 1] * pixel;
                    gy += sobel_y[ki + 1][kj + 1] * pixel;
                }
            }
            
            int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
            magnitude = std::min(255, std::max(0, magnitude));
            result.data[i * result.width + j] = static_cast<unsigned char>(magnitude);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sobel completed in " << duration.count() << " ms" << std::endl;
    
    return result;
}

Image EdgeDetector::cannyEdgeDetection(double low_threshold, double high_threshold) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Gaussian blur
    Image blurred = gaussianBlur(input_image, 5);
    
    // Step 2: Compute gradients
    Image grad_magnitude, grad_direction;
    grad_magnitude.width = input_image.width;
    grad_magnitude.height = input_image.height;
    grad_magnitude.data.resize(input_image.width * input_image.height, 0);
    grad_direction.width = input_image.width;
    grad_direction.height = input_image.height;
    grad_direction.data.resize(input_image.width * input_image.height, 0);
    
    #pragma omp parallel for
    for (int i = 1; i < input_image.height - 1; i++) {
        for (int j = 1; j < input_image.width - 1; j++) {
            int gx = 0, gy = 0;
            
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    unsigned char pixel = blurred.data[(i + ki) * blurred.width + (j + kj)];
                    gx += sobel_x[ki + 1][kj + 1] * pixel;
                    gy += sobel_y[ki + 1][kj + 1] * pixel;
                }
            }
            
            int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
            magnitude = std::min(255, std::max(0, magnitude));
            grad_magnitude.data[i * grad_magnitude.width + j] = static_cast<unsigned char>(magnitude);
            
            double angle = atan2(gy, gx) * 180.0 / M_PI;
            angle = fmod(angle + 180.0, 180.0);
            
            unsigned char direction = 0;
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180))
                direction = 0;
            else if (angle >= 22.5 && angle < 67.5)
                direction = 45;
            else if (angle >= 67.5 && angle < 112.5)
                direction = 90;
            else if (angle >= 112.5 && angle < 157.5)
                direction = 135;
                
            grad_direction.data[i * grad_direction.width + j] = direction;
        }
    }
    
    // Step 3: Non-Maximum Suppression
    Image nms_output;
    nms_output.width = input_image.width;
    nms_output.height = input_image.height;
    nms_output.data.resize(input_image.width * input_image.height, 0);
    
    #pragma omp parallel for
    for (int i = 1; i < input_image.height - 1; i++) {
        for (int j = 1; j < input_image.width - 1; j++) {
            unsigned char mag = grad_magnitude.data[i * grad_magnitude.width + j];
            unsigned char dir = grad_direction.data[i * grad_direction.width + j];
            
            unsigned char neighbor1 = 0, neighbor2 = 0;
            
            switch (dir) {
                case 0:
                    neighbor1 = grad_magnitude.data[i * grad_magnitude.width + (j - 1)];
                    neighbor2 = grad_magnitude.data[i * grad_magnitude.width + (j + 1)];
                    break;
                case 45:
                    neighbor1 = grad_magnitude.data[(i - 1) * grad_magnitude.width + (j + 1)];
                    neighbor2 = grad_magnitude.data[(i + 1) * grad_magnitude.width + (j - 1)];
                    break;
                case 90:
                    neighbor1 = grad_magnitude.data[(i - 1) * grad_magnitude.width + j];
                    neighbor2 = grad_magnitude.data[(i + 1) * grad_magnitude.width + j];
                    break;
                case 135:
                    neighbor1 = grad_magnitude.data[(i - 1) * grad_magnitude.width + (j - 1)];
                    neighbor2 = grad_magnitude.data[(i + 1) * grad_magnitude.width + (j + 1)];
                    break;
            }
            
            if (mag >= neighbor1 && mag >= neighbor2) {
                nms_output.data[i * nms_output.width + j] = mag;
            }
        }
    }
    
    // Step 4: Double thresholding
    std::vector<int> strong_pixels;
    
    #pragma omp parallel for
    for (int i = 0; i < nms_output.height; i++) {
        for (int j = 0; j < nms_output.width; j++) {
            unsigned char val = nms_output.data[i * nms_output.width + j];
            if (val >= high_threshold) {
                nms_output.data[i * nms_output.width + j] = 255;
                #pragma omp critical
                strong_pixels.push_back(i * nms_output.width + j);
            } else if (val >= low_threshold) {
                nms_output.data[i * nms_output.width + j] = 128;
            } else {
                nms_output.data[i * nms_output.width + j] = 0;
            }
        }
    }
    
    // Step 5: Hysteresis (sequential)
    std::vector<unsigned char> final_output = nms_output.data;
    
    for (int idx : strong_pixels) {
        int i = idx / nms_output.width;
        int j = idx % nms_output.width;
        
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int ni = i + di, nj = j + dj;
                if (ni >= 0 && ni < nms_output.height && nj >= 0 && nj < nms_output.width) {
                    if (final_output[ni * nms_output.width + nj] == 128) {
                        final_output[ni * nms_output.width + nj] = 255;
                    }
                }
            }
        }
    }
    
    for (unsigned char& val : final_output) {
        if (val == 128) val = 0;
    }
    
    Image result;
    result.width = nms_output.width;
    result.height = nms_output.height;
    result.data = final_output;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Canny completed in " << duration.count() << " ms" << std::endl;
    
    return result;
}

void EdgeDetector::saveResult(const std::string& filename, const Image& img) {
    stbi_write_png(filename.c_str(), img.width, img.height, 1, img.data.data(), img.width);
    std::cout << "Saved: " << filename << std::endl;
}

void EdgeDetector::printImageInfo() {
    std::cout << "Image: " << input_image.width << " x " << input_image.height 
              << ", Pixels: " << input_image.data.size() << std::endl;
}

TimingResult EdgeDetector::benchmarkParallel() {
    TimingResult result;
    
    std::cout << "\n=== BENCHMARK ===" << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Image: " << input_image.width << "x" << input_image.height << std::endl;
    
    // Serial Sobel
    omp_set_num_threads(1);
    auto start = std::chrono::high_resolution_clock::now();
    Image serial_sobel = sobelEdgeDetection();
    auto end = std::chrono::high_resolution_clock::now();
    double serial_sobel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Parallel Sobel
    omp_set_num_threads(num_threads);
    start = std::chrono::high_resolution_clock::now();
    Image parallel_sobel = sobelEdgeDetection();
    end = std::chrono::high_resolution_clock::now();
    double parallel_sobel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    result.sobel_time = parallel_sobel_time;
    result.sobel_speedup = serial_sobel_time / parallel_sobel_time;
    
    // Serial Canny
    omp_set_num_threads(1);
    start = std::chrono::high_resolution_clock::now();
    Image serial_canny = cannyEdgeDetection();
    end = std::chrono::high_resolution_clock::now();
    double serial_canny_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Parallel Canny
    omp_set_num_threads(num_threads);
    start = std::chrono::high_resolution_clock::now();
    Image parallel_canny = cannyEdgeDetection();
    end = std::chrono::high_resolution_clock::now();
    double parallel_canny_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    result.canny_time = parallel_canny_time;
    result.canny_speedup = serial_canny_time / parallel_canny_time;
    
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Sobel:  Serial=" << serial_sobel_time << "ms, Parallel=" << parallel_sobel_time << "ms, Speedup=" 
              << std::fixed << std::setprecision(2) << result.sobel_speedup << "x" << std::endl;
    std::cout << "Canny: Serial=" << serial_canny_time << "ms, Parallel=" << parallel_canny_time << "ms, Speedup=" 
              << result.sobel_speedup << "x" << std::endl;
    
    return result;
}
