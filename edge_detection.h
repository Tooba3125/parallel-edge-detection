#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <vector>
#include <string>
#include <chrono>

struct Image {
    int width;
    int height;
    std::vector<unsigned char> data;  // Grayscale values 0-255
};

struct TimingResult {
    double sobel_time;
    double canny_time;
    double sobel_speedup;
    double canny_speedup;
};

class EdgeDetector {
private:
    Image input_image;
    int num_threads;
    
    // Sobel kernels
    const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    // Helper functions
    Image convertToGrayscale(const std::string& filename);
    Image gaussianBlur(const Image& img, int kernel_size = 5);
    
public:
    EdgeDetector(int threads = 4);
    
    // Load image from file
    bool loadImage(const std::string& filename);
    
    // Parallel Sobel edge detection
    Image sobelEdgeDetection();
    
    // Parallel Canny edge detection
    Image cannyEdgeDetection(double low_threshold = 50, double high_threshold = 150);
    
    // Benchmark
    TimingResult benchmarkParallel();
    
    // Save results
    void saveResult(const std::string& filename, const Image& img);
    
    // Get image info
    void printImageInfo();
    int getWidth() const { return input_image.width; }
    int getHeight() const { return input_image.height; }
};

#endif
