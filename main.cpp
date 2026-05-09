#include "edge_detection.h"
#include <iostream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace std;

// Structure to store test results
struct SpeedupData {
    string test_name;
    double sobel_speedup;
    double canny_speedup;
};

// Function to check if file exists
bool fileExists(const string& filename) {
    ifstream f(filename.c_str());
    return f.good();
}

void printHeader() {
    cout << "\n========================================" << endl;
    cout << "  PARALLEL EDGE DETECTION FRAMEWORK" << endl;
    cout << "========================================" << endl;
}

void showMenu() {
    cout << "\n========================================" << endl;
    cout << "           MAIN MENU" << endl;
    cout << "========================================" << endl;
    cout << "  1. Process Single Image" << endl;
    cout << "     - Small (400x300)" << endl;
    cout << "     - Medium (1080x720)" << endl;
    cout << "     - Large (1920x1080)" << endl;
    cout << "     - Custom image" << endl;
    cout << endl;
    cout << "  2. Run All Test Cases with Comparison" << endl;
    cout << "     (Shows Serial vs Parallel speedup)" << endl;
    cout << endl;
    cout << "  3. Exit" << endl;
    cout << "========================================" << endl;
    cout << "Choice: ";
}

double printComparisonTable(const string& image_name, int width, int height, 
                          int serial_sobel, int parallel_sobel,
                          int serial_canny, int parallel_canny) {
    cout << "\n========================================" << endl;
    cout << "  " << image_name << " (" << width << "x" << height << ")" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // Table header
    cout << left << setw(15) << "Algorithm" 
         << right << setw(18) << "Serial (1 thread)" 
         << right << setw(18) << "Parallel (2 threads)" 
         << right << setw(12) << "Speedup" << endl;
    cout << "--------------------------------------------------------------" << endl;
    
    // Sobel row
    double sobel_speedup = (double)serial_sobel / parallel_sobel;
    cout << left << setw(15) << "Sobel" 
         << right << setw(18) << to_string(serial_sobel) + " ms"
         << right << setw(18) << to_string(parallel_sobel) + " ms"
         << right << setw(12) << fixed << setprecision(2) << sobel_speedup << "x" << endl;
    
    // Canny row
    double canny_speedup = (double)serial_canny / parallel_canny;
    cout << left << setw(15) << "Canny" 
         << right << setw(18) << to_string(serial_canny) + " ms"
         << right << setw(18) << to_string(parallel_canny) + " ms"
         << right << setw(12) << fixed << setprecision(2) << canny_speedup << "x" << endl;
    
    cout << "--------------------------------------------------------------" << endl;
    cout << "\nResults saved: results/" << image_name << "_*.png" << endl;
    
    // Return the speedup values for summary
    return sobel_speedup;
}

void runSingleImageComparison(EdgeDetector& serial_detector, EdgeDetector& parallel_detector, 
                              const string& image_path, const string& image_name, 
                              int width, int height, double& sobel_speedup, double& canny_speedup) {
    
    // Serial execution (1 thread)
    serial_detector.loadImage(image_path);
    
    // Run Sobel serial
    auto start = chrono::high_resolution_clock::now();
    Image serial_sobel = serial_detector.sobelEdgeDetection();
    auto end = chrono::high_resolution_clock::now();
    int serial_sobel_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    // Run Canny serial
    start = chrono::high_resolution_clock::now();
    Image serial_canny = serial_detector.cannyEdgeDetection();
    end = chrono::high_resolution_clock::now();
    int serial_canny_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    // Save serial results
    serial_detector.saveResult("results/" + image_name + "_serial_sobel.png", serial_sobel);
    serial_detector.saveResult("results/" + image_name + "_serial_canny.png", serial_canny);
    
    // Parallel execution (2 threads)
    parallel_detector.loadImage(image_path);
    
    // Run Sobel parallel
    start = chrono::high_resolution_clock::now();
    Image parallel_sobel = parallel_detector.sobelEdgeDetection();
    end = chrono::high_resolution_clock::now();
    int parallel_sobel_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    // Run Canny parallel
    start = chrono::high_resolution_clock::now();
    Image parallel_canny = parallel_detector.cannyEdgeDetection();
    end = chrono::high_resolution_clock::now();
    int parallel_canny_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    // Save parallel results
    parallel_detector.saveResult("results/" + image_name + "_parallel_sobel.png", parallel_sobel);
    parallel_detector.saveResult("results/" + image_name + "_parallel_canny.png", parallel_canny);
    
    // Calculate speedups
    sobel_speedup = (double)serial_sobel_time / parallel_sobel_time;
    canny_speedup = (double)serial_canny_time / parallel_canny_time;
    
    // Print comparison table
    printComparisonTable(image_name, width, height, 
                         serial_sobel_time, parallel_sobel_time,
                         serial_canny_time, parallel_canny_time);
}

void processSingleImage(EdgeDetector& serial_detector, EdgeDetector& parallel_detector) {
    int imgChoice;
    string image_path;
    string image_name;
    int width, height;
    double sobel_speedup, canny_speedup;
    
    cout << "\n--- Select Image Size ---" << endl;
    cout << "1. Small (400x300)" << endl;
    cout << "2. Medium (1080x720)" << endl;
    cout << "3. Large (1920x1080)" << endl;
    cout << "Choice: ";
    cin >> imgChoice;
    
    switch(imgChoice) {
        case 1:
            image_path = "test_images/small.png";
            image_name = "small";
            width = 400;
            height = 300;
            break;
        case 2:
            image_path = "test_images/medium.png";
            image_name = "medium";
            width = 1080;
            height = 720;
            break;
        case 3:
            image_path = "test_images/large.png";
            image_name = "large";
            width = 1920;
            height = 1080;
            break;
        default:
            cout << "Invalid choice!" << endl;
            return;
    }
    
    cout << "\n========================================" << endl;
    cout << "Processing: " << image_name << endl;
    cout << "========================================" << endl;
    
    runSingleImageComparison(serial_detector, parallel_detector, 
                             image_path, image_name, width, height, 
                             sobel_speedup, canny_speedup);
}

void runAllTestCases(EdgeDetector& serial_detector, EdgeDetector& parallel_detector) {
    cout << "\n========================================" << endl;
    cout << "RUNNING ALL TEST CASES WITH COMPARISON" << endl;
    cout << "Serial vs Parallel" << endl;
    cout << "========================================" << endl;
    
    // Store results for summary
    vector<SpeedupData> results;
    
    // Test Case 1: Small Image
    cout << "\n>>> TEST CASE 1: Small Image (400x300) <<<" << endl;
    double small_sobel, small_canny;
    runSingleImageComparison(serial_detector, parallel_detector, 
                            "test_images/small.png", "test1_small", 400, 300,
                            small_sobel, small_canny);
    results.push_back({"Small (400x300)", small_sobel, small_canny});
    
    // Test Case 2: Medium Image
    cout << "\n>>> TEST CASE 2: Medium Image (1080x720) <<<" << endl;
    double medium_sobel, medium_canny;
    runSingleImageComparison(serial_detector, parallel_detector, 
                            "test_images/medium.png", "test2_medium", 1080, 720,
                            medium_sobel, medium_canny);
    results.push_back({"Medium (1080x720)", medium_sobel, medium_canny});
    
    // Test Case 3: Large Image
    cout << "\n>>> TEST CASE 3: Large Image (1920x1080) <<<" << endl;
    double large_sobel, large_canny;
    runSingleImageComparison(serial_detector, parallel_detector, 
                            "test_images/large.png", "test3_large", 1920, 1080,
                            large_sobel, large_canny);
    results.push_back({"Large (1920x1080)", large_sobel, large_canny});
    
    cout << "\n========================================" << endl;
    cout << "ALL TEST CASES COMPLETE" << endl;
    cout << "Results saved in results/ folder" << endl;
    cout << "========================================" << endl;
    
    // Summary table with actual values
    cout << "\n========================================" << endl;
    cout << "SUMMARY: SPEEDUP COMPARISON" << endl;
    cout << "========================================" << endl;
    cout << left << setw(22) << "Test Case" 
         << right << setw(12) << "Sobel" 
         << right << setw(12) << "Canny" << endl;
    cout << "----------------------------------------" << endl;
    
    for (const auto& r : results) {
        cout << left << setw(22) << r.test_name 
             << right << setw(12) << fixed << setprecision(2) << r.sobel_speedup << "x"
             << right << setw(12) << fixed << setprecision(2) << r.canny_speedup << "x" << endl;
    }
    
    cout << "========================================" << endl;
}

int main(int argc, char* argv[]) {
    printHeader();
    
    // Get number of threads from command line (default: 2 for parallel)
    int threads = 2;
    if (argc > 1) threads = atoi(argv[1]);
    
    cout << "\nConfiguration:" << endl;
    cout << "   Parallel will use: " << threads << " threads" << endl;
    cout << "   Serial will use: 1 thread" << endl;
    
    // Create results directory
    system("mkdir -p results");
    
    // Create detectors (serial with 1 thread, parallel with user threads)
    EdgeDetector serial_detector(1);
    EdgeDetector parallel_detector(threads);
    
    // Command line mode: if user provided image directly
    if (argc > 2) {
        string image_file = argv[2];
        string image_name = "custom";
        double dummy1, dummy2;
        if (image_file.find("small") != string::npos) image_name = "small";
        else if (image_file.find("medium") != string::npos) image_name = "medium";
        else if (image_file.find("large") != string::npos) image_name = "large";
        
        runSingleImageComparison(serial_detector, parallel_detector, 
                                 image_file, image_name, 0, 0, dummy1, dummy2);
        return 0;
    }
    
    // Interactive mode
    int choice;
    do {
        showMenu();
        cin >> choice;
        
        switch(choice) {
            case 1:
                processSingleImage(serial_detector, parallel_detector);
                break;
            case 2:
                runAllTestCases(serial_detector, parallel_detector);
                break;
            case 3:
                cout << "\n Exiting... Thank you for using Parallel Edge Detection!" << endl;
                break;
            default:
                cout << "Invalid choice! Please enter 1-3" << endl;
        }
    } while (choice != 3);
    
    return 0;
}
