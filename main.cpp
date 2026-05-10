#include "edge_detection.h"
#include <iostream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace std;

// Structure to store test results
struct SpeedupData {
    string test_name;
    int threads;
    double sobel_time_ms;
    double canny_time_ms;
    double sobel_speedup;
    double canny_speedup;
};

struct BenchmarkResults {
    string image_name;
    int width;
    int height;
    vector<SpeedupData> results;
};

void printHeader() {
    cout << "\n========================================" << endl;
    cout << "  PARALLEL EDGE DETECTION BENCHMARK" << endl;
    cout << "========================================" << endl;
}

void printComparisonTable(int threads, const string& image_name, int width, int height, 
                          double serial_sobel, double parallel_sobel,
                          double serial_canny, double parallel_canny) {
    cout << "\n========================================" << endl;
    cout << "  " << image_name << " (" << width << "x" << height << ")" << endl;
    cout << "  Threads: " << threads << endl;
    cout << "========================================" << endl;
    
    // Table header
    cout << left << setw(15) << "Algorithm" 
         << right << setw(22) << "Serial (1 thread)" 
         << right << setw(22) << ("Parallel (" + to_string(threads) + " threads)")
         << right << setw(12) << "Speedup" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    
    // Sobel row
    double sobel_speedup = (parallel_sobel > 0) ? serial_sobel / parallel_sobel : 0;
    cout << left << setw(15) << "Sobel" 
         << right << setw(22) << fixed << setprecision(4) << serial_sobel << " ms"
         << right << setw(22) << fixed << setprecision(4) << parallel_sobel << " ms"
         << right << setw(12) << fixed << setprecision(2) << sobel_speedup << "x" << endl;
    
    // Canny row
    double canny_speedup = (parallel_canny > 0) ? serial_canny / parallel_canny : 0;
    cout << left << setw(15) << "Canny" 
         << right << setw(22) << fixed << setprecision(4) << serial_canny << " ms"
         << right << setw(22) << fixed << setprecision(4) << parallel_canny << " ms"
         << right << setw(12) << fixed << setprecision(2) << canny_speedup << "x" << endl;
    
    cout << "--------------------------------------------------------------------------------" << endl;
}

void runBenchmarkForThreads(EdgeDetector& serial_detector, 
                           const string& image_path, const string& image_name, 
                           int width, int height, 
                           const vector<int>& thread_counts,
                           BenchmarkResults& benchmark) {
    
    benchmark.image_name = image_name;
    benchmark.width = width;
    benchmark.height = height;
    
    // First, get serial execution times (1 thread)
    cout << "\n>>> Running serial benchmark (1 thread) for " << image_name << "..." << endl;
    
    if (!serial_detector.loadImage(image_path)) {
        cerr << "Error: Failed to load image " << image_path << endl;
        return;
    }
    
    // Use microseconds for higher precision
    const int NUM_RUNS = 3;
    double serial_sobel_times = 0;
    double serial_canny_times = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        auto start = chrono::high_resolution_clock::now();
        Image serial_sobel = serial_detector.sobelEdgeDetection();
        auto end = chrono::high_resolution_clock::now();
        serial_sobel_times += chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        
        start = chrono::high_resolution_clock::now();
        Image serial_canny = serial_detector.cannyEdgeDetection();
        end = chrono::high_resolution_clock::now();
        serial_canny_times += chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
    }
    
    double avg_serial_sobel = serial_sobel_times / NUM_RUNS;
    double avg_serial_canny = serial_canny_times / NUM_RUNS;
    
    // Save serial results once
    Image serial_sobel_result = serial_detector.sobelEdgeDetection();
    Image serial_canny_result = serial_detector.cannyEdgeDetection();
    serial_detector.saveResult("results/" + image_name + "_serial_sobel.png", serial_sobel_result);
    serial_detector.saveResult("results/" + image_name + "_serial_canny.png", serial_canny_result);
    
    // Now run for each thread count
    for (int threads : thread_counts) {
        if (threads == 1) continue; // Skip serial as we already did it
        
        cout << ">>> Running benchmark with " << threads << " threads for " << image_name << "..." << endl;
        
        EdgeDetector parallel_detector(threads);
        
        if (!parallel_detector.loadImage(image_path)) {
            cerr << "Error: Failed to load image " << image_path << " for " << threads << " threads" << endl;
            continue;
        }
        
        // Run Sobel parallel (multiple runs)
        double parallel_sobel_times = 0;
        double parallel_canny_times = 0;
        
        for (int run = 0; run < NUM_RUNS; run++) {
            auto start = chrono::high_resolution_clock::now();
            Image parallel_sobel = parallel_detector.sobelEdgeDetection();
            auto end = chrono::high_resolution_clock::now();
            parallel_sobel_times += chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
            
            start = chrono::high_resolution_clock::now();
            Image parallel_canny = parallel_detector.cannyEdgeDetection();
            end = chrono::high_resolution_clock::now();
            parallel_canny_times += chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        }
        
        double avg_parallel_sobel = parallel_sobel_times / NUM_RUNS;
        double avg_parallel_canny = parallel_canny_times / NUM_RUNS;
        
        // Save parallel results for selected thread counts
        if (threads == 4 || threads == 8) {
            Image parallel_sobel_result = parallel_detector.sobelEdgeDetection();
            Image parallel_canny_result = parallel_detector.cannyEdgeDetection();
            parallel_detector.saveResult("results/" + image_name + "_parallel_" + to_string(threads) + "threads_sobel.png", parallel_sobel_result);
            parallel_detector.saveResult("results/" + image_name + "_parallel_" + to_string(threads) + "threads_canny.png", parallel_canny_result);
        }
        
        // Store results
        SpeedupData data;
        data.test_name = image_name;
        data.threads = threads;
        data.sobel_time_ms = avg_parallel_sobel;
        data.canny_time_ms = avg_parallel_canny;
        data.sobel_speedup = avg_serial_sobel / avg_parallel_sobel;
        data.canny_speedup = avg_serial_canny / avg_parallel_canny;
        benchmark.results.push_back(data);
        
        // Print individual comparison table
        printComparisonTable(threads, image_name, width, height, 
                           avg_serial_sobel, avg_parallel_sobel,
                           avg_serial_canny, avg_parallel_canny);
    }
    
    // Store serial baseline as threads=1
    SpeedupData serial_data;
    serial_data.test_name = image_name;
    serial_data.threads = 1;
    serial_data.sobel_time_ms = avg_serial_sobel;
    serial_data.canny_time_ms = avg_serial_canny;
    serial_data.sobel_speedup = 1.0;
    serial_data.canny_speedup = 1.0;
    benchmark.results.insert(benchmark.results.begin(), serial_data);
}

void printFullSummary(const vector<BenchmarkResults>& all_benchmarks) {
    cout << "\n\n";
    cout << "################################################################################\n";
    cout << "                    COMPLETE BENCHMARK SUMMARY\n";
    cout << "################################################################################\n";
    
    for (const auto& benchmark : all_benchmarks) {
        cout << "\n========================================\n";
        cout << "Image: " << benchmark.image_name << " (" << benchmark.width << "x" << benchmark.height << ")\n";
        cout << "========================================\n";
        
        cout << left << setw(12) << "Threads"
             << right << setw(20) << "Sobel (ms)"
             << right << setw(20) << "Sobel Speedup"
             << right << setw(20) << "Canny (ms)"
             << right << setw(20) << "Canny Speedup" << endl;
        cout << "--------------------------------------------------------------------------------\n";
        
        for (const auto& result : benchmark.results) {
            cout << left << setw(12) << result.threads
                 << right << setw(20) << fixed << setprecision(4) << result.sobel_time_ms
                 << right << setw(20) << fixed << setprecision(2) << result.sobel_speedup << "x"
                 << right << setw(20) << fixed << setprecision(4) << result.canny_time_ms
                 << right << setw(20) << fixed << setprecision(2) << result.canny_speedup << "x" << endl;
        }
        cout << "--------------------------------------------------------------------------------\n";
    }
}

void saveResultsToCSV(const vector<BenchmarkResults>& all_benchmarks) {
    ofstream csv_file("results/benchmark_results.csv");
    
    if (!csv_file.is_open()) {
        cerr << "Warning: Could not create CSV file" << endl;
        return;
    }
    
    // Write header
    csv_file << "Image,Width,Height,Threads,Sobel_Time_ms,Sobel_Speedup,Canny_Time_ms,Canny_Speedup\n";
    
    // Write data with high precision
    csv_file << fixed << setprecision(6);
    for (const auto& benchmark : all_benchmarks) {
        for (const auto& result : benchmark.results) {
            csv_file << benchmark.image_name << ","
                    << benchmark.width << ","
                    << benchmark.height << ","
                    << result.threads << ","
                    << result.sobel_time_ms << ","
                    << result.sobel_speedup << ","
                    << result.canny_time_ms << ","
                    << result.canny_speedup << "\n";
        }
    }
    
    csv_file.close();
    cout << "\n✓ Results saved to results/benchmark_results.csv" << endl;
}

int main() {
    printHeader();
    
    // Define thread counts to benchmark
    vector<int> thread_counts = {1, 2, 4, 6, 8, 10};
    
    cout << "\nBenchmark Configuration:" << endl;
    cout << "   Thread counts to test: ";
    for (size_t i = 0; i < thread_counts.size(); i++) {
        cout << thread_counts[i];
        if (i < thread_counts.size() - 1) cout << ", ";
    }
    cout << endl;
    cout << "   Each test will run 3 times for accuracy" << endl;
    cout << "   Results will be averaged with microsecond precision" << endl;
    cout << "   Times are displayed in milliseconds (4 decimal places)" << endl;
    
    // Create serial detector (1 thread) for baseline
    EdgeDetector serial_detector(1);
    
    // Define test images
    vector<tuple<string, string, int, int>> test_images = {
        {"test_images/small.png", "small", 400, 300},
        {"test_images/medium.png", "medium", 1080, 720},
        {"test_images/large.png", "large", 1920, 1080}
    };
    
    vector<BenchmarkResults> all_benchmarks;
    
    // Run benchmarks for each image
    for (const auto& img : test_images) {
        string image_path = get<0>(img);
        string image_name = get<1>(img);
        int width = get<2>(img);
        int height = get<3>(img);
        
        // Check if image exists
        ifstream f(image_path.c_str());
        if (!f.good()) {
            cout << "\n⚠ Warning: " << image_path << " not found. Skipping..." << endl;
            continue;
        }
        f.close();
        
        BenchmarkResults benchmark;
        runBenchmarkForThreads(serial_detector, image_path, image_name, 
                              width, height, thread_counts, benchmark);
        all_benchmarks.push_back(benchmark);
    }
    
    if (all_benchmarks.empty()) {
        cout << "\n❌ No test images found! Please run: bash get_test_images.sh" << endl;
        return 1;
    }
    
    // Print complete summary
    printFullSummary(all_benchmarks);
    
    // Save results to CSV for further analysis
    saveResultsToCSV(all_benchmarks);
    
    cout << "\n========================================" << endl;
    cout << "BENCHMARK COMPLETE!" << endl;
    cout << "========================================" << endl;
    cout << "Results saved in results/ folder" << endl;
    cout << "  - Images with edge detection results" << endl;
    cout << "  - benchmark_results.csv for spreadsheet analysis" << endl;
    cout << "\nYou can now analyze the speedup across different thread counts." << endl;
    cout << "========================================\n" << endl;
    
    return 0;
}