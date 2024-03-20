#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "include/yolo/inference.h"

using namespace std;
using namespace cv;
const int TARGET_SIZE = 640;
const int MAX_STRIDE = 32;

cv::Mat resize_and_pad_image(cv::Mat input_image, int target_size) {
	cv::Mat resized_image, output_image;
	float scale = std::min((float)target_size / input_image.cols, (float)target_size / input_image.rows);
	cv::resize(input_image, resized_image, cv::Size(), scale, scale, cv::INTER_LINEAR);

	int top = (target_size - resized_image.rows) / 2;
	int bottom = target_size - resized_image.rows - top;
	int left = (target_size - resized_image.cols) / 2;
	int right = target_size - resized_image.cols - left;

	cv::copyMakeBorder(resized_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

	return output_image;
}

int main(int argc, char **argv)
{
    std::string projectBasePath = "/mnt/d/JetsonXavierProject/ultralytics"; // Set your ultralytics base path

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //



	cv::VideoCapture cap("/mnt/c/Users/user/CLionProjects/JetsonCpp/data/2023-03-23_11-19-46.mp4");
	if (!cap.isOpened()) {
		std::cout << "Error: Could not open the video file." << std::endl;
		return -1;
	}
	double frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);   // Get the width of the video
	double frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT); // Get the height of the video
	double fps = cap.get(cv::CAP_PROP_FPS);  // Get the FPS of the input video
	double total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

	std::cout << std::endl;
	std::cout << "Video Info:" << std::endl;
	std::cout << "Frame Width: " << frame_width << std::endl;
	std::cout << "Frame Height: " << frame_height << std::endl;
	std::cout << "FPS: " << fps << std::endl;
	std::cout << "Frames: " << total_frames << std::endl;


	// Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
	Inference inf(projectBasePath + "/yolov8s_640x640.onnx", cv::Size(TARGET_SIZE, TARGET_SIZE), "classes.txt", runOnGPU);

	cv::Mat frame;
	cv::Mat resized_img;
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			break;
		}
        // Inference starts here...
		resized_img = resize_and_pad_image(frame, TARGET_SIZE);
        std::vector<Detection> output = inf.runInference(resized_img);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

			const int PADDING_TOP = 140; // 	int top = (target_size - resized_image.rows) / 2;
			const float SCALE = 0.33; //  	float scale = std::min((float)target_size / input_image.cols, (float)target_size / input_image.rows);
			box.y -= PADDING_TOP;

			// Масштабируем bbox обратно к размерам исходного изображения
			box.x = box.x / SCALE;
			box.y = box.y / SCALE;
			box.width = box.width / SCALE;
			box.height = box.height / SCALE;
            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        cv::waitKey(-1);
    }
}
