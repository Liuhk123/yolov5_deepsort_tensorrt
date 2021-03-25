#pragma once
#ifndef YOLOV5_LOADER_H_
#define YOLOV5_LOADER_H_
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "yolov5.h"

#endif

#define FPS 30
#define INPUT_H Yolo::INPUT_H
#define INPUT_W Yolo::INPUT_W

std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method);
void preProcessImg(cv::Mat* img, float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W]);
cv::Mat preprocess_img(cv::Mat& img);

class LoadVideo {
public:

    cv::VideoCapture cap;
	int index = 0;
	int maxIndex;

	LoadVideo(std::string filename);

	float* getBatch();

	int getLoops();

	void release() {
		this->cap.release();
	}
};

class WebCam {
private:
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	cv::VideoCapture cap;
public:

	WebCam(int port);

	float* getBatch();

	cv::Mat getFrame();

	void release() {
		this->cap.release();
	}
};