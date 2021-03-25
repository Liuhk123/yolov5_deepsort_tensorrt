#include <iostream>
#include "Loader.h"
#include "yololayer.h"

using namespace std;
using namespace cv;

std::string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
	return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
		std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
		"/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, "  + "format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


LoadVideo::LoadVideo(string filename) {
    std::cout<<"Loading video: "<<filename<<std::endl;
	cap.open(filename);
	if (!cap.isOpened()) {
		std::cout << "failed to open video " << filename << std::endl;
		return;
	}
	int frameCount = cap.get(CAP_PROP_FRAME_COUNT);
	maxIndex = int(frameCount / BATCH_SIZE);
}

float* LoadVideo::getBatch() {
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	cv::Mat img;
	for (int b = 0; b < BATCH_SIZE; b++) {
		cap.read(img);
		cv::Mat pr_img = preprocess_img(img);
		int i = 0;
		for (int row = 0; row < INPUT_H; ++row) {
			uchar* uc_pixel = pr_img.data + row * pr_img.step;
			for (int col = 0; col < INPUT_W; ++col) {
				data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
				uc_pixel += 3;
				++i;
			}
		}
	}
	return data;
}

int LoadVideo::getLoops() {
	return maxIndex;
}

WebCam::WebCam(int port) {
	std::string pipeline = gstreamer_pipeline(Yolo::ORIGINAL_W,
		Yolo::ORIGINAL_H,
		Yolo::ORIGINAL_W,
		Yolo::ORIGINAL_H,
		FPS,
		0);
	std::cout << "Using pipeline: \n\t" << pipeline << "\n";

	cout << "00" << endl;

	this->cap = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);

	//this->cap.open(port);
	cout << "01"<< endl;

	if (!this->cap.isOpened()) {
		std::cout << "Failed to open camera." << std::endl;
	}

}

cv::Mat WebCam::getFrame() {
	cv::Mat img;
	while (!this->cap.read(img));
	return img;
}

float* WebCam::getBatch() {
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	cv::Mat img;
	for (int b = 0; b < BATCH_SIZE; b++) {
		if (cap.read(img)) {
			cv::Mat pr_img = preprocess_img(img);
			int i = 0;
			for (int row = 0; row < INPUT_H; ++row) {
				uchar* uc_pixel = pr_img.data + row * pr_img.step;
				for (int col = 0; col < INPUT_W; ++col) {
					data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
					data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
					data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
					uc_pixel += 3;
					++i;
				}
			}
		}
	}
	return data;
}



void preProcessImg(cv::Mat img[BATCH_SIZE], float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W]) {
	for (int b = 0; b < BATCH_SIZE; b++) {
		cv::Mat pr_img = preprocess_img(img[b]);
		int i = 0;
		for (int row = 0; row < INPUT_H; ++row) {
			uchar* uc_pixel = pr_img.data + row * pr_img.step;
			for (int col = 0; col < INPUT_W; ++col) {
				data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
				uc_pixel += 3;
				++i;
			}
		}
	}
}
