#ifndef TRACKCOUNTER_YOLOV5_H_
#define TRACKCOUNTER_YOLOV5_H_

#include <vector>
#include "yololayer.h"
#include "NvInfer.h"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.3
#define CONF_THRESH 0.45
#define BATCH_SIZE 16
#define WEIGHT "../yolov5/weights/yolov5.wts"

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

#endif

using namespace nvinfer1;

int yolov5_build_engine(std::string weightFile, std::string engineFile, float depth_multiple, float width_multiple);

class yolov5_detector {
private:
    IExecutionContext* context;
	cudaStream_t stream;
	void* buffers[2];
	static float prob[BATCH_SIZE * (Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1)];
	ICudaEngine* engine;
	IRuntime* runtime;


public:
	yolov5_detector(std::string engine_name);
	std::vector<std::vector<Yolo::Detection>> detect(void* data);
	void release();
};
