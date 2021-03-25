#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5.h"

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
float yolov5_detector::prob[BATCH_SIZE * OUTPUT_SIZE];
static float gw;
static float gd;

int get_width(int x, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}

int get_depth(int x) {
    if (x == 1) {
        return 1;
    }
    else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

ICudaEngine* build_engine(std::string weightFile,unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(weightFile);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128), get_width(128), get_depth(3), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256), get_width(256), get_depth(9), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512), get_width(512), get_depth(9), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024), get_width(1024), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024), get_width(1024), get_depth(3), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512), 1, 1, 1, "model.10");

    float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * get_width(512) * 2 * 2));
    for (int i = 0; i < get_width(512) * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, get_width(512) * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), get_width(512), DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(get_width(512));
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024), get_width(512), get_depth(3), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256), 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, get_width(256) * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), get_width(256), DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(get_width(256));
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512), get_width(256), get_depth(3), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512), get_width(512), get_depth(3), false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024), get_width(1024), get_depth(3), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}



void APIToModel(std::string weightFile, unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    //ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
    ICudaEngine* engine = build_engine(weightFile, maxBatchSize, builder, config, DataType::kFLOAT);

    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, void* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

yolov5_detector::yolov5_detector(std::string engine_name) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr };
    size_t size{ 0 };

    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));
}

void yolov5_detector::release() {
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

std::vector<std::vector<Yolo::Detection>> yolov5_detector::detect(void* data) {
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    std::vector<std::vector<Yolo::Detection>> batch_res(BATCH_SIZE);
    for (int b = 0; b < BATCH_SIZE; b++) {
        auto& res = batch_res[b];
        nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
    }
    return batch_res;
}


int yolov5_build_engine(std::string weightFile, std::string engineFile,float depth_multiple, float width_multiple) {
    gw = width_multiple;
    gd = depth_multiple;
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    std::cout << -1 << std::endl;
    //std::string engine_name = STR2(NET);
    //engine_name = "yolov5" + engine_name + ".engine";
    std::string engine_name = engineFile;
    
    std::cout << -2 << std::endl;
    IHostMemory* modelStream{ nullptr };
    std::cout << -3 << std::endl;
    APIToModel(weightFile,BATCH_SIZE, &modelStream);
    std::cout << 0 << std::endl;
    assert(modelStream != nullptr);
    std::cout << 0 << std::endl;
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;
}
