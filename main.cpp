//
// Created by xuyufeng1 on 2021/3/23.
//
#include <chrono>
#include <string>
#include "yolov5/Loader.h"
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "cuda_runtime.h"
#include "npp.h"

#include "DeepSORT/track_deepsort.h"
using namespace std;

void make_input(const cv::Mat& img, void* gpu_data_planes, int count);

cv::Rect get_rect(cv::Size size, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (size.width * 1.0);
    float r_h = INPUT_H / (size.height * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * size.height) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * size.height) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * size.width) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * size.width) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    // x
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {

        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f- (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f- (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;

    } else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f- (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

void get_detections(DETECTBOX box, float confidence, int type, DETECTIONS &d, int timestamp) {
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);
    tmpRow.type = type;
    tmpRow.timestamp = timestamp;
    tmpRow.shelter = false;
    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}

int main(int argc, char** argv) {
    string engineFileName = "../yolov5s.engine";
    if (argc == 2 && string(argv[1]) == "-s") {
        yolov5_build_engine("../yolov5/yolov5s.wts", engineFileName, 0.33, 0.50);
    }
    else if (argc == 3 && string(argv[1]) == "-v") {
        if (!access("./result", F_OK)) {
            system("rm -rf /home/xuyufeng/projects/cpp/yolov5/cmake-build-debug/result");
        }
        mkdir("./result",  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        string filename = string(argv[2]);
        cv::VideoCapture cap;
        cap.open(filename);
        if (!cap.isOpened()) {
            std::cout << "failed to open video " << filename << std::endl;
            return -1;
        }
        int frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
        int maxIndex = int(frameCount / BATCH_SIZE);

        yolov5_detector detector(engineFileName);
        auto fps = cap.get(cv::CAP_PROP_FPS);
        auto size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("fps[%d], \n", fps);
        vector<cv::Mat > images;
        static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        void* gpu_data_planes;
        cudaMalloc(&gpu_data_planes, BATCH_SIZE * INPUT_W * INPUT_H * 3 * sizeof(float));
        int timestamp = 0;
        track_deepsort track;
        for (int f = 0; f < 10; f++) {
            auto start = std::chrono::system_clock::now();
            vector<cv::Mat> imgs;
            for (int b = 0; b < BATCH_SIZE; b++) {
                cv::Mat img;
                cap.read(img);
                cv::Mat copy_img = img.clone();
                images.push_back(copy_img);
                make_input(img, gpu_data_planes, b);
            }

            //detect
            auto end = std::chrono::system_clock::now();
            std::cout << "Preprocess Time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/BATCH_SIZE
                      << "ms" << std::endl;
            start = std::chrono::system_clock::now();
            std::vector<std::vector<Yolo::Detection>> results = detector.detect(gpu_data_planes);
            end = std::chrono::system_clock::now();
            std::cout << "Detect Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/BATCH_SIZE
                      << "ms" << std::endl;

            for (int b = 0; b < BATCH_SIZE; b++) {
                auto& res = results[b];
                auto& image = images[b];

                //std::cout << res.size() << std::endl;
                for (size_t j = 0; j < res.size(); j++) {
                    cv::Rect r = get_rect(size, res[j].bbox);
                    cv::rectangle(image, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(image, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                }
                cv::imwrite("./result/a_" + to_string(f) + "_" +  to_string(b) + ".jpg", image);

                // Track
                DETECTIONS detections;
                for (int i = 0; i < res.size(); i++) {
                    cv::Rect r = get_rect(size, res[i].bbox);
                    int x = r.x;
                    int y = r.y;
                    int w = r.width;
                    int h = r.height;
                    if (x >= size.width || x < 0 || y >= size.height || y < 0) {
                        continue;
                    }
                    get_detections(DETECTBOX(x, y, w, h), res[i].conf, res[i].class_id, detections, timestamp);
                }
                timestamp += 1;
                track.run(detections);
            }
            images.clear();
        }
        detector.release();
        cap.release();
    }
    return  0;
}

void make_input(const cv::Mat& img, void* gpu_data_planes, int count)
{
    void *gpu_img_buf, *gpu_img_resize_buf, *gpu_data_buf;
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

    int width_in = img.cols;
    int height_in = img.rows;

    uchar* img_data = img.data;

    cudaMalloc(&gpu_img_buf, width_in * height_in * 3 * sizeof(uchar));
    cudaMalloc(&gpu_img_resize_buf, INPUT_W * INPUT_H * 3 * sizeof(uchar));
    cudaMalloc(&gpu_data_buf, INPUT_W * INPUT_H * 3 * sizeof(float));

    Npp32f m_scale[3] = {0.00392157, 0.00392157, 0.00392157};
    //Npp32f a_scale[3] = {-1, -1, -1};
    Npp32f* r_plane = (Npp32f*)(gpu_data_planes + count * 3 * INPUT_H * INPUT_W*sizeof(float));
    Npp32f* g_plane = (Npp32f*)(gpu_data_planes + INPUT_W*INPUT_H*sizeof(float) + count * 3 * INPUT_H * INPUT_W*sizeof(float));
    Npp32f* b_plane = (Npp32f*)(gpu_data_planes + INPUT_W*INPUT_H*2*sizeof(float) + count * 3 * INPUT_H * INPUT_W*sizeof(float));
    Npp32f* dst_planes[3] = {r_plane, g_plane, b_plane};
    int aDstOrder[3] = {2, 1, 0};


    NppiSize srcSize = {width_in, height_in};
    NppiRect srcROI = {0, 0, width_in, height_in};
    NppiSize dstSize = {INPUT_W, INPUT_H};
    NppiRect dstROI = {x, y, w, h};

    cudaMemcpy(gpu_img_buf, img_data, width_in*height_in*3, cudaMemcpyHostToDevice);
    nppiResize_8u_C3R((Npp8u*)gpu_img_buf, width_in*3, srcSize, srcROI,
                      (Npp8u*)gpu_img_resize_buf, INPUT_W*3, dstSize, dstROI,
                      NPPI_INTER_LINEAR);      //resize

    nppiSwapChannels_8u_C3IR((Npp8u*)gpu_img_resize_buf, INPUT_W*3, dstSize, aDstOrder);   //rbg2bgr
    nppiConvert_8u32f_C3R((Npp8u*)gpu_img_resize_buf, INPUT_W*3, (Npp32f*)gpu_data_buf, INPUT_W*3*sizeof(float), dstSize);  //转换成浮点型
    nppiMulC_32f_C3IR(m_scale, (Npp32f*)gpu_data_buf, INPUT_W*3*sizeof(float), dstSize);    //缩放
    //nppiAddC_32f_C3IR(a_scale, (Npp32f*)this->gpu_data_buf, this->INPUT_W*3*sizeof(float), dstSize);
    nppiCopy_32f_C3P3R((Npp32f*)gpu_data_buf, INPUT_W*3*sizeof(float), dst_planes, INPUT_W*sizeof(float), dstSize); //	四通道32位浮点打包到平面图像副本

    cudaFree(gpu_img_buf);
    cudaFree(gpu_img_resize_buf);
    cudaFree(gpu_data_buf);
}

