## yolov5_deepsort_tensorrt

This repository uses yolov5 and TensorRT to deploy object tracking algorithms. Deep-sort algorithm are used to track the objects.Thank you all for  the contribution of tensorrtx and deepsort.This project is all implemented in C++ language, and the preprocessing code of the detection part is run on the GPU using npp programming.The cost time of the inference is 3ms, and the preprocessing time is 6ms.

## environment
Tesla T4
cuda 11.1
TensorRT 7.2.2.3

## Install

```shell
git clone https://github.com/xuyufeng1995/yolov5_deepsort_tensorrt.git
cd tensorrt_yolov5_tracker
mkdir build
cd build
cmake ..
make
```

## Usage

You could change the batch size in `yolov5/yolov5.h` You could change the input image size in `yolov5/yololayer.h`

We provide the support for building models of yolov5-4.0 in our project. The default branch is corresponding to yolov5-4.0. 

If other versions are wanted, please kindly refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx) to build the engine and then copy to the folder of this project.

### Build engine

At first, please put the `yolov5/gen_wts.py` in the folder of corresponding version of [**Yolov5**](https://github.com/ultralytics/yolov5). For example, if you use the model of yolov5-4.0, please download the release of yolov5-4.0 and put the `gen_wts.py` in its folder. Then run the code to convert model from `.pt` to `.wts`. Then please use the following instructions to build your engines. This code will generate yolov5s engine file and filename is yolov5s.engine.

```
./Tracker -s # build yolov5 model

```

### Process video

```
./Tracker -v ../huangxing.mp4 # run
```

## Reference

[**tensorrtx**](https://github.com/wang-xinyu/tensorrtx)

[**Yolov5**](https://github.com/ultralytics/yolov5)

[**deepsort**](https://github.com/shaoshengsong/DeepSORT)

[**tensorrt_yolov5_tracker**](https://github.com/AsakusaRinne/tensorrt_yolov5_tracker)
