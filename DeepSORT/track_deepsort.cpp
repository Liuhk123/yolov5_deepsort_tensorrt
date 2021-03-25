//
// Created by xuyufeng1 on 2021/3/25.
//

#include "track_deepsort.h"
#include <chrono>
#include <iostream>

#define MIN_HEIGHT 96 //输出最优帧的最小高度
#define MIN_WIDTH 48 //输出最优帧的最小宽度
#define MIN_HITS 20 //输出最优帧的最小匹配次数

using namespace std;

track_deepsort::track_deepsort() : mytracker(MAX_COSINE_DISTANCE, NN_BUDGET), missed_num(0) {
    //mytracker = tracker(MAX_COSINE_DISTANCE, NN_BUDGET);
}

void track_deepsort::run(DETECTIONS &detections) {
    std::vector<int> ReleaseCacheFrames;
    //遮挡计算
    for (int i = 0; i < detections.size(); i++) {
        DETECTION_ROW &detectionRow = detections[i];
        if (detectionRow.shelter) continue;
        DETECTBOX tlwh = detectionRow.tlwh;
        int x1 = tlwh.array()[0];
        int y1 = tlwh.array()[1];
        int w = tlwh.array()[2];
        int h = tlwh.array()[3];
        int x2 = x1 + w;
        int y2 = y1 + h;
        int s = w * h;
        for (int j = i + 1; j < detections.size(); j++) {
            DETECTION_ROW _detectionRow = detections[j];
            DETECTBOX _tlwh = _detectionRow.tlwh;
            int _x1 = _tlwh.array()[0];
            int _y1 = _tlwh.array()[1];
            int _w = _tlwh.array()[2];
            int _h = _tlwh.array()[3];
            int _x2 = _x1 + _w;
            int _y2 = _y1 + _h;
            int startx = min(x1, _x1);
            int starty = min(y1, _y1);
            int endx = max(x2, _x2);
            int endy = max(y2, _y2);
            int width = w + _w - (endx - startx);  // 重叠部分宽
            int height = h + _h - (endy - starty);  // 重叠部分高
            if (width > 0 and height > 0) {
                int s_iou = width * height;
                if (s_iou / s >= 0.3) {
                    detections[j].shelter = true;
                    detections[i].shelter = true;
                    break;
                }
            }
        }
    }
    //处理上一帧输出的最优帧的残余数据
    for (Track &track: nextReleaseTrack) {
        std::map<int, std::set<std::string>>::iterator iter;
        iter = timestamplist.find(track.best_timestamp);
        if (iter != timestamplist.end()) {
            std::set<std::string> track_uuid_list = iter->second;
            if (track_uuid_list.count(track.uuid) != 0) {
                track_uuid_list.erase(track.uuid);
            }
            if (track_uuid_list.size() == 0) {
                timestamplist.erase(iter);
                ReleaseCacheFrames.push_back(track.best_timestamp);
            } else {
                timestamplist.erase(iter);
                timestamplist.insert(std::pair<int, std::set<std::string>>(track.best_timestamp, track_uuid_list));
            }
        }
    }
    nextReleaseTrack.clear();
    std::vector<DETECTION_ROW> filter_tracks;
    std::set<std::string> cached_track_list;


    auto trackStart = std::chrono::system_clock::now();
//        卡尔曼滤波更新
    mytracker.predict();
    std::vector<Track> missed_tracks = mytracker.update(detections);


    //输出消失track的最优帧
    for (Track &track: missed_tracks) {
        missed_num++;
        nextReleaseTrack.push_back(track);
        if (track.best_timestamp == -1) continue;
        DETECTBOX tlwh = track.best_img.tlwh;
        int h = tlwh.array()[3];
        int w = tlwh.array()[2];
        std::cout << "release track uuid:" << track.uuid << std::endl;
        if (track.hits > MIN_HITS and h > MIN_HEIGHT and w > MIN_WIDTH) {
            filter_tracks.push_back(track.best_img);
        }
        //对于每一个该帧输出的track，在下个时间戳应该释放掉包含该track的缓存帧

    }
    std::cout<<"missed num:"<<missed_num<<std::endl;
    std::cout<<"tracks num:"<<mytracker.tracks.size()<<std::endl;
}


