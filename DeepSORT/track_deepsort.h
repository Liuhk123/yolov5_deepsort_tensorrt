//
// Created by xuyufeng1 on 2021/3/25.
//

#ifndef TRACKCOUNTER_TRACK_DEEPSORT_H
#define TRACKCOUNTER_TRACK_DEEPSORT_H

#include "KalmanFilter/tracker.h"
#include <vector>
#include <map>
#include <set>

#define NN_BUDGET 100
#define MAX_COSINE_DISTANCE 0.2

class track_deepsort {
public:
    track_deepsort();
    ~track_deepsort() = default;
    void run(DETECTIONS& detections);

private:
    std::vector<Track> nextReleaseTrack;
    std::map<int, std::set<std::string>> timestamplist;
    tracker mytracker;
    int missed_num;
};


#endif //TRACKCOUNTER_TRACK_DEEPSORT_H
