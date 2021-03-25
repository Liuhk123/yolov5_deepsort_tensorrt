#include "tracker.h"
#include "nn_matching.h"
#include "../DeepAppearanceDescriptor/model.h"
#include "linear_assignment.h"
using namespace std;

//#define MY_inner_DEBUG
#ifdef MY_inner_DEBUG
#include <string>
#include <iostream>
#endif

tracker::tracker(/*NearNeighborDisMetric *metric,*/
		float max_cosine_distance, int nn_budget,
		float max_iou_distance, int max_age, int n_init)
{
    this->metric = new NearNeighborDisMetric(
    		NearNeighborDisMetric::METRIC_TYPE::cosine,
    		max_cosine_distance, nn_budget);
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;

    this->kf = new KalmanFilter();
    this->tracks.clear();
    this->_next_idx = 1;
}

void tracker::predict()
{
    printf("卡尔曼滤波预测track数目为%d：",tracks.size());
    for(Track& track:tracks) {
        track.predit(kf);
    }
}

std::vector<Track> tracker::update(const DETECTIONS &detections) {
    TRACHER_MATCHD res;
    _match(detections, res);

    vector<MATCH_DATA>& matches = res.matches;

    for(MATCH_DATA& data:matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        //#ifdef MY_inner_DEBUG
        //        printf("\t%d == %d;\n", track_idx, detection_idx);
        //#endif
        tracks[track_idx].update(this->kf, detections[detection_idx], detections[detection_idx].type);
    }
    vector<int>& unmatched_tracks = res.unmatched_tracks;
    //#ifdef MY_inner_DEBUG
    //    printf("res.unmatched_tracks size = %d\n", unmatched_tracks.size());
    //#endif
    //对于没匹配上的track，标记为missed，输出
    std::vector<Track> missed_tracks;
    for(int& track_idx:unmatched_tracks) {
        this->tracks[track_idx].mark_missed();
        missed_tracks.push_back(this->tracks[track_idx]);
    }
    vector<int>& unmatched_detections = res.unmatched_detections;
    //#ifdef MY_inner_DEBUG
    //    printf("res.unmatched_detections size = %d\n", unmatched_detections.size());
    //#endif
    for(int& detection_idx:unmatched_detections) {
        this->_initiate_track(detections[detection_idx]);
    }
    //#ifdef MY_inner_DEBUG
    //    int size = tracks.size();
    //    printf("now track size = %d\n", size);
    //#endif
    vector<Track>::iterator it;
    for(it = tracks.begin(); it != tracks.end();) {
        if((*it).is_deleted()) it = tracks.erase(it);
        else ++it;
    }
    return missed_tracks;
}

void tracker::_match(const DETECTIONS &detections, TRACHER_MATCHD &res)
{
    vector<int> confirmed_tracks;
    vector<int> unconfirmed_tracks;
    int idx = 0;
    for(Track& t:tracks) {
        if(t.is_confirmed()) confirmed_tracks.push_back(idx);
        else unconfirmed_tracks.push_back(idx);
        idx++;
    }

//    TRACHER_MATCHD matcha = linear_assignment::getInstance()->matching_cascade(
//                this, &tracker::gated_matric,
//                this->metric->mating_threshold,
//                this->max_age,
//                this->tracks,
//                detections,
//                confirmed_tracks);
    TRACHER_MATCHD matcha;
    vector<int> iou_track_candidates;
    iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
    vector<int>::iterator it;		
    for(it = confirmed_tracks.begin(); it != confirmed_tracks.end();) {
        int idx = *it;
        if(tracks[idx].time_since_update == 1) { //push into unconfirmed
            iou_track_candidates.push_back(idx);
            it = confirmed_tracks.erase(it);
            continue;
        } else {
            matcha.unmatched_tracks.push_back(idx);
            it = confirmed_tracks.erase(it);
            continue;
        }
        ++it;
    }
    for(int i=0;i<detections.size();i++){
        matcha.unmatched_detections.push_back(i);
    }

    TRACHER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
                this, &tracker::iou_cost,
                this->max_iou_distance,
                this->tracks,
                detections,
                iou_track_candidates,
                matcha.unmatched_detections);
    //get result:
    res.matches.assign(matchb.matches.begin(), matchb.matches.end());
//    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
    //unmatched_tracks;
    res.unmatched_tracks.assign(
                matcha.unmatched_tracks.begin(),
                matcha.unmatched_tracks.end());
    res.unmatched_tracks.insert(
                res.unmatched_tracks.end(),
                matchb.unmatched_tracks.begin(),
                matchb.unmatched_tracks.end());
    res.unmatched_detections.assign(
                matchb.unmatched_detections.begin(),
                matchb.unmatched_detections.end());
}

void tracker::_initiate_track(const DETECTION_ROW &detection)
{
    KAL_DATA data = kf->initiate(detection.to_xyah());
    KAL_MEAN mean = data.first;
    KAL_COVA covariance = data.second;
    int type = detection.type;

    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->n_init,
                                 this->max_age, type, detection));
    _next_idx += 1;
}

DYNAMICM
tracker::iou_cost(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices)
{
    //!!!python diff: track_indices && detection_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }
    //    if(detection_indices.empty() == true) {
    //        for(size_t i = 0; i < dets.size(); i++) {
    //            detection_indices.push_back(i);
    //        }
    //    }
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for(int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        if(tracks[track_idx].time_since_update > 1) {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
            continue;
        }
        DETECTBOX bbox = tracks[track_idx].to_tlwh();
        int csize = detection_indices.size();
        DETECTBOXSS candidates(csize, 4);
        for(int k = 0; k < csize; k++) candidates.row(k) = dets[detection_indices[k]].tlwh;
        Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
        cost_matrix.row(i) = rowV;
    }
    return cost_matrix;
}

Eigen::VectorXf
tracker::iou(DETECTBOX& bbox, DETECTBOXSS& candidates)
{
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];

    Eigen::Matrix<float, -1, 2> candidates_tl;
    Eigen::Matrix<float, -1, 2> candidates_br;
    candidates_tl = candidates.leftCols(2) ;
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    //    Eigen::VectorXf area_intersection(size);
    //    Eigen::VectorXf area_candidates(size);
    Eigen::VectorXf res(size);
    for(int i = 0; i < size; i++) {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1; w = (w < 0? 0: w);
        float h = br_2 - tl_2; h = (h < 0? 0: h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection/(area_bbox + area_candidates - area_intersection);
    }
    //#ifdef MY_inner_DEBUG
    //        std::cout << res << std::endl;
    //#endif
    return res;
}

