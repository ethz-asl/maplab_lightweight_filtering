/*
 * FilterBase.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef FilterBase_HPP_
#define FilterBase_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
#include <map>
#include "Prediction.hpp"

namespace LWF{

template<typename State>
class FilterState{
 public:
  typedef State mtState;
  typedef typename mtState::CovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  mtCovMat cov_;
  mtState state_;
  double t_;
  FilterState(){
    t_ = 0.0;
  };
  virtual ~FilterState(){};
};

template<typename State>
class FilterBase{
 public:
  typedef State mtState;
  typedef typename mtState::CovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  FilterState<State> safe_;
  FilterState<State> front_;
  bool validFront_;
  PredictionBase<mtState>* mpDefaultPrediction_;

  std::map<double,PredictionBase<mtState>*> predictionMap_;
  std::map<double,UpdateBase<mtState>*> updateMap_; //TODO: adapt
  FilterBase(){
    validFront_ = false;
    mpDefaultPrediction_ = nullptr;
  };
  virtual ~FilterBase(){};
  void updateSafe(){
    if(predictionMap_.empty() || updateMap_.empty()) return;
    double nextSafeTime = std::min(predictionMap_.rbegin()->first,updateMap_.rbegin()->first);
    if(front_.t_<=nextSafeTime && validFront_ && front_.t_>safe_.t_){
      safe_.state_ = front_.state_;
      safe_.cov_ = front_.cov_;
    }
    update(safe_,nextSafeTime);
    clean(nextSafeTime);
  }
  void update(FilterState<State>& filterState,const double& tEnd){
    typename std::map<double,PredictionBase<mtState>>::iterator itPredictionMeas;
    typename std::map<double,UpdateBase<mtState>>::iterator itUpdateMeas;
    itPredictionMeas = predictionMap_.upper_bound(filterState.t_);
    itUpdateMeas = updateMap_.upper_bound(filterState.t_);
    double tNext = filterState.t_;
    PredictionBase<mtState>* mpUsedPrediction;
    while(filterState.t_<tEnd){
      bool isFullUpdate = false;
      tNext = tEnd;
      if(itPredictionMeas!=predictionMap_.end()){
        if(itPredictionMeas->first<tNext){
          tNext = itPredictionMeas->first;
        }
        mpUsedPrediction = itPredictionMeas->second;
      } else {
        mpUsedPrediction = mpDefaultPrediction_;
      }
      if(itUpdateMeas!=updateMap_.end()){
        if(itUpdateMeas->first<=tNext){
          tNext = itUpdateMeas->first;
          isFullUpdate = true;
        }
      }
      if(mpUsedPrediction!=nullptr){
        mpUsedPrediction->predictEKF(filterState.state_,filterState.cov_,tNext-filterState.t_);
      } else {
        LWF::PredictionDefault<State>::predictEKF(filterState.state_,filterState.cov_,tNext-filterState.t_);
      }
      filterState.t_ = tNext;
      itUpdateMeas->updateEKF(filterState.state_,filterState.cov_);
      itPredictionMeas = predictionMap_.upper_bound(filterState.t_);
      itUpdateMeas = updateMap_.upper_bound(filterState.t_);
    }
  }
  void clean(const double& t){
    while(!predictionMap_.empty() && predictionMap_.begin()->first<=t){
      delete predictionMap_.begin()->second; // TODO: Careful: assumes ownership
      predictionMap_.erase(predictionMap_.begin());
    }
    while(!updateMap_.empty() && updateMap_.begin()->first<=t){
      delete updateMap_.begin()->second; // TODO: Careful: assumes ownership
      updateMap_.erase(updateMap_.begin());
    }
  }
  void addPrediction(PredictionBase<mtState>* mpPrediction, const double& t){
    assert(t>safe_.t_);
    predictionMap_[t] = mpPrediction; // TODO: Careful: takes over ownership
    if(front_.t_>=t) validFront_ = false;
  }
  void addUpdateMeas(UpdateBase<mtState>* mpUpdate, const double& t){
    assert(t>safe_.t_);
    updateMap_[t] = mpUpdate; // TODO: Careful: takes over ownership
    if(front_.t_>=t) validFront_ = false;
  }
};

}

#endif /* FilterBase_HPP_ */
