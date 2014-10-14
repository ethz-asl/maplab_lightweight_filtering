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

class AuxillaryData{
 public:
  AuxillaryData(){};
  ~AuxillaryData(){};
};

template<typename State, typename Auxillary = AuxillaryData>
class FilterState{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  typedef Auxillary mtAuxillary;
  mtCovMat cov_;
  mtState state_;
  mtAuxillary aux_;
  double t_;
  FilterState(){
    t_ = 0.0;
    cov_.setIdentity();
    state_.setIdentity();
  };
  virtual ~FilterState(){};
};

template<typename State, unsigned int nUpdType = 1>
class FilterBase{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  static const unsigned int nUT_ = nUpdType;
  FilterState<State> safe_;
  FilterState<State> front_;
  FilterState<State> init_;
  bool validFront_;
  PredictionBase<mtState>* mpDefaultPrediction_; // TODO

  std::map<double,PredictionBase<mtState>*> predictionMap_;
  std::map<double,UpdateBase<mtState>*> updateMap_[nUT_];
  double maxWaitTime[nUT_];
  FilterBase(){
    validFront_ = false;
    mpDefaultPrediction_ = nullptr;
    for(unsigned int i = 0; i<nUT_;i++){
      maxWaitTime[i] = 1.0;
    }
  };
  virtual ~FilterBase(){};
  void resetFilter(){
    safe_ = init_;
    front_ = init_;
    validFront_ = false;
  }
  bool getSafeTime(double& safeTime){
    double maxPredictionTime;
    // Get maximal time where prediction is available
    if(!predictionMap_.empty()){
      maxPredictionTime = predictionMap_.rbegin()->first;
    } else {
      return false;
    }
    safeTime = maxPredictionTime;
    // Check if we have to wait for update measurements
    double updateTime;
    for(unsigned int i=0;i<nUT_;i++){
      if(!updateMap_[i].empty()){
        updateTime = updateMap_[i].rbegin()->first;
      } else {
        updateTime = safe_.t_;
      }
      if(updateTime < maxPredictionTime - maxWaitTime[i]){
        updateTime = maxPredictionTime - maxWaitTime[i];
      }
      if(updateTime < safeTime){
        safeTime = updateTime;
      }
    }
    if(safeTime <= safe_.t_) return false;
    return true;
  }
  void updateSafe(){
    double nextSafeTime;
    if(!getSafeTime(nextSafeTime)) return;
    if(front_.t_<=nextSafeTime && validFront_ && front_.t_>safe_.t_){
      safe_ = front_;
    }
    update(safe_,nextSafeTime);
    clean(nextSafeTime);
  }
  void updateFront(const double& tEnd){
    updateSafe();
    if(!validFront_ || front_.t_<=safe_.t_){
      front_ = safe_;
    }
    update(front_,tEnd);
  }
  void update(FilterState<State>& filterState,const double& tEnd){
    typename std::map<double,PredictionBase<mtState>*>::iterator itPredictionMeas;
    typename std::map<double,UpdateBase<mtState>*>::iterator itUpdateMeas[nUT_];
    itPredictionMeas = predictionMap_.upper_bound(filterState.t_);
    for(unsigned int i=0;i<nUT_;i++){
      itUpdateMeas[i] = updateMap_[i].upper_bound(filterState.t_);
    }
    double tNext = filterState.t_;
    PredictionBase<mtState>* mpUsedPrediction;
    while(filterState.t_<tEnd){
      tNext = tEnd;
      if(itPredictionMeas!=predictionMap_.end()){
        if(itPredictionMeas->first<tNext){
          tNext = itPredictionMeas->first;
        }
        mpUsedPrediction = itPredictionMeas->second;
      } else {
        mpUsedPrediction = mpDefaultPrediction_;
      }
      for(unsigned int i=0;i<nUT_;i++){
        if(itUpdateMeas[i]!=updateMap_[i].end() && itUpdateMeas[i]->first<=tNext){
          tNext = itUpdateMeas[i]->first;
        }
      }
      if(mpUsedPrediction!=nullptr){
        mpUsedPrediction->predictEKF(filterState.state_,filterState.cov_,tNext-filterState.t_);
      }
      filterState.t_ = tNext;
      for(unsigned int i=0;i<nUT_;i++){
        if(itUpdateMeas[i]!=updateMap_[i].end() && itUpdateMeas[i]->first<=tNext){
          itUpdateMeas[i]->second->updateEKF(filterState.state_,filterState.cov_);
        }
      }
      itPredictionMeas = predictionMap_.upper_bound(filterState.t_);
      for(unsigned int i=0;i<nUT_;i++){
        itUpdateMeas[i] = updateMap_[i].upper_bound(filterState.t_);
      }
    }
  }
  void clean(const double& t){
    while(!predictionMap_.empty() && predictionMap_.begin()->first<=t){
      delete predictionMap_.begin()->second; // TODO: Careful: assumes ownership
      predictionMap_.erase(predictionMap_.begin());
    }
    for(unsigned int i=0;i<nUT_;i++){
      while(!updateMap_[i].empty() && updateMap_[i].begin()->first<=t){
        delete updateMap_[i].begin()->second; // TODO: Careful: assumes ownership
        updateMap_[i].erase(updateMap_[i].begin());
      }
    }
  }
  void addPrediction(PredictionBase<mtState>* mpPrediction, const double& t){
    assert(t>safe_.t_);
    predictionMap_[t] = mpPrediction; // TODO: Careful: takes over ownership
    if(front_.t_>=t) validFront_ = false;
  }
  void addUpdate(UpdateBase<mtState>* mpUpdate, const double& t,unsigned int type = 0){
    assert(type<nUT_);
    assert(t>safe_.t_);
    updateMap_[type][t] = mpUpdate; // TODO: Careful: takes over ownership
    if(front_.t_>=t) validFront_ = false;
  }
};

}

#endif /* FilterBase_HPP_ */
