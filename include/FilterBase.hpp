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

class DefaultAuxillary{
 public:
  DefaultAuxillary(){};
  ~DefaultAuxillary(){};
};

template<typename State, typename Auxillary = DefaultAuxillary>
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
  PredictionBase<mtState>* mpDefaultPrediction_;

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
  virtual ~FilterBase(){
    delete mpDefaultPrediction_;
    while(!predictionMap_.empty()){
      delete predictionMap_.begin()->second;
      predictionMap_.erase(predictionMap_.begin());
    }
    for(unsigned int i=0;i<nUT_;i++){
      while(!updateMap_[i].empty()){
        delete updateMap_[i].begin()->second;
        updateMap_[i].erase(updateMap_[i].begin());
      }
    }
  };
  void resetFilter(){
    safe_ = init_;
    front_ = init_;
    validFront_ = false;
  }
  void setDefaultPrediction(PredictionBase<mtState>* mpPrediction){
    mpDefaultPrediction_ = mpPrediction; // TODO: Careful: takes over ownership
  }
  void removeDefaultPrediction(){
    delete mpDefaultPrediction_;
    mpDefaultPrediction_ = nullptr;
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
    typename std::map<double,PredictionBase<mtState>*>::iterator itPrediction;
    typename std::map<double,UpdateBase<mtState>*>::iterator itUpdate[nUT_];
    itPrediction = predictionMap_.upper_bound(filterState.t_);
    for(unsigned int i=0;i<nUT_;i++){
      itUpdate[i] = updateMap_[i].upper_bound(filterState.t_);
    }
    double tNext = filterState.t_;
    int availableCoupledPrediction = -1;
    PredictionBase<mtState>* mpUsedPrediction;
    while(filterState.t_<tEnd){
      tNext = tEnd;
      if(itPrediction!=predictionMap_.end()){
        if(itPrediction->first<tNext){
          tNext = itPrediction->first;
        }
        mpUsedPrediction = itPrediction->second;
      } else {
        mpUsedPrediction = mpDefaultPrediction_;
      }
      for(unsigned int i=0;i<nUT_;i++){
        if(itUpdate[i]!=updateMap_[i].end() && itUpdate[i]->first<=tNext){
          if(itUpdate[i]->first<tNext){
            tNext = itUpdate[i]->first;
            availableCoupledPrediction = -1;
          }
          if(itUpdate[i]->second->isCoupledToPrediction_){
            if(availableCoupledPrediction==-1) availableCoupledPrediction = i;
            else std::cout << "ERROR: multiple coupled updates";
          }
        }
      }
      if(mpUsedPrediction!=nullptr){
        if(availableCoupledPrediction==-1){
          if(mpUsedPrediction->predictEKF(filterState.state_,filterState.cov_,tNext-filterState.t_)!=0) std::cout << "ERROR in predictEKF";
        }
        else if(itUpdate[availableCoupledPrediction]->second->predictAndUpdateEKF(filterState.state_,filterState.cov_,mpUsedPrediction,tNext-filterState.t_)!=0) std::cout << "ERROR in predictAndUpdateEKF";
      }
      filterState.t_ = tNext;
      for(unsigned int i=0;i<nUT_;i++){
        if(itUpdate[i]!=updateMap_[i].end() && itUpdate[i]->first<=tNext && i!=availableCoupledPrediction){
          if(itUpdate[i]->second->updateEKF(filterState.state_,filterState.cov_)!=0) std::cout << "ERROR in updateEKF";
        }
      }
      itPrediction = predictionMap_.upper_bound(filterState.t_);
      for(unsigned int i=0;i<nUT_;i++){
        itUpdate[i] = updateMap_[i].upper_bound(filterState.t_);
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
