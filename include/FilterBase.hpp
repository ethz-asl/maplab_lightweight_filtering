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
  typedef typename mtState::mtCovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  mtCovMat cov_;
  mtState state_;
  double t_;
  FilterState(){
    t_ = 0.0;
    cov_.setIdentity();
    state_.setIdentity();
  };
  virtual ~FilterState(){};
};

template<typename State>
class UpdateManagerBase{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  UpdateManagerBase(){
    maxWaitTime_ = 1.0;
  };
  virtual ~UpdateManagerBase(){};
  virtual bool getNextUpdateTime(double tCurrent, double& nextUpdateTime) = 0;
  virtual void clean(double t) = 0;
  virtual void constrainSafeUpdateTime(double maxPredictionTime, double& safeUpdateTime) = 0;
  virtual void update(FilterState<mtState>& filterState) = 0;
  double maxWaitTime_;
};

template<typename Update>
class UpdateManager: public UpdateManagerBase<typename Update::mtState>{
 public:
  using UpdateManagerBase<typename Update::mtState>::maxWaitTime_;
  typedef typename Update::mtState mtState;
  typedef typename Update::mtMeas mtMeas;
  typedef typename mtState::mtCovMat mtCovMat;
  std::map<double,mtMeas> updateMeasMap_;
  Update update_;
  typename std::map<double,mtMeas>::iterator itMeas_;
  UpdateManager(){};
  ~UpdateManager(){};
  void addMeas(const mtMeas& meas, const double& t){
//    assert(t>safe_.t_); // TODO
//    if(front_.t_>=t) validFront_ = false;
    updateMeasMap_[t] = meas;
  }
  void clean(double t){
    while(!updateMeasMap_.empty() && updateMeasMap_.begin()->first<=t){
      updateMeasMap_.erase(updateMeasMap_.begin());
    }
  }
  bool getNextUpdateTime(double tCurrent, double& nextUpdateTime){
    itMeas_ = updateMeasMap_.upper_bound(tCurrent);
    if(itMeas_!=updateMeasMap_.end()){
      nextUpdateTime = itMeas_->first;
      return true;
    } else {
      return false;
    }
  }
  void constrainSafeUpdateTime(double maxPredictionTime, double& safeUpdateTime){
    double updateTime = maxPredictionTime-maxWaitTime_;
    if(!updateMeasMap_.empty() && updateMeasMap_.rbegin()->first > updateTime){
      updateTime = updateMeasMap_.rbegin()->first;
    }
    if(safeUpdateTime > updateTime){
      safeUpdateTime = updateTime;
    }
  }
  void update(FilterState<mtState>& filterState){
    if(updateMeasMap_.count(filterState.t_)){
      int r = update_.updateEKF(filterState.state_,filterState.cov_,updateMeasMap_[filterState.t_]);
      if(r!=0) std::cout << "Error during update: " << r << std::endl;
    }
  }
  // predictAndUpdate
};

template<typename Prediction, typename DefaultPrediction = Prediction>
class PredictionManager{
 public:
  typedef typename Prediction::mtState mtState;
  typedef typename Prediction::mtCovMat mtCovMat;
  typedef typename Prediction::mtMeas mtMeas;
  std::map<double,mtMeas> predictionMeasMap_;
  Prediction prediction_;
  DefaultPrediction defaultPrediction_;
  typename std::map<double,mtMeas>::iterator itMeas_;
  PredictionManager(){};
  ~PredictionManager(){};
  void addMeas(const mtMeas& meas, const double& t){
//    assert(t>safe_.t_); // TODO
//    if(front_.t_>=t) validFront_ = false;
    predictionMeasMap_[t] = meas;
  }
  void clean(double t){
    while(!predictionMeasMap_.empty() && predictionMeasMap_.begin()->first<=t){
      predictionMeasMap_.erase(predictionMeasMap_.begin());
    }
  }
  bool getMaxPredictionTime(double& maxPredictionTime){
    if(!predictionMeasMap_.empty()){
      maxPredictionTime = predictionMeasMap_.rbegin()->first;
      return true;
    } else {
      return false;
    }
  }
  void predict(FilterState<mtState>& filterState, double tNext){
    while(filterState.t_<tNext){
      itMeas_ = predictionMeasMap_.upper_bound(filterState.t_);
      if(itMeas_ != predictionMeasMap_.end()){
        int r = prediction_.predictEKF(filterState.state_,filterState.cov_,itMeas_->second,itMeas_->first-filterState.t_);
        if(r!=0) std::cout << "Error during prediction: " << r << std::endl;
        filterState.t_ = itMeas_->first;
      } else {
        mtMeas meas; // TODO = mtMeas::Identity();
        int r = defaultPrediction_.predictEKF(filterState.state_,filterState.cov_,meas,tNext-filterState.t_);
        if(r!=0) std::cout << "Error during prediction: " << r << std::endl;
        filterState.t_ = tNext;
      }
    }
  }
  void predictAndUpdate(FilterState<mtState>& filterState, UpdateManagerBase<mtState>* updateManagerBase, double tNext){
    while(filterState.t_<tNext){
      itMeas_ = predictionMeasMap_.upper_bound(filterState.t_);
      if(itMeas_ != predictionMeasMap_.end()){
        int r = prediction_.predictEKF(filterState.state_,filterState.cov_,itMeas_->second,itMeas_->first-filterState.t_);
        if(r!=0) std::cout << "Error during prediction: " << r << std::endl;
        filterState.t_ = itMeas_->first;
      } else { // Drop error and ignore update
//        mtMeas meas; // TODO = mtMeas::Identity();
//        int r = defaultPrediction_.predictEKF(filterState.state_,filterState.cov_,meas,tNext-filterState.t_);
//        if(r!=0) std::cout << "Error during prediction: " << r << std::endl;
//        filterState.t_ = tNext;
      }
    }
  }
};

template<typename Prediction, unsigned int nUpdType = 1>
class FilterBase{
 public:
  typedef Prediction mtPrediction;
  typedef typename mtPrediction::mtState mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  static const unsigned int nUT_ = nUpdType;
  FilterState<mtState> safe_;
  FilterState<mtState> front_;
  FilterState<mtState> init_;
  bool validFront_;
  PredictionManager<mtPrediction> predictionManager_;
  UpdateManagerBase<mtState>* updateManagerBase_[nUT_];
  FilterBase(){
    validFront_ = false;
    for(unsigned int i = 0; i<nUT_;i++){
      updateManagerBase_[i] = nullptr;
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
    if(!predictionManager_.getMaxPredictionTime(maxPredictionTime)){
      return false;
    }
    safeTime = maxPredictionTime;
    // Check if we have to wait for update measurements
    for(unsigned int i=0;i<nUT_;i++){
      if(updateManagerBase_[i] != nullptr) updateManagerBase_[i]->constrainSafeUpdateTime(maxPredictionTime,safeTime);
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
  void update(FilterState<mtState>& filterState,const double& tEnd){
    double tNext = filterState.t_;
    double tNextUpdate;
    int availableCoupledPrediction = -1;
    while(filterState.t_<tEnd){
      tNext = tEnd;
      for(unsigned int i=0;i<nUT_;i++){
        if(updateManagerBase_[i] != nullptr){
          if(updateManagerBase_[i]->getNextUpdateTime(tNext,tNextUpdate) && tNextUpdate <= tNext){
            if(tNextUpdate<tNext){
              tNext = tNextUpdate;
              availableCoupledPrediction = -1;
            }
//            if(itUpdate[i]->second->isCoupledToPrediction_){ // TODO
//              if(availableCoupledPrediction==-1) availableCoupledPrediction = i;
//              else std::cout << "ERROR: multiple coupled updates";
//            }
          }
        }
      }
      if(availableCoupledPrediction==-1){
        predictionManager_.predict(filterState,tNext);
      } else {
        predictionManager_.predictAndUpdate(filterState,updateManagerBase_[availableCoupledPrediction],tNext);
      }
      for(unsigned int i=0;i<nUT_;i++){
        if(updateManagerBase_[i] != nullptr && i!=availableCoupledPrediction){
          updateManagerBase_[i]->update(filterState);
        }
      }
    }
  }
  void clean(const double& t){
    predictionManager_.clean(t);
    for(unsigned int i=0;i<nUT_;i++){
      if(updateManagerBase_[i] != nullptr) updateManagerBase_[i]->clean(t);
    }
  }
};

}

#endif /* FilterBase_HPP_ */
