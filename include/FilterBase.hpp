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

class MeasurementTimelineBase{
 public:
  MeasurementTimelineBase(){};
  virtual ~MeasurementTimelineBase(){};
  virtual bool getNextTime(double actualTime, double& nextTime) = 0;
  virtual void clean(double t) = 0;
  virtual void constrainTime(double actualTime, double& time) = 0;
  virtual bool getLastTime(double& lastTime) = 0;
  virtual bool hasMeasurementAt(double t) = 0;
};

template<typename Meas>
class MeasurementTimeline: public virtual MeasurementTimelineBase{
 public:
  typedef Meas mtMeas;
  MeasurementTimeline(){
    maxWaitTime_ = 1.0;
  };
  virtual ~MeasurementTimeline(){};
  std::map<double,mtMeas> measMap_;
  typename std::map<double,mtMeas>::iterator itMeas_;
  double maxWaitTime_;
  void addMeas(const mtMeas& meas, const double& t){
//    assert(t>safe_.t_); // TODO
//    if(front_.t_>=t) validFront_ = false;
    measMap_[t] = meas;
  }
  void clean(double t){
    while(!measMap_.empty() && measMap_.begin()->first<=t){
      measMap_.erase(measMap_.begin());
    }
  }
  bool getNextTime(double actualTime, double& nextTime){ // TODO rename functions
    itMeas_ = measMap_.upper_bound(actualTime);
    if(itMeas_!=measMap_.end()){
      nextTime = itMeas_->first;
      return true;
    } else {
      return false;
    }
  }
  void constrainTime(double actualTime, double& time){
    double measurementTime = actualTime-maxWaitTime_;
    if(!measMap_.empty() && measMap_.rbegin()->first > measurementTime){
      measurementTime = measMap_.rbegin()->first;
    }
    if(time > measurementTime){
      time = measurementTime;
    }
  }
  bool getLastTime(double& lastTime){
    if(!measMap_.empty()){
      lastTime = measMap_.rbegin()->first;
      return true;
    } else {
      return false;
    }
  }
  bool hasMeasurementAt(double t){
    return measMap_.count(t)>0;
  }
};

template<typename State>
class UpdateManagerBase: public virtual MeasurementTimelineBase{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  UpdateManagerBase(bool coupledToPrediction = false): coupledToPrediction_(coupledToPrediction){};
  virtual ~UpdateManagerBase(){};
  virtual void update(FilterState<mtState>& filterState) = 0;
  const bool coupledToPrediction_;
};

template<typename Update>
class UpdateManager: public MeasurementTimeline<typename Update::mtMeas>,public UpdateManagerBase<typename Update::mtState>{
 public:
  using MeasurementTimeline<typename Update::mtMeas>::measMap_;
  typedef typename Update::mtState mtState;
  typedef typename Update::mtMeas mtMeas;
  typedef typename mtState::mtCovMat mtCovMat;
  Update update_;
  UpdateManager(){};
  ~UpdateManager(){};
  void update(FilterState<mtState>& filterState){
    if(this->hasMeasurementAt(filterState.t_)){
      int r = update_.updateEKF(filterState.state_,filterState.cov_,measMap_[filterState.t_]);
      if(r!=0) std::cout << "Error during update: " << r << std::endl;
    }
  }
};

template<typename State, typename Prediction>
class UpdateAndPredictManagerBase: public virtual UpdateManagerBase<State>{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  UpdateAndPredictManagerBase(): UpdateManagerBase<State>(true){};
  virtual ~UpdateAndPredictManagerBase(){};
  virtual void predictAndUpdate(FilterState<mtState>& filterState, Prediction& prediction, const typename Prediction::mtMeas& predictionMeas, double dt) = 0;
};

template<typename Update, typename Prediction>
class UpdateAndPredictManager:public MeasurementTimeline<typename Update::mtMeas>, public UpdateAndPredictManagerBase<typename Update::mtState, Prediction>{
 public:
  using MeasurementTimeline<typename Update::mtMeas>::measMap_;
  typedef typename Update::mtState mtState;
  typedef typename Update::mtMeas mtMeas;
  typedef typename mtState::mtCovMat mtCovMat;
  Update update_;
  UpdateAndPredictManager(){};
  ~UpdateAndPredictManager(){};
  void update(FilterState<mtState>& filterState){
    assert(0);
  }
  void predictAndUpdate(FilterState<mtState>& filterState, Prediction& prediction, const typename Prediction::mtMeas& predictionMeas, double dt){
    if(hasMeasurementAt(filterState.t_)){
      int r = update_.predictAndUpdateEKF(filterState.state_,filterState.cov_,measMap_[filterState.t_],prediction,predictionMeas,dt);
      if(r!=0) std::cout << "Error during update: " << r << std::endl;
    }
  }
};

template<typename Prediction, typename DefaultPrediction = Prediction>
class PredictionManager: public MeasurementTimeline<typename Prediction::mtMeas>{
 public:
  using MeasurementTimeline<typename Prediction::mtMeas>::measMap_;
  using MeasurementTimeline<typename Prediction::mtMeas>::itMeas_;
  typedef typename Prediction::mtState mtState;
  typedef typename Prediction::mtCovMat mtCovMat;
  typedef typename Prediction::mtMeas mtMeas;
  Prediction prediction_;
  DefaultPrediction defaultPrediction_;
  PredictionManager(){};
  ~PredictionManager(){};
  std::vector<UpdateAndPredictManagerBase<mtState,Prediction>*> mCoupledUpdates_;
  void predict(FilterState<mtState>& filterState, double tNext){
    double tPrediction = tNext;
    int coupledPredictionIndex = -1;
    while(filterState.t_<tNext){
      itMeas_ = measMap_.upper_bound(filterState.t_);
      if(itMeas_ != measMap_.end()){
        tPrediction = std::min(tNext,itMeas_->first);
        coupledPredictionIndex = -1;
        for(unsigned int i=0;i<mCoupledUpdates_.size();i++){
          if(mCoupledUpdates_[i]->hasMeasurementAt(tPrediction)){
            if(coupledPredictionIndex == -1){
              coupledPredictionIndex = i;
            } else {
              std::cout << "Error found multiple updates, only considering first" << std::endl;
            }
          }
        }
        if(coupledPredictionIndex < 0){
          int r = prediction_.predictEKF(filterState.state_,filterState.cov_,itMeas_->second,tPrediction-filterState.t_);
          if(r!=0) std::cout << "Error during prediction: " << r << std::endl;
        } else {
          mCoupledUpdates_[coupledPredictionIndex]->predictAndUpdate(filterState,prediction_,itMeas_->second,tPrediction-filterState.t_);
        }
        filterState.t_ = tPrediction;
      } else {
        mtMeas meas;
        int r = defaultPrediction_.predictEKF(filterState.state_,filterState.cov_,meas,tNext-filterState.t_);
        if(r!=0) std::cout << "Error during prediction: " << r << std::endl;
        filterState.t_ = tNext;
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
    if(!predictionManager_.getLastTime(maxPredictionTime)){
      return false;
    }
    safeTime = maxPredictionTime;
    // Check if we have to wait for update measurements
    for(unsigned int i=0;i<nUT_;i++){
      if(updateManagerBase_[i] != nullptr) updateManagerBase_[i]->constrainTime(maxPredictionTime,safeTime);
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
    while(filterState.t_<tEnd){
      tNext = tEnd;
      for(unsigned int i=0;i<nUT_;i++){
        if(updateManagerBase_[i] != nullptr){
          if(updateManagerBase_[i]->getNextTime(tNext,tNextUpdate) && tNextUpdate < tNext){
            tNext = tNextUpdate;
          }
        }
      }
      predictionManager_.predict(filterState,tNext);
      for(unsigned int i=0;i<nUT_;i++){
        if(updateManagerBase_[i] != nullptr && !updateManagerBase_[i]->coupledToPrediction_){
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
