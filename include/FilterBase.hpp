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
#include "PropertyHandler.hpp"
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

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
  MeasurementTimelineBase(){
    safeWarningTime_ = 0.0;
    frontWarningTime_ = 0.0;
    gotFrontWarning_ = false;
  };
  virtual ~MeasurementTimelineBase(){};
  virtual bool getNextTime(double actualTime, double& nextTime) = 0;
  virtual void clean(double t) = 0;
  virtual void constrainTime(double actualTime, double& time) = 0;
  virtual bool getLastTime(double& lastTime) = 0;
  virtual bool hasMeasurementAt(double t) = 0;
  double safeWarningTime_;
  double frontWarningTime_;
  bool gotFrontWarning_;
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
    if(t<= safeWarningTime_) std::cout << "Warning: included measurements before safeTime" << std::endl;
    if(t<= frontWarningTime_) gotFrontWarning_ = true;
    measMap_[t] = meas;
  }
  void clean(double t){
    while(!measMap_.empty() && measMap_.begin()->first<=t){
      measMap_.erase(measMap_.begin());
    }
  }
  bool getNextTime(double actualTime, double& nextTime){
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
class UpdateManagerBase: public virtual MeasurementTimelineBase, public PropertyHandler{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  UpdateManagerBase(bool coupledToPrediction = false, UpdateFilteringMode filteringMode = UpdateEKF): coupledToPrediction_(coupledToPrediction), filteringMode_(filteringMode){};
  virtual ~UpdateManagerBase(){};
  virtual void update(FilterState<mtState>& filterState) = 0;
  const bool coupledToPrediction_;
  const UpdateFilteringMode filteringMode_;
};

template<typename Update>
class UpdateManager: public MeasurementTimeline<typename Update::mtMeas>,public UpdateManagerBase<typename Update::mtState>{
 public:
  using MeasurementTimeline<typename Update::mtMeas>::measMap_;
  using UpdateManagerBase<typename Update::mtState>::filteringMode_;
  using UpdateManagerBase<typename Update::mtState>::doubleRegister_;
  typedef typename Update::mtState mtState;
  typedef typename Update::mtMeas mtMeas;
  typedef typename mtState::mtCovMat mtCovMat;
  Update update_;
  double test;
  UpdateManager(UpdateFilteringMode filteringMode = UpdateEKF): UpdateManagerBase<typename Update::mtState>(false,filteringMode){
    test = 0.5;
    doubleRegister_.registerScalar("updnoiP00",test);
  };
  ~UpdateManager(){};
  void update(FilterState<mtState>& filterState){
    if(this->hasMeasurementAt(filterState.t_)){
      int r = update_.updateState(filterState.state_,filterState.cov_,measMap_[filterState.t_],filteringMode_);
      if(r!=0) std::cout << "Error during update: " << r << std::endl;
    }
  }
};

template<typename State, typename Prediction>
class UpdateAndPredictManagerBase: public virtual UpdateManagerBase<State>{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  UpdateAndPredictManagerBase(UpdateFilteringMode filteringMode = UpdateEKF): UpdateManagerBase<State>(true,filteringMode){};
  virtual ~UpdateAndPredictManagerBase(){};
  virtual void predictAndUpdate(FilterState<mtState>& filterState, Prediction& prediction, const typename Prediction::mtMeas& predictionMeas, double dt) = 0;
};

template<typename Update, typename Prediction>
class UpdateAndPredictManager:public MeasurementTimeline<typename Update::mtMeas>, public UpdateAndPredictManagerBase<typename Update::mtState, Prediction>{
 public:
  using MeasurementTimeline<typename Update::mtMeas>::measMap_;
  using UpdateAndPredictManagerBase<typename Update::mtState, Prediction>::filteringMode_;
  typedef typename Update::mtState mtState;
  typedef typename Update::mtMeas mtMeas;
  typedef typename mtState::mtCovMat mtCovMat;
  Update update_;
  UpdateAndPredictManager(UpdateFilteringMode filteringMode = UpdateEKF): UpdateAndPredictManagerBase<typename Update::mtState, Prediction>(filteringMode){};
  ~UpdateAndPredictManager(){};
  void update(FilterState<mtState>& filterState){
    assert(0);
  }
  void predictAndUpdate(FilterState<mtState>& filterState, Prediction& prediction, const typename Prediction::mtMeas& predictionMeas, double dt){
    if(hasMeasurementAt(filterState.t_)){
      int r = update_.predictAndUpdate(filterState.state_,filterState.cov_,measMap_[filterState.t_],prediction,predictionMeas,dt,filteringMode_());
      if(r!=0) std::cout << "Error during predictAndUpdate: " << r << std::endl;
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
  const PredictionFilteringMode filteringMode_;
  PredictionManager(PredictionFilteringMode filteringMode = PredictionEKF): filteringMode_(filteringMode){};
  ~PredictionManager(){};
  std::vector<UpdateAndPredictManagerBase<mtState,Prediction>*> mCoupledUpdates_;
  void predict(FilterState<mtState>& filterState, const double tNext){
    double tPrediction = tNext;
    int r = 0;

    // Count mergeable prediction steps (always without update)
    itMeas_ = measMap_.upper_bound(filterState.t_);
    unsigned int countMergeable = 0;
    while(itMeas_ != measMap_.end() && itMeas_->first < tNext){
      countMergeable++;
      itMeas_++;
    }
    itMeas_ = measMap_.upper_bound(filterState.t_); // Reset Iterator
    if(countMergeable>1){
      if(prediction_.mbMergePredictions_){
        r = prediction_.predictMerged(filterState.state_,filterState.cov_,filterState.t_,itMeas_,countMergeable,filteringMode_);
        if(r!=0) std::cout << "Error during predictMerged: " << r << std::endl;
      } else {
        for(unsigned int i=0;i<countMergeable;i++){
          r = prediction_.predict(filterState.state_,filterState.cov_,itMeas_->second,itMeas_->first-filterState.t_,filteringMode_);
          if(r!=0) std::cout << "Error during predict: " << r << std::endl;
          filterState.t_ = itMeas_->first;
          itMeas_++;
        }
      }
    }

    // Check for coupled update
    itMeas_ = measMap_.upper_bound(filterState.t_); // Reset Iterator
    int coupledPredictionIndex = -1;
    for(unsigned int i=0;i<mCoupledUpdates_.size();i++){
      if(mCoupledUpdates_[i]->hasMeasurementAt(tNext)){
        if(coupledPredictionIndex == -1){
          coupledPredictionIndex = i;
        } else {
          std::cout << "Warning found multiple updates, only considering first" << std::endl;
        }
      }
    }
    if(itMeas_ != measMap_.end()){
      if(coupledPredictionIndex < 0){
        r = prediction_.predict(filterState.state_,filterState.cov_,itMeas_->second,tNext-filterState.t_,filteringMode_);
        if(r!=0) std::cout << "Error during predict: " << r << std::endl;
      } else {
        mCoupledUpdates_[coupledPredictionIndex]->predictAndUpdate(filterState,prediction_,itMeas_->second,tNext-filterState.t_);
      }
    } else {
      if(coupledPredictionIndex >= 0) std::cout << "No prediction available for coupled update, ignoring update" << std::endl;
      mtMeas meas;
      r = defaultPrediction_.predict(filterState.state_,filterState.cov_,meas,tNext-filterState.t_,filteringMode_);
      if(r!=0) std::cout << "Error during predict: " << r << std::endl;
    }
    filterState.t_ = tNext;
  }
};

template<typename Prediction>
class FilterBase: public PropertyHandler{
 public:
  typedef Prediction mtPrediction;
  typedef typename mtPrediction::mtState mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  FilterState<mtState> safe_;
  FilterState<mtState> front_;
  FilterState<mtState> init_;
  PredictionManager<mtPrediction> predictionManager_;
  std::vector<UpdateManagerBase<mtState>*> mUpdateVector_;
  FilterBase(){
    doubleRegister_.registerScalar("t",safe_.t_);
  };
  virtual ~FilterBase(){
  };
  void resetFilter(){
    safe_ = init_;
    front_ = init_;
    setSafeWarningTime(safe_.t_);
    resetFrontWarningTime(front_.t_);
  }
  bool getSafeTime(double& safeTime){
    double maxPredictionTime;
    if(!predictionManager_.getLastTime(maxPredictionTime)){
      return false;
    }
    safeTime = maxPredictionTime;
    // Check if we have to wait for update measurements
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      mUpdateVector_[i]->constrainTime(maxPredictionTime,safeTime);
    }
    if(safeTime <= safe_.t_) return false;
    return true;
  }
  void setSafeWarningTime(double safeTime){
    predictionManager_.safeWarningTime_ = safeTime;
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      mUpdateVector_[i]->safeWarningTime_ = safeTime;
    }
  }
  void resetFrontWarningTime(double frontTime){
    predictionManager_.frontWarningTime_ = frontTime;
    predictionManager_.gotFrontWarning_ = false;
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      mUpdateVector_[i]->frontWarningTime_ = frontTime;
      mUpdateVector_[i]->gotFrontWarning_ = false;
    }
  }
  bool checkFrontWarning(){
    bool gotFrontWarning = predictionManager_.gotFrontWarning_;
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      gotFrontWarning = mUpdateVector_[i]->gotFrontWarning_ | gotFrontWarning;
    }
    return gotFrontWarning;
  }
  void updateSafe(){
    double nextSafeTime;
    if(!getSafeTime(nextSafeTime)) return;
    if(front_.t_<=nextSafeTime && !checkFrontWarning() && front_.t_>safe_.t_){
      safe_ = front_;
    }
    update(safe_,nextSafeTime);
    clean(nextSafeTime);
    setSafeWarningTime(nextSafeTime);
  }
  void updateFront(const double& tEnd){
    updateSafe();
    if(checkFrontWarning() || front_.t_<=safe_.t_){
      front_ = safe_;
    }
    update(front_,tEnd);
    resetFrontWarningTime(tEnd);
  }
  void update(FilterState<mtState>& filterState,const double& tEnd){
    double tNext = filterState.t_;
    double tNextUpdate;
    while(filterState.t_<tEnd){
      tNext = tEnd;
      for(unsigned int i=0;i<mUpdateVector_.size();i++){
        if(mUpdateVector_[i]->getNextTime(tNext,tNextUpdate) && tNextUpdate < tNext){
          tNext = tNextUpdate;
        }
      }
      predictionManager_.predict(filterState,tNext);
      for(unsigned int i=0;i<mUpdateVector_.size();i++){
        if(!mUpdateVector_[i]->coupledToPrediction_){
          mUpdateVector_[i]->update(filterState);
        }
      }
    }
  }
  void clean(const double& t){
    predictionManager_.clean(t);
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      if(mUpdateVector_[i] != nullptr) mUpdateVector_[i]->clean(t);
    }
  }
  void registerUpdateManager(UpdateManagerBase<mtState>& updateManagerBase){
    mUpdateVector_.push_back(&updateManagerBase); // TODO make unique
    registerSubHandler("test",updateManagerBase); // TODO change name
  }
};

}

#endif /* FilterBase_HPP_ */
