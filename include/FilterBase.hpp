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
#include <tuple>

namespace LWF{

template<typename State>
class FilterState{ // TODO: include outlier detection
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

template<typename Meas>
class MeasurementTimeline{
 public:
  typedef Meas mtMeas;
  MeasurementTimeline(){
    maxWaitTime_ = 0.1;
  };
  virtual ~MeasurementTimeline(){};
  std::map<double,mtMeas> measMap_;
  typename std::map<double,mtMeas>::iterator itMeas_;
  double maxWaitTime_;
  void addMeas(const mtMeas& meas, const double& t){
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
  void waitTime(double actualTime, double& time){
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

template<typename Prediction,typename... Updates>
class FilterBase: public PropertyHandler{
 public:
  typedef typename Prediction::mtState mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  static const unsigned int D_ = mtState::D_;
  static const unsigned int nUpdates_ = sizeof...(Updates);
  FilterState<mtState> safe_;
  FilterState<mtState> front_;
  FilterState<mtState> init_;
  MeasurementTimeline<typename Prediction::mtMeas> predictionTimeline_;
  std::tuple<MeasurementTimeline<typename Updates::mtMeas>...> updateTimelineTuple_;
  Prediction mPrediction_;
  std::tuple<Updates...> mUpdates_;
  double safeWarningTime_;
  double frontWarningTime_;
  bool gotFrontWarning_;
  FilterBase(){
    init_.state_.registerToPropertyHandler(this,"Init.State.");
    doubleRegister_.registerDiagonalMatrix("Init.Covariance.p",init_.cov_);
    registerSubHandler("Prediction",mPrediction_);
    registerUpdates();
    safeWarningTime_ = 0.0;
    frontWarningTime_ = 0.0;
    gotFrontWarning_ = false;
  };
  virtual ~FilterBase(){
  };
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_-1)>::type* = nullptr>
  void registerUpdates(){
    registerSubHandler("Update" + std::to_string(i),std::get<i>(mUpdates_));
    registerUpdates<i+1>();
  }
  template<unsigned int i=0, typename std::enable_if<(i==nUpdates_-1)>::type* = nullptr>
  void registerUpdates(){
    registerSubHandler("Update" + std::to_string(i),std::get<i>(mUpdates_));
  }
  void addPredictionMeas(const typename Prediction::mtMeas& meas, double t){
    if(t<= safeWarningTime_) std::cout << "Warning: included measurements before safeTime" << std::endl;
    if(t<= frontWarningTime_) gotFrontWarning_ = true;
    predictionTimeline_.addMeas(meas,t);
  }
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_)>::type* = nullptr>
  void addUpdateMeas(const typename std::tuple_element<i,decltype(mUpdates_)>::type::mtMeas& meas, double t){
    if(t<= safeWarningTime_) std::cout << "Warning: included measurements before safeTime" << std::endl;
    if(t<= frontWarningTime_) gotFrontWarning_ = true;
    std::get<i>(updateTimelineTuple_).addMeas(meas,t);
  }
  bool getSafeTime(double& safeTime){
    double maxPredictionTime;
    if(!predictionTimeline_.getLastTime(maxPredictionTime)) return false;
    safeTime = maxPredictionTime;
    // Check if we have to wait for update measurements
    checkUpdateWaitTime(maxPredictionTime,safeTime);
    if(safeTime <= safe_.t_) return false;
    return true;
  }
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_-1)>::type* = nullptr>
  void checkUpdateWaitTime(double actualTime,double& time){
    std::get<i>(updateTimelineTuple_).waitTime(actualTime,time);
    checkUpdateWaitTime<i+1>(actualTime,time);
  }
  template<unsigned int i=0, typename std::enable_if<(i==nUpdates_-1)>::type* = nullptr>
  void checkUpdateWaitTime(double actualTime,double& time){
    std::get<i>(updateTimelineTuple_).waitTime(actualTime,time);
  }
  void updateSafe(){
    double nextSafeTime;
    if(!getSafeTime(nextSafeTime)) return;
    if(front_.t_<=nextSafeTime && !gotFrontWarning_ && front_.t_>safe_.t_){
      safe_ = front_;
    }
    update(safe_,nextSafeTime);
    clean(nextSafeTime);
    safeWarningTime_ = nextSafeTime;
  }
  void updateFront(const double& tEnd){
    updateSafe();
    if(gotFrontWarning_ || front_.t_<=safe_.t_){
      front_ = safe_;
    }
    update(front_,tEnd);
    frontWarningTime_ = tEnd;
    gotFrontWarning_ = false;
  }
  void update(FilterState<mtState>& filterState,const double& tEnd){
    double tNext = filterState.t_;
    while(filterState.t_<tEnd){
      tNext = tEnd;
      getNextUpdate(filterState.t_,tNext);
      double tPrediction = tNext;
      int r = 0;

      // Count mergeable prediction steps (always without update)
      predictionTimeline_.itMeas_ = predictionTimeline_.measMap_.upper_bound(filterState.t_);
      unsigned int countMergeable = 0;
      while(predictionTimeline_.itMeas_ != predictionTimeline_.measMap_.end() && predictionTimeline_.itMeas_->first < tNext){
        countMergeable++;
        predictionTimeline_.itMeas_++;
      }
      predictionTimeline_.itMeas_ = predictionTimeline_.measMap_.upper_bound(filterState.t_); // Reset Iterator
      if(countMergeable>0){
        if(mPrediction_.mbMergePredictions_){
          r = mPrediction_.predictMerged(filterState.state_,filterState.cov_,filterState.t_,predictionTimeline_.itMeas_,countMergeable);
          if(r!=0) std::cout << "Error during predictMerged: " << r << std::endl;
          filterState.t_ = next(predictionTimeline_.itMeas_,countMergeable-1)->first;
        } else {
          for(unsigned int i=0;i<countMergeable;i++){
            r = mPrediction_.predict(filterState.state_,filterState.cov_,predictionTimeline_.itMeas_->second,predictionTimeline_.itMeas_->first-filterState.t_);
            if(r!=0) std::cout << "Error during predict: " << r << std::endl;
            filterState.t_ = predictionTimeline_.itMeas_->first;
            predictionTimeline_.itMeas_++;
          }
        }
      }

      // Check for coupled update
      bool doneCoupledUpdate = false;
      doCoupledPredictAndUpdateIfAvailable(filterState,tNext,doneCoupledUpdate);
      if(!doneCoupledUpdate){
        predictionTimeline_.itMeas_ = predictionTimeline_.measMap_.upper_bound(filterState.t_); // Reset Iterator
        if(predictionTimeline_.itMeas_ != predictionTimeline_.measMap_.end()){
          r = mPrediction_.predict(filterState.state_,filterState.cov_,predictionTimeline_.itMeas_->second,tNext-filterState.t_);
          if(r!=0) std::cout << "Error during predict: " << r << std::endl;
        } else {
          r = mPrediction_.predict(filterState.state_,filterState.cov_,tNext-filterState.t_);
          if(r!=0) std::cout << "Error during predict: " << r << std::endl;
        }
      }
      filterState.t_ = tNext;

      doAvailableUpdates(filterState,tNext);
    }
  }
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_-1)>::type* = nullptr>
  void getNextUpdate(double actualTime, double& nextTime){
    double tNextUpdate;
    if(std::get<i>(updateTimelineTuple_).getNextTime(actualTime,tNextUpdate) && tNextUpdate < nextTime) nextTime = tNextUpdate;
    getNextUpdate<i+1>(actualTime, nextTime);
  }
  template<unsigned int i=0, typename std::enable_if<(i==nUpdates_-1)>::type* = nullptr>
  void getNextUpdate(double actualTime, double& nextTime){
    double tNextUpdate;
    if(std::get<i>(updateTimelineTuple_).getNextTime(actualTime,tNextUpdate) && tNextUpdate < nextTime) nextTime = tNextUpdate;
  }
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_-1 & std::tuple_element<i,decltype(mUpdates_)>::type::coupledToPrediction_)>::type* = nullptr>
  void doCoupledPredictAndUpdateIfAvailable(FilterState<mtState>& filterState, double tNext, bool& alreadyDone){
    if(std::get<i>(updateTimelineTuple_).hasMeasurementAt(tNext)){
      if(!alreadyDone){
        predictionTimeline_.itMeas_ = predictionTimeline_.measMap_.upper_bound(filterState.t_);
        if(predictionTimeline_.itMeas_ != predictionTimeline_.measMap_.end()){
          int r = std::get<i>(mUpdates_).predictAndUpdate(filterState.state_,filterState.cov_,std::get<i>(updateTimelineTuple_).measMap_[tNext],mPrediction_,predictionTimeline_.itMeas_->second,tNext-filterState.t_);
          if(r!=0) std::cout << "Error during predictAndUpdate: " << r << std::endl;
          alreadyDone = true;
        } else {
          std::cout << "No prediction available for coupled update, ignoring update" << std::endl;
        }
      } else {
        std::cout << "Warning found multiple updates, only considering first" << std::endl;
      }
    }
    doCoupledPredictAndUpdateIfAvailable<i+1>(filterState,tNext,alreadyDone);
  }
  template<unsigned int i=0, typename std::enable_if<(i==nUpdates_-1 & std::tuple_element<i,decltype(mUpdates_)>::type::coupledToPrediction_)>::type* = nullptr>
  void doCoupledPredictAndUpdateIfAvailable(FilterState<mtState>& filterState, double tNext, bool& alreadyDone){
    if(std::get<i>(updateTimelineTuple_).hasMeasurementAt(tNext)){
      if(!alreadyDone){
        predictionTimeline_.itMeas_ = predictionTimeline_.measMap_.upper_bound(filterState.t_);
        if(predictionTimeline_.itMeas_ != predictionTimeline_.measMap_.end()){
          int r = std::get<i>(mUpdates_).predictAndUpdate(filterState.state_,filterState.cov_,std::get<i>(updateTimelineTuple_).measMap_[tNext],mPrediction_,predictionTimeline_.itMeas_->second,tNext-filterState.t_);
          if(r!=0) std::cout << "Error during predictAndUpdate: " << r << std::endl;
          alreadyDone = true;
        } else {
          std::cout << "No prediction available for coupled update, ignoring update" << std::endl;
        }
      } else {
        std::cout << "Warning found multiple updates, only considering first" << std::endl;
      }
    }
  }
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_-1 & !std::tuple_element<i,decltype(mUpdates_)>::type::coupledToPrediction_)>::type* = nullptr>
  void doCoupledPredictAndUpdateIfAvailable(FilterState<mtState>& filterState, double tNext, bool& alreadyDone){
    doCoupledPredictAndUpdateIfAvailable<i+1>(filterState,tNext,alreadyDone);
  }
  template<unsigned int i=0, typename std::enable_if<(i==nUpdates_-1 & !std::tuple_element<i,decltype(mUpdates_)>::type::coupledToPrediction_)>::type* = nullptr>
  void doCoupledPredictAndUpdateIfAvailable(FilterState<mtState>& filterState, double tNext, bool& alreadyDone){
  }
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_-1)>::type* = nullptr>
  void doAvailableUpdates(FilterState<mtState>& filterState, double tNext){
    if(!std::get<i>(mUpdates_).coupledToPrediction_ && std::get<i>(updateTimelineTuple_).hasMeasurementAt(tNext)){
          int r = std::get<i>(mUpdates_).updateState(filterState.state_,filterState.cov_,std::get<i>(updateTimelineTuple_).measMap_[tNext]);
          if(r!=0) std::cout << "Error during predictAndUpdate: " << r << std::endl;
    }
    doAvailableUpdates<i+1>(filterState,tNext);
  }
  template<unsigned int i=0, typename std::enable_if<(i==nUpdates_-1)>::type* = nullptr>
  void doAvailableUpdates(FilterState<mtState>& filterState, double tNext){
    if(!std::get<i>(mUpdates_).coupledToPrediction_ && std::get<i>(updateTimelineTuple_).hasMeasurementAt(tNext)){
          int r = std::get<i>(mUpdates_).updateState(filterState.state_,filterState.cov_,std::get<i>(updateTimelineTuple_).measMap_[tNext]);
          if(r!=0) std::cout << "Error during predictAndUpdate: " << r << std::endl;
    }
  }
  void clean(const double& t){
    predictionTimeline_.clean(t);
    cleanUpdateTimeline(t);
  }
  template<unsigned int i=0, typename std::enable_if<(i<nUpdates_-1)>::type* = nullptr>
  void cleanUpdateTimeline(const double& t){
    std::get<i>(updateTimelineTuple_).clean(t);
    cleanUpdateTimeline<i+1>(t);
  }
  template<unsigned int i=0, typename std::enable_if<(i==nUpdates_-1)>::type* = nullptr>
  void cleanUpdateTimeline(const double& t){
    std::get<i>(updateTimelineTuple_).clean(t);
  }
};

}

#endif /* FilterBase_HPP_ */
