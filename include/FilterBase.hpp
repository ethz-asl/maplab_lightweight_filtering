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
  virtual void waitTime(double actualTime, double& time) = 0;
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
    maxWaitTime_ = 0.1;
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
  UpdateManager(UpdateFilteringMode filteringMode = UpdateEKF): UpdateManagerBase<typename Update::mtState>(false,filteringMode){
    doubleRegister_.registerDiagonalMatrix("UpdateNoise",update_.updnoiP_);
    doubleRegister_.registerScalar("alpha",update_.alpha_);
    doubleRegister_.registerScalar("beta",update_.beta_);
    doubleRegister_.registerScalar("kappa",update_.kappa_);
  };
  ~UpdateManager(){};
  void update(FilterState<mtState>& filterState){
    if(this->hasMeasurementAt(filterState.t_)){
      int r = update_.updateState(filterState.state_,filterState.cov_,measMap_[filterState.t_]);
      if(r!=0) std::cout << "Error during update: " << r << std::endl;
    }
  }
  void refreshProperties(){
    update_.refreshProperties();
  }
};

template<typename State, typename Prediction>
class UpdateAndPredictManagerBase: public UpdateManagerBase<State>{
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
  using MeasurementTimeline<typename Update::mtMeas>::hasMeasurementAt;
  using UpdateAndPredictManagerBase<typename Update::mtState, Prediction>::filteringMode_;
  using UpdateAndPredictManagerBase<typename Update::mtState, Prediction>::doubleRegister_;
  typedef typename Update::mtState mtState;
  typedef typename Update::mtMeas mtMeas;
  typedef typename mtState::mtCovMat mtCovMat;
  Update update_;
  UpdateAndPredictManager(UpdateFilteringMode filteringMode = UpdateEKF): UpdateAndPredictManagerBase<typename Update::mtState, Prediction>(filteringMode){
    doubleRegister_.registerDiagonalMatrix("UpdateNoise",update_.updnoiP_);
    doubleRegister_.registerMatrix("CorrelatedNoise",update_.preupdnoiP_);
    doubleRegister_.registerScalar("alpha",update_.alpha_);
    doubleRegister_.registerScalar("beta",update_.beta_);
    doubleRegister_.registerScalar("kappa",update_.kappa_);
  };
  ~UpdateAndPredictManager(){};
  void update(FilterState<mtState>& filterState){
    assert(0);
  }
  void predictAndUpdate(FilterState<mtState>& filterState, Prediction& prediction, const typename Prediction::mtMeas& predictionMeas, double dt){
    if(hasMeasurementAt(filterState.t_+dt)){
      int r = update_.predictAndUpdate(filterState.state_,filterState.cov_,measMap_[filterState.t_+dt],prediction,predictionMeas,dt);
      if(r!=0) std::cout << "Error during predictAndUpdate: " << r << std::endl;
    }
  }
  void refreshProperties(){
    update_.refreshProperties();
  }
};

template<typename Prediction, typename DefaultPrediction = Prediction>
class PredictionManager: public MeasurementTimeline<typename Prediction::mtMeas>, public PropertyHandler{
 public:
  using MeasurementTimeline<typename Prediction::mtMeas>::measMap_;
  using MeasurementTimeline<typename Prediction::mtMeas>::itMeas_;
  typedef typename Prediction::mtState mtState;
  typedef typename Prediction::mtCovMat mtCovMat;
  typedef typename Prediction::mtMeas mtMeas;
  Prediction prediction_;
  DefaultPrediction defaultPrediction_;
  const PredictionFilteringMode filteringMode_;
  PredictionManager(PredictionFilteringMode filteringMode = PredictionEKF): filteringMode_(filteringMode){
    doubleRegister_.registerDiagonalMatrix("PredictionNoise",prediction_.prenoiP_);
    doubleRegister_.registerScalar("alpha",prediction_.alpha_);
    doubleRegister_.registerScalar("beta",prediction_.beta_);
    doubleRegister_.registerScalar("kappa",prediction_.kappa_);
  };
  ~PredictionManager(){};
  std::vector<UpdateAndPredictManagerBase<mtState,Prediction>*> mCoupledUpdates_;
  void registerUpdateAndPredictManager(UpdateAndPredictManagerBase<mtState,Prediction>& updateAndPredictManagerBase){
    mCoupledUpdates_.push_back(&updateAndPredictManagerBase);
  }
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
    if(countMergeable>0){
      if(prediction_.mbMergePredictions_){
        r = prediction_.predictMerged(filterState.state_,filterState.cov_,filterState.t_,itMeas_,countMergeable);
        if(r!=0) std::cout << "Error during predictMerged: " << r << std::endl;
        filterState.t_ = next(itMeas_,countMergeable-1)->first;
      } else {
        for(unsigned int i=0;i<countMergeable;i++){
          r = prediction_.predict(filterState.state_,filterState.cov_,itMeas_->second,itMeas_->first-filterState.t_);
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
        r = prediction_.predict(filterState.state_,filterState.cov_,itMeas_->second,tNext-filterState.t_);
        if(r!=0) std::cout << "Error during predict: " << r << std::endl;
      } else {
        mCoupledUpdates_[coupledPredictionIndex]->predictAndUpdate(filterState,prediction_,itMeas_->second,tNext-filterState.t_);
      }
    } else {
      if(coupledPredictionIndex >= 0) std::cout << "No prediction available for coupled update, ignoring update" << std::endl;
      mtMeas meas;
      meas.setIdentity();
      r = defaultPrediction_.predict(filterState.state_,filterState.cov_,meas,tNext-filterState.t_);
      if(r!=0) std::cout << "Error during predict: " << r << std::endl;
    }
    filterState.t_ = tNext;
  }
  void refreshProperties(){
    prediction_.refreshProperties();
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
  PredictionManager<mtPrediction>* mpPredictionManager_;
  std::vector<UpdateManagerBase<mtState>*> mUpdateVector_;
  FilterBase(){
    mpPredictionManager_ = nullptr;
    init_.state_.registerToPropertyHandler(this,"Init.State.");
    doubleRegister_.registerDiagonalMatrix("Init.Covariance.p",init_.cov_);
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
    if(!mpPredictionManager_->getLastTime(maxPredictionTime)){
      return false;
    }
    safeTime = maxPredictionTime;
    // Check if we have to wait for update measurements
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      mUpdateVector_[i]->waitTime(maxPredictionTime,safeTime);
    }
    if(safeTime <= safe_.t_) return false;
    return true;
  }
  void setSafeWarningTime(double safeTime){
    mpPredictionManager_->safeWarningTime_ = safeTime;
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      mUpdateVector_[i]->safeWarningTime_ = safeTime;
    }
  }
  void resetFrontWarningTime(double frontTime){
    mpPredictionManager_->frontWarningTime_ = frontTime;
    mpPredictionManager_->gotFrontWarning_ = false;
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      mUpdateVector_[i]->frontWarningTime_ = frontTime;
      mUpdateVector_[i]->gotFrontWarning_ = false;
    }
  }
  bool checkFrontWarning(){
    bool gotFrontWarning = mpPredictionManager_->gotFrontWarning_;
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
        if(mUpdateVector_[i]->getNextTime(filterState.t_,tNextUpdate) && tNextUpdate < tNext){
          tNext = tNextUpdate;
        }
      }
      mpPredictionManager_->predict(filterState,tNext);
      for(unsigned int i=0;i<mUpdateVector_.size();i++){
        if(mUpdateVector_[i]->hasMeasurementAt(tNext) && !mUpdateVector_[i]->coupledToPrediction_){
          mUpdateVector_[i]->update(filterState);
        }
      }
    }
  }
  void clean(const double& t){
    mpPredictionManager_->clean(t);
    for(unsigned int i=0;i<mUpdateVector_.size();i++){
      mUpdateVector_[i]->clean(t);
    }
  }
  void registerPredictionManager(PredictionManager<Prediction>& predictionManager, std::string str){
    mpPredictionManager_ = &predictionManager;
    registerSubHandler(str,predictionManager);
  }
  void registerUpdateManager(UpdateManagerBase<mtState>& updateManagerBase, std::string str){
    mUpdateVector_.push_back(&updateManagerBase);
    registerSubHandler(str,updateManagerBase);
  }
  void registerUpdateAndPredictManager(UpdateAndPredictManagerBase<mtState,Prediction>& updateAndPredictManagerBase, std::string str){
    mUpdateVector_.push_back(&updateAndPredictManagerBase);
    if(mpPredictionManager_ != nullptr){
      mpPredictionManager_->registerUpdateAndPredictManager(updateAndPredictManagerBase);
    } else {
      std::cout << "Error: Could not register UpdateAndPredict, please first register a prediction" << std::endl;
    }
    registerSubHandler(str,updateAndPredictManagerBase);
  }
  void refreshProperties(){}
};


template<typename Prediction,typename... Updates>
class FilterBase2: public PropertyHandler{
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
  FilterBase2(){
    init_.state_.registerToPropertyHandler(this,"Init.State.");
    doubleRegister_.registerDiagonalMatrix("Init.Covariance.p",init_.cov_);
    safeWarningTime_ = 0.0;
    frontWarningTime_ = 0.0;
    gotFrontWarning_ = false;
  };
  virtual ~FilterBase2(){
  };
  bool getSafeTime(double& safeTime){
    double maxPredictionTime;
    if(!predictionTimeline_.getLastTime(maxPredictionTime)) return false;
    safeTime = maxPredictionTime;
    // Check if we have to wait for update measurements
    checkUpdateWaitTime(maxPredictionTime,safeTime);
    if(safeTime <= safe_.t_) return false;
    return true;
  }
  template<unsigned int i=0>
  void checkUpdateWaitTime(double actualTime,double& time){
    std::get<i>(updateTimelineTuple_).waitTime(actualTime,time);
    if(i+1<nUpdates_) checkUpdateWaitTime<i+1>(actualTime,time);
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
      if(!doCoupledPredictAndUpdateIfAvailable(filterState,tNext)){
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
  template<unsigned int i=0>
  void getNextUpdate(double actualTime, double& nextTime){
    double tNextUpdate;
    if(std::get<i>(updateTimelineTuple_).getNextTime(actualTime,tNextUpdate) && tNextUpdate < nextTime) nextTime = tNextUpdate;
    if(i+1<nUpdates_) getNextUpdate<i+1>(actualTime, nextTime);
  }
  template<unsigned int i=0>
  bool doCoupledPredictAndUpdateIfAvailable(FilterState<mtState>& filterState, double tNext, bool& alreadyDone = false){
    if(std::get<i>(updateTimelineTuple_).coupledToPrediction_ && std::get<i>(updateTimelineTuple_).hasMeasurementAt(tNext)){
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
    if(i+1<nUpdates_) doCoupledPredictAndUpdateIfAvailable<i+1>(filterState,tNext,alreadyDone);
    return alreadyDone;
  }
  template<unsigned int i=0>
  void doAvailableUpdates(FilterState<mtState>& filterState, double tNext){
    if(!std::get<i>(updateTimelineTuple_).coupledToPrediction_ && std::get<i>(updateTimelineTuple_).hasMeasurementAt(tNext)){
          int r = std::get<i>(mUpdates_).updateState(filterState.state_,filterState.cov_,std::get<i>(updateTimelineTuple_).measMap_[tNext]);
          if(r!=0) std::cout << "Error during predictAndUpdate: " << r << std::endl;
    }
    if(i+1<nUpdates_) doAvailableUpdates<i+1>(filterState,tNext);
  }
  void clean(const double& t){
    predictionTimeline_.clean(t);
    cleanUpdateTimeline(t);
  }
  template<unsigned int i=0>
  void cleanUpdateTimeline(const double& t){
    std::get<i>(updateTimelineTuple_).clean(t);
    if(i+1<nUpdates_) cleanUpdateTimeline<i+1>(t);
  }
};

}

#endif /* FilterBase_HPP_ */
