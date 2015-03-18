/*
 * Prediction.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef PREDICTIONMODEL_HPP_
#define PREDICTIONMODEL_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
#include "ModelBase.hpp"
#include "SigmaPoints.hpp"
#include "State.hpp"
#include "FilterState.hpp"
#include "PropertyHandler.hpp"
#include <map>

namespace LWF{

template<typename FilterState>
class Prediction: public ModelBase<typename FilterState::mtState,typename FilterState::mtState,typename FilterState::mtPredictionMeas,typename FilterState::mtPredictionNoise,FilterState::useDynamicMatrix_>, public PropertyHandler{
 public:
  typedef FilterState mtFilterState;
  typedef typename mtFilterState::mtState mtState;
  typedef typename mtFilterState::mtFilterCovMat mtFilterCovMat;
  typedef typename mtFilterState::mtPredictionMeas mtMeas;
  typedef typename mtFilterState::mtPredictionNoise mtNoise;
  typedef ModelBase<mtState,mtState,mtMeas,mtNoise> mtModelBase;
  LWFMatrix<mtNoise::D_,mtNoise::D_,mtFilterState::useDynamicMatrix_> prenoiP_;
  Prediction(){
    prenoiP_.setIdentity();
    prenoiP_ *= 0.0001;
    mtNoise n;
    n.registerCovarianceToPropertyHandler_(prenoiP_,this,"PredictionNoise.");
  };
  virtual void noMeasCase(mtFilterState& filterState, mtMeas& meas, double dt){};
  virtual void preProcess(mtFilterState& filterState, const mtMeas& meas, double dt){};
  virtual void postProcess(mtFilterState& filterState, const mtMeas& meas, double dt){};
  virtual ~Prediction(){};
  int performPrediction(mtFilterState& filterState, const mtMeas& meas, double dt){
    switch(filterState.mode_){
      case ModeEKF:
        return performPredictionEKF(filterState,meas,dt);
      case ModeUKF:
        return performPredictionUKF(filterState,meas,dt);
      case ModeIEKF:
        return performPredictionEKF(filterState,meas,dt);
      default:
        return performPredictionEKF(filterState,meas,dt);
    }
  }
  int performPrediction(mtFilterState& filterState, double dt){
    mtMeas meas;
    meas.setIdentity();
    noMeasCase(filterState,meas,dt);
    return performPrediction(filterState,meas,dt);
  }
  int performPredictionEKF(mtFilterState& filterState, const mtMeas& meas, double dt){
    preProcess(filterState,meas,dt);
    this->jacInput(filterState.F_,filterState.state_,meas,dt);
    this->jacNoise(filterState.G_,filterState.state_,meas,dt);
    this->eval(filterState.state_,filterState.state_,meas,dt);
    filterState.cov_ = filterState.F_*filterState.cov_*filterState.F_.transpose() + filterState.G_*prenoiP_*filterState.G_.transpose();
    filterState.state_.fix();
    filterState.cov_ = 0.5*(filterState.cov_+filterState.cov_.transpose()); // Enforce symmetry
    filterState.t_ += dt;
    postProcess(filterState,meas,dt);
    return 0;
  }
  int performPredictionUKF(mtFilterState& filterState, const mtMeas& meas, double dt){
    filterState.refreshNoiseSigmaPoints(prenoiP_);
    preProcess(filterState,meas,dt);
    filterState.stateSigmaPoints_.computeFromGaussian(filterState.state_,filterState.cov_);

    // Prediction
    for(unsigned int i=0;i<filterState.stateSigmaPoints_.L_;i++){
      this->eval(filterState.stateSigmaPointsPre_(i),filterState.stateSigmaPoints_(i),meas,filterState.stateSigmaPointsNoi_(i),dt);
    }
    // Calculate mean and variance
    filterState.state_ = filterState.stateSigmaPointsPre_.getMean();
    filterState.cov_ = filterState.stateSigmaPointsPre_.getCovarianceMatrix(filterState.state_);
    filterState.state_.fix();
    filterState.t_ += dt;
    postProcess(filterState,meas,dt);
    return 0;
  }
  int predictMerged(mtFilterState& filterState, double tTarget, const std::map<double,mtMeas>& measMap){
    switch(filterState.mode_){
      case ModeEKF:
        return predictMergedEKF(filterState,tTarget,measMap);
      case ModeUKF:
        return predictMergedUKF(filterState,tTarget,measMap);
      case ModeIEKF:
        return predictMergedEKF(filterState,tTarget,measMap);
      default:
        return predictMergedEKF(filterState,tTarget,measMap);
    }
  }
  virtual int predictMergedEKF(mtFilterState& filterState, double tTarget, const std::map<double,mtMeas>& measMap){
    const typename std::map<double,mtMeas>::const_iterator itMeasStart = measMap.upper_bound(filterState.t_);
    if(itMeasStart == measMap.end()) return 0;
    const typename std::map<double,mtMeas>::const_iterator itMeasEnd = measMap.upper_bound(tTarget);
    if(itMeasEnd == measMap.begin()) return 0;
    double dT = std::prev(itMeasEnd)->first-filterState.t_;

    // Compute mean Measurement
    mtMeas meanMeas;
    typename mtMeas::mtDifVec vec;
    typename mtMeas::mtDifVec difVec;
    vec.setZero();
    double t = itMeasStart->first;
    for(typename std::map<double,mtMeas>::const_iterator itMeas=next(itMeasStart);itMeas!=itMeasEnd;itMeas++){
      itMeasStart->second.boxMinus(itMeas->second,difVec);
      vec = vec + difVec*(itMeas->first-t);
      t = itMeas->first;
    }
    vec = vec/dT;
    itMeasStart->second.boxPlus(vec,meanMeas);

    preProcess(filterState,meanMeas,dT);
    this->jacInput(filterState.F_,filterState.state_,meanMeas,dT);
    this->jacNoise(filterState.G_,filterState.state_,meanMeas,dT); // Works for time continuous parametrization of noise
    for(typename std::map<double,mtMeas>::const_iterator itMeas=itMeasStart;itMeas!=itMeasEnd;itMeas++){
      this->eval(filterState.state_,filterState.state_,itMeas->second,itMeas->first-filterState.t_);
      filterState.t_ = itMeas->first;
    }
    filterState.cov_ = filterState.F_*filterState.cov_*filterState.F_.transpose() + filterState.G_*prenoiP_*filterState.G_.transpose();
    filterState.state_.fix();
    filterState.cov_ = 0.5*(filterState.cov_+filterState.cov_.transpose()); // Enforce symmetry
    filterState.t_ = std::prev(itMeasEnd)->first;
    postProcess(filterState,meanMeas,dT);
    return 0;
  }
  virtual int predictMergedUKF(mtFilterState& filterState, double tTarget, const std::map<double,mtMeas>& measMap){
    filterState.refreshNoiseSigmaPoints(prenoiP_);
    const typename std::map<double,mtMeas>::const_iterator itMeasStart = measMap.upper_bound(filterState.t_);
    if(itMeasStart == measMap.end()) return 0;
    const typename std::map<double,mtMeas>::const_iterator itMeasEnd = measMap.upper_bound(tTarget);
    if(itMeasEnd == measMap.begin()) return 0;
    double dT = std::prev(itMeasEnd)->first-filterState.t_;

    // Compute mean Measurement
    mtMeas meanMeas;
    typename mtMeas::mtDifVec vec;
    typename mtMeas::mtDifVec difVec;
    vec.setZero();
    double t = itMeasStart->first;
    for(typename std::map<double,mtMeas>::const_iterator itMeas=next(itMeasStart);itMeas!=itMeasEnd;itMeas++){
      itMeasStart->second.boxMinus(itMeas->second,difVec);
      vec = vec + difVec*(itMeas->first-t);
      t = itMeas->first;
    }
    vec = vec/dT;
    itMeasStart->second.boxPlus(vec,meanMeas);

    preProcess(filterState,meanMeas,dT);
    filterState.stateSigmaPoints_.computeFromGaussian(filterState.state_,filterState.cov_);

    // Prediction
    for(unsigned int i=0;i<filterState.stateSigmaPoints_.L_;i++){
      this->eval(filterState.stateSigmaPointsPre_(i),filterState.stateSigmaPoints_(i),meanMeas,filterState.stateSigmaPointsNoi_(i),dT);
    }
    filterState.state_ = filterState.stateSigmaPointsPre_.getMean();
    filterState.cov_ = filterState.stateSigmaPointsPre_.getCovarianceMatrix(filterState.state_);
    filterState.state_.fix();
    filterState.t_ = std::prev(itMeasEnd)->first;
    postProcess(filterState,meanMeas,dT);
    return 0;
  }
};

}

#endif /* PREDICTIONMODEL_HPP_ */
