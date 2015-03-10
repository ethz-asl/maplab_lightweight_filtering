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
#include "PropertyHandler.hpp"
#include <map>

namespace LWF{

enum PredictionFilteringMode{
  PredictionEKF,
  PredictionUKF
};

template<typename FilterState, typename Meas, typename Noise>
class Prediction: public ModelBase<typename FilterState::mtState,typename FilterState::mtState,Meas,Noise>, public PropertyHandler{
 public:
  typedef FilterState mtFilterState;
  typedef typename mtFilterState::mtState mtState;
  typedef typename mtFilterState::mtFilterCovMat mtFilterCovMat;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef ModelBase<mtState,mtState,mtMeas,mtNoise> mtModelBase;
  typename mtModelBase::mtJacInput F_;
  typename mtModelBase::mtJacNoise Fn_;
  typename mtNoise::mtCovMat prenoiP_;
  typename mtNoise::mtCovMat noiP_; // automatic change tracking
  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+mtNoise::D_)+1,0> stateSigmaPoints_;
  SigmaPoints<mtNoise,2*mtNoise::D_+1,2*(mtState::D_+mtNoise::D_)+1,2*mtState::D_> stateSigmaPointsNoi_;
  SigmaPoints<mtState,2*(mtState::D_+mtNoise::D_)+1,2*(mtState::D_+mtNoise::D_)+1,0> stateSigmaPointsPre_;
  bool mbMergePredictions_;
  PredictionFilteringMode mode_;
  double alpha_;
  double beta_;
  double kappa_;
  Prediction(bool mbMergePredictions = false): mbMergePredictions_(mbMergePredictions){
    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    mode_ = PredictionEKF;
    prenoiP_ = mtNoise::mtCovMat::Identity()*0.0001;
    noiP_.setZero();
    resetPrediction();
    mtNoise n;
    n.registerCovarianceToPropertyHandler_(prenoiP_,this,"PredictionNoise.");
    doubleRegister_.registerScalar("alpha",alpha_);
    doubleRegister_.registerScalar("beta",beta_);
    doubleRegister_.registerScalar("kappa",kappa_);
  };
  void refreshNoiseSigmaPoints(){
    if(noiP_ != prenoiP_){
      noiP_ = prenoiP_;
      stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    }
  }
  void refreshUKFParameter(){
    stateSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsPre_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
  }
  void refreshProperties(){
    refreshPropertiesCustom();
    refreshUKFParameter();
  }
  virtual void refreshPropertiesCustom(){}
  virtual void noMeasCase(mtFilterState& filterState, mtMeas& meas, double dt){};
  virtual void preProcess(mtFilterState& filterState, const mtMeas& meas, double dt){};
  virtual void postProcess(mtFilterState& filterState, const mtMeas& meas, double dt){};
  void resetPrediction(){
    refreshNoiseSigmaPoints();
    refreshUKFParameter();
  }
  virtual ~Prediction(){};
  void setMode(PredictionFilteringMode mode){
    mode_ = mode;
  }
  int performPrediction(mtFilterState& filterState, const mtMeas& meas, double dt){
    switch(mode_){
      case PredictionEKF:
        return performPredictionEKF(filterState,meas,dt);
      case PredictionUKF:
        return performPredictionUKF(filterState,meas,dt);
      default:
        return performPredictionEKF(filterState,meas,dt);
    }
  }
  int performPrediction(mtFilterState& filterState, double dt){
    mtMeas meas;
    meas.setIdentity();
    noMeasCase(filterState,meas,dt);
    switch(mode_){
      case PredictionEKF:
        return performPredictionEKF(filterState,meas,dt);
      case PredictionUKF:
        return performPredictionUKF(filterState,meas,dt);
      default:
        return performPredictionEKF(filterState,meas,dt);
    }
  }
  int performPredictionEKF(mtFilterState& filterState, const mtMeas& meas, double dt){
    preProcess(filterState,meas,dt);
    this->jacInput(F_,filterState.state_,meas,dt);
    this->jacNoise(Fn_,filterState.state_,meas,dt);
    this->eval(filterState.state_,filterState.state_,meas,dt);
    filterState.cov_ = F_*filterState.cov_*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    postProcess(filterState,meas,dt);
    filterState.state_.fix();
    filterState.cov_ = 0.5*(filterState.cov_+filterState.cov_.transpose()); // Enforce symmetry
    return 0;
  }
  int performPredictionUKF(mtFilterState& filterState, const mtMeas& meas, double dt){
    refreshNoiseSigmaPoints();
    preProcess(filterState,meas,dt);
    stateSigmaPoints_.computeFromGaussian(filterState.state_,filterState.cov_);

    // Prediction
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      this->eval(stateSigmaPointsPre_(i),stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i),dt);
    }
    // Calculate mean and variance
    filterState.state_ = stateSigmaPointsPre_.getMean();
    filterState.cov_ = stateSigmaPointsPre_.getCovarianceMatrix(filterState.state_);
    postProcess(filterState,meas,dt);
    filterState.state_.fix();
    return 0;
  }
  int predictMerged(mtFilterState& filterState, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N){
    switch(mode_){
      case PredictionEKF:
        return predictMergedEKF(filterState,tStart,itMeasStart,N);
      case PredictionUKF:
        return predictMergedUKF(filterState,tStart,itMeasStart,N);
      default:
        return predictMergedEKF(filterState,tStart,itMeasStart,N);
    }
  }
  virtual int predictMergedEKF(mtFilterState& filterState, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N){
    const typename std::map<double,mtMeas>::iterator itMeasEnd = next(itMeasStart,N);
    typename std::map<double,mtMeas>::iterator itMeas = itMeasStart;
    double dT = next(itMeasStart,N-1)->first-tStart;

    // Compute mean Measurement
    mtMeas meanMeas;
    typename mtMeas::mtDifVec vec;
    typename mtMeas::mtDifVec difVec;
    vec.setZero();
    for(itMeas=next(itMeasStart);itMeas!=itMeasEnd;itMeas++){
      itMeasStart->second.boxMinus(itMeas->second,difVec);
      vec = vec + difVec;
    }
    vec = vec/N;
    itMeasStart->second.boxPlus(vec,meanMeas);

    preProcess(filterState,meanMeas,dT);
    this->jacInput(F_,filterState.state_,meanMeas,dT);
    this->jacNoise(Fn_,filterState.state_,meanMeas,dT); // Works for time continuous parametrization of noise
    double t = tStart;
    for(itMeas=itMeasStart;itMeas!=itMeasEnd;itMeas++){
      this->eval(filterState.state_,filterState.state_,itMeas->second,itMeas->first-t);
      t = itMeas->first;
    }
    filterState.cov_ = F_*filterState.cov_*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    postProcess(filterState,meanMeas,dT);
    filterState.state_.fix();
    filterState.cov_ = 0.5*(filterState.cov_+filterState.cov_.transpose()); // Enforce symmetry
    return 0;
  }
  virtual int predictMergedUKF(mtFilterState& filterState, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N){
    refreshNoiseSigmaPoints();
    const typename std::map<double,mtMeas>::iterator itMeasEnd = next(itMeasStart,N);
    typename std::map<double,mtMeas>::iterator itMeas = itMeasStart;
    double dT = next(itMeasStart,N-1)->first-tStart;

    // Compute mean Measurement
    mtMeas meanMeas;
    typename mtMeas::mtDifVec vec;
    typename mtMeas::mtDifVec difVec;
    vec.setZero();
    for(itMeas=next(itMeasStart);itMeas!=itMeasEnd;itMeas++){
      itMeasStart->second.boxMinus(itMeas->second,difVec);
      vec = vec + difVec;
    }
    vec = vec/N;
    itMeasStart->second.boxPlus(vec,meanMeas);

    preProcess(filterState,meanMeas,dT);
    stateSigmaPoints_.computeFromGaussian(filterState.state_,filterState.cov_);

    // Prediction
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      this->eval(stateSigmaPointsPre_(i),stateSigmaPoints_(i),meanMeas,stateSigmaPointsNoi_(i),dT);
    }
    filterState.state_ = stateSigmaPointsPre_.getMean();
    filterState.cov_ = stateSigmaPointsPre_.getCovarianceMatrix(filterState.state_);
    postProcess(filterState,meanMeas,dT);
    filterState.state_.fix();
    return 0;
  }
};

class DummyPrediction: public Prediction<LWF::FilterStateNew<ScalarElement,LWF::ModeEKF,false,false,false,false>,ScalarElement,ScalarElement>{
 public:
  void eval(ScalarElement& output, const ScalarElement& state, const ScalarElement& meas, const ScalarElement noise, double dt) const{
    output.s_ = state.s_ + meas.s_ + noise.s_;
  }
  void jacInput(Eigen::Matrix<double,1,1>& J, const ScalarElement& state, const ScalarElement& meas, double dt) const{
    J.setIdentity();
  }
  void jacNoise(Eigen::Matrix<double,1,1>& J,const ScalarElement& state, const ScalarElement& meas, double dt) const{
    J.setIdentity();
  }
};

}

#endif /* PREDICTIONMODEL_HPP_ */
