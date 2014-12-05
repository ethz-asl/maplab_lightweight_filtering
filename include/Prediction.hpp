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

template<typename State, typename Meas, typename Noise>
class Prediction: public ModelBase<State,State,Meas,Noise>, public PropertyHandler{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typename ModelBase<State,State,Meas,Noise>::mtJacInput F_;
  typename ModelBase<State,State,Meas,Noise>::mtJacNoise Fn_;
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
    resetPrediction();
    doubleRegister_.registerDiagonalMatrix("PredictionNoise",prenoiP_);
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
  virtual void noMeasCase(mtState& state, mtCovMat& cov, mtMeas& meas, double dt){};
  virtual void preProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){};
  virtual void postProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){};
  void resetPrediction(){
    refreshNoiseSigmaPoints();
    refreshUKFParameter();
  }
  virtual ~Prediction(){};
  void setMode(PredictionFilteringMode mode){
    mode_ = mode;
  }
  int performPrediction(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    switch(mode_){
      case PredictionEKF:
        return performPredictionEKF(state,cov,meas,dt);
      case PredictionUKF:
        return performPredictionUKF(state,cov,meas,dt);
      default:
        return performPredictionEKF(state,cov,meas,dt);
    }
  }
  int performPrediction(mtState& state, mtCovMat& cov, double dt){
    mtMeas meas;
    meas.setIdentity();
    noMeasCase(state,cov,meas,dt);
    switch(mode_){
      case PredictionEKF:
        return performPredictionEKF(state,cov,meas,dt);
      case PredictionUKF:
        return performPredictionUKF(state,cov,meas,dt);
      default:
        return performPredictionEKF(state,cov,meas,dt);
    }
  }
  int performPredictionEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    preProcess(state,cov,meas,dt);
    F_ = this->jacInput(state,meas,dt);
    Fn_ = this->jacNoise(state,meas,dt);
    state = this->eval(state,meas,dt);
    cov = F_*cov*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    postProcess(state,cov,meas,dt);
    state.fix();
    return 0;
  }
  int performPredictionUKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    refreshNoiseSigmaPoints();
    preProcess(state,cov,meas,dt);
    stateSigmaPoints_.computeFromGaussian(state,cov);

    // Prediction
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      stateSigmaPointsPre_(i) = this->eval(stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i),dt);
    }
    // Calculate mean and variance
    state = stateSigmaPointsPre_.getMean();
    cov = stateSigmaPointsPre_.getCovarianceMatrix(state);
    postProcess(state,cov,meas,dt);
    state.fix();
    return 0;
  }
  int predictMerged(mtState& state, mtCovMat& cov, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N){
    switch(mode_){
      case PredictionEKF:
        return predictMergedEKF(state,cov,tStart,itMeasStart,N);
      case PredictionUKF:
        return predictMergedUKF(state,cov,tStart,itMeasStart,N);
      default:
        return predictMergedEKF(state,cov,tStart,itMeasStart,N);
    }
  }
  virtual int predictMergedEKF(mtState& state, mtCovMat& cov, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N){
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

    preProcess(state,cov,meanMeas,dT);
    F_ = this->jacInput(state,meanMeas,dT);
    Fn_ = this->jacNoise(state,meanMeas,dT); // Works for time continuous parametrization of noise
    double t = tStart;
    for(itMeas=itMeasStart;itMeas!=itMeasEnd;itMeas++){
      state = this->eval(state,itMeas->second,itMeas->first-t);
      t = itMeas->first;
    }
    cov = F_*cov*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    postProcess(state,cov,meanMeas,dT);
    state.fix();
    return 0;
  }
  virtual int predictMergedUKF(mtState& state, mtCovMat& cov, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N){
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

    preProcess(state,cov,meanMeas,dT);
    stateSigmaPoints_.computeFromGaussian(state,cov);

    // Prediction
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      stateSigmaPointsPre_(i) = this->eval(stateSigmaPoints_(i),meanMeas,stateSigmaPointsNoi_(i),dT);
    }
    state = stateSigmaPointsPre_.getMean();
    cov = stateSigmaPointsPre_.getCovarianceMatrix(state);
    postProcess(state,cov,meanMeas,dT);
    state.fix();
    return 0;
  }
};

class DummyPrediction: public Prediction<ScalarElement,ScalarElement,ScalarElement>{
 public:
  ScalarElement eval(const ScalarElement& state, const ScalarElement& meas, const ScalarElement noise, double dt) const{
    ScalarElement output;
    output.s_ = state.s_ + meas.s_ + noise.s_;
    return output;
  }
  Eigen::Matrix<double,1,1> jacInput(const ScalarElement& state, const ScalarElement& meas, double dt) const{
    Eigen::Matrix<double,1,1> J;
    J.setIdentity();
    return J;
  }
  Eigen::Matrix<double,1,1> jacNoise(const ScalarElement& state, const ScalarElement& meas, double dt) const{
    Eigen::Matrix<double,1,1> J;
    J.setIdentity();
    return J;
  }
};

}

#endif /* PREDICTIONMODEL_HPP_ */
