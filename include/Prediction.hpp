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
#include <map>

namespace LWF{

enum PredictionFilteringMode{
  PredictionEKF,
  PredictionUKF
};

template<typename State, typename Meas, typename Noise>
class Prediction: public ModelBase<State,State,Meas,Noise>{
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
  Prediction(bool mbMergePredictions = false): mbMergePredictions_(mbMergePredictions){
    resetPrediction();
  };
  void refreshNoiseSigmaPoints(){
    if(noiP_ != prenoiP_){
      noiP_ = prenoiP_;
      stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    }
  }
  void setUKFParameter(double alpha,double beta, double kappa){
    stateSigmaPoints_.computeParameter(alpha,beta,kappa);
    stateSigmaPointsNoi_.computeParameter(alpha,beta,kappa);
    stateSigmaPointsPre_.computeParameter(alpha,beta,kappa);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
  }
  virtual void preProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){};
  virtual void postProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){};
  void resetPrediction(){
    prenoiP_ = mtNoise::mtCovMat::Identity()*0.0001;
    refreshNoiseSigmaPoints();
    setUKFParameter(1e-3,2.0,0.0);
  }
  virtual ~Prediction(){};
  int predict(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt, PredictionFilteringMode mode = PredictionEKF){
    switch(mode){
      case PredictionEKF:
        return predictEKF(state,cov,meas,dt);
      case PredictionUKF:
        return predictUKF(state,cov,meas,dt);
      default:
        return predictEKF(state,cov,meas,dt);
    }
  }
  int predictEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    preProcess(state,cov,meas,dt);
    F_ = this->jacInput(state,meas,dt);
    Fn_ = this->jacNoise(state,meas,dt);
    state = this->eval(state,meas,dt);
    cov = F_*cov*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    postProcess(state,cov,meas,dt);
    state.fix();
    return 0;
  }
  int predictUKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
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
  int predictMerged(mtState& state, mtCovMat& cov, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N, PredictionFilteringMode mode = PredictionEKF){
    switch(mode){
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

}

#endif /* PREDICTIONMODEL_HPP_ */
