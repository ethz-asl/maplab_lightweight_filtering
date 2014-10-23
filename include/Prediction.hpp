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
  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+mtNoise::D_)+1,0> stateSigmaPoints_;
  SigmaPoints<mtNoise,2*mtNoise::D_+1,2*(mtState::D_+mtNoise::D_)+1,2*mtState::D_> stateSigmaPointsNoi_;
  SigmaPoints<mtState,2*(mtState::D_+mtNoise::D_)+1,2*(mtState::D_+mtNoise::D_)+1,0> stateSigmaPointsPre_;
  const bool mbMergePredictions_;
  Prediction(bool mbMergePredictions = false): mbMergePredictions_(mbMergePredictions){
    resetPrediction();
  };
  virtual void preProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){};
  virtual void postProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){};
  void resetPrediction(){
    prenoiP_ = mtNoise::mtCovMat::Identity()*0.0001;
    stateSigmaPoints_.computeParameter(1e-3,2.0,0.0);
    stateSigmaPointsNoi_.computeParameter(1e-3,2.0,0.0);
    stateSigmaPointsPre_.computeParameter(1e-3,2.0,0.0);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(prenoiP_);
  }
  virtual ~Prediction(){};
  int predictEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    preProcess(state,cov,meas,dt);
    F_ = this->jacInput(state,meas,dt);
    Fn_ = this->jacNoise(state,meas,dt);
    state = this->eval(state,meas,dt);
    state.fix();
    cov = F_*cov*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    postProcess(state,cov,meas,dt);
    return 0;
  }
  int predictUKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    preProcess(state,cov,meas,dt);
    stateSigmaPoints_.computeFromGaussian(state,cov);

    // Prediction
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      stateSigmaPointsPre_(i) = this->eval(stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i),dt);
    }

    // Calculate mean and variance
    state = stateSigmaPointsPre_.getMean();
    state.fix();
    cov = stateSigmaPointsPre_.getCovarianceMatrix(state);
    postProcess(state,cov,meas,dt);
    return 0;
  }
  virtual int predictMergedEKF(mtState& state, mtCovMat& cov, double tStart, const typename std::map<double,mtMeas>::iterator itMeasStart, unsigned int N){
    const typename std::map<double,mtMeas>::iterator itMeasEnd = next(itMeasStart,N-1);
    typename std::map<double,mtMeas>::iterator itMeas = itMeasStart;
    double dT = itMeasEnd->first-tStart;
    preProcess(state,cov,itMeasStart->second,dT);
    F_ = this->jacInput(state,itMeasStart->second,dT);
    Fn_ = this->jacNoise(state,itMeasStart->second,dT); // TODO
    double t = tStart;
    for(unsigned int i=0;i<N;i++){
      state = this->eval(state,itMeas->second,itMeas->first-t);
      t = itMeas->first;
      itMeas++;
    }
    state.fix();
    cov = F_*cov*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    postProcess(state,cov,itMeasEnd->second,dT);
    return 0;
  }
};

}

#endif /* PREDICTIONMODEL_HPP_ */
