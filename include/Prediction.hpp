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
  Prediction(){
    resetPrediction();
  };
  void resetPrediction(){
    prenoiP_ = mtNoise::mtCovMat::Identity()*0.0001;
    stateSigmaPoints_.computeParameter(1e-3,2.0,0.0);
    stateSigmaPointsNoi_.computeParameter(1e-3,2.0,0.0);
    stateSigmaPointsPre_.computeParameter(1e-3,2.0,0.0);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(prenoiP_);
  }
  virtual ~Prediction(){};
  int predictEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    F_ = this->jacInput(state,meas,dt);
    Fn_ = this->jacNoise(state,meas,dt);
    state = this->eval(state,meas,dt);
    state.fix();
    cov = F_*cov*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    return 0;
  }
  int predictUKF(mtState& state, mtCovMat& cov, const mtMeas& meas, double dt){
    stateSigmaPoints_.computeFromGaussian(state,cov);

    // Prediction
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      stateSigmaPointsPre_(i) = this->eval(stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i),dt);
    }

    // Calculate mean and variance
    state = stateSigmaPointsPre_.getMean();
    state.fix();
    cov = stateSigmaPointsPre_.getCovarianceMatrix(state);
    return 0;
  }
};

}

#endif /* PREDICTIONMODEL_HPP_ */
