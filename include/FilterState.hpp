/*
 * FilterState.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef FILTERSTATE_HPP_
#define FILTERSTATE_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "SigmaPoints.hpp"

namespace LWF{

enum FilteringMode{
  ModeEKF,
  ModeUKF
};

template<typename State, typename PredictionMeas, typename PredictionNoise, unsigned int noiseExtensionDim = 0>
class FilterState{
 public:
  typedef State mtState;
  typedef PredictionMeas mtPredictionMeas;
  typedef PredictionNoise mtPredictionNoise;
  typedef typename State::mtCovMat mtFilterCovMat;
  typedef Eigen::Matrix<double,State::D_,State::D_> mtJacStaPrediction;
  typedef Eigen::Matrix<double,State::D_,PredictionNoise::D_> mtJacNoiPrediction;
  FilteringMode mode_;
  bool usePredictionMerge_;
  bool useDynamicMatrix_;
  static constexpr unsigned int noiseExtensionDim_ = noiseExtensionDim;
  double t_;
  mtState state_;
  mtFilterCovMat cov_;
  mtJacStaPrediction F_;
  mtJacNoiPrediction G_;
  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+mtPredictionNoise::D_+noiseExtensionDim)+1,0> stateSigmaPoints_;
  SigmaPoints<mtPredictionNoise,2*mtPredictionNoise::D_+1,2*(mtState::D_+mtPredictionNoise::D_+noiseExtensionDim)+1,2*mtState::D_> stateSigmaPointsNoi_;
  SigmaPoints<mtState,2*(mtState::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtPredictionNoise::D_+noiseExtensionDim)+1,0> stateSigmaPointsPre_;
  typename mtPredictionNoise::mtCovMat prenoiP_; // automatic change tracking
  typename mtState::mtDifVec difVecLin_;
  double alpha_;
  double beta_;
  double kappa_;
  FilterState(){
    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    mode_ = ModeEKF;
    usePredictionMerge_ = false;
    useDynamicMatrix_ = false;
    t_ = 0.0;
    prenoiP_ = mtPredictionNoise::mtCovMat::Identity()*0.0001;
    difVecLin_.setIdentity();
    refreshUKFParameter();
  }
  void refreshUKFParameter(){
    stateSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsPre_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(prenoiP_);
  }
  void refreshNoiseSigmaPoints(const typename mtPredictionNoise::mtCovMat& prenoiP){
    if(prenoiP_ != prenoiP){
      prenoiP_ = prenoiP;
      stateSigmaPointsNoi_.computeFromZeroMeanGaussian(prenoiP_);
    }
  }
};

}

#endif /* FILTERSTATE_HPP_ */
