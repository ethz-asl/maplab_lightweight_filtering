/*
 * FilterState.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_FILTERSTATE_HPP_
#define LWF_FILTERSTATE_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "lightweight_filtering/SigmaPoints.hpp"
#include "lightweight_filtering/common.hpp"

namespace LWF{

enum FilteringMode{
  ModeEKF,
  ModeUKF,
  ModeIEKF
};

template<typename State, typename PredictionMeas, typename PredictionNoise, unsigned int noiseExtensionDim = 0,bool useDynamicMatrix = false>
class FilterState{
 public:
  typedef State mtState;
  typedef PredictionMeas mtPredictionMeas;
  typedef PredictionNoise mtPredictionNoise;
  typedef LWFMatrix<State::D_,State::D_,useDynamicMatrix> mtFilterCovMat;
  typedef LWFMatrix<State::D_,PredictionNoise::D_,useDynamicMatrix> mtJacNoiPrediction;
  typedef LWFMatrix<PredictionNoise::D_,PredictionNoise::D_,useDynamicMatrix> mtPredictionNoiseMat;
  FilteringMode mode_;
  bool usePredictionMerge_;
  static constexpr unsigned int noiseExtensionDim_ = noiseExtensionDim;
  static constexpr bool useDynamicMatrix_ = useDynamicMatrix;
  double t_;
  mtState state_;
  mtFilterCovMat cov_;
  mtFilterCovMat F_;
  mtJacNoiPrediction G_;
  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+mtPredictionNoise::D_+noiseExtensionDim)+1,0,useDynamicMatrix> stateSigmaPoints_;
  SigmaPoints<mtPredictionNoise,2*mtPredictionNoise::D_+1,2*(mtState::D_+mtPredictionNoise::D_+noiseExtensionDim)+1,2*mtState::D_,useDynamicMatrix> stateSigmaPointsNoi_;
  SigmaPoints<mtState,2*(mtState::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtPredictionNoise::D_+noiseExtensionDim)+1,0,useDynamicMatrix> stateSigmaPointsPre_;
  mtPredictionNoiseMat prenoiP_; // automatic change tracking
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
    t_ = 0.0;
    state_.setIdentity();
    cov_.setIdentity();
    F_.setIdentity();
    G_.setZero();
    prenoiP_.setIdentity();
    prenoiP_ *= 0.0001;
    difVecLin_.setIdentity();
    refreshUKFParameter();
  }
  void refreshUKFParameter(){
    stateSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsPre_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(prenoiP_);
  }
  void refreshNoiseSigmaPoints(const mtPredictionNoiseMat& prenoiP){
    if(prenoiP_ != prenoiP){
      prenoiP_ = prenoiP;
      stateSigmaPointsNoi_.computeFromZeroMeanGaussian(prenoiP_);
    }
  }
};

}

#endif /* LWF_FILTERSTATE_HPP_ */
