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

namespace LWF{

template<typename State>
class PredictionBase{
 public:
  typedef State mtState;
  typedef typename mtState::CovMat mtCovMat;
  PredictionBase(){};
  virtual ~PredictionBase(){};
  virtual int predictEKF(mtState& state, mtCovMat& cov, const double t) = 0;
  virtual int predictUKF(mtState& state, mtCovMat& cov, const double t) = 0;
};

template<typename State>
class PredictionDefault: public PredictionBase<State>{
 public:
  typedef State mtState;
  typedef typename mtState::CovMat mtCovMat;
  PredictionDefault(){};
  ~PredictionDefault(){};
  static int predictEKF(mtState& state, mtCovMat& cov, const double t){
    return 0;
  }
  static int predictUKF(mtState& state, mtCovMat& cov, const double t){
    return 0;
  }
};

template<typename State, typename Meas, typename Noise>
class Prediction: public PredictionBase<State>, ModelBase<State,State,Meas,Noise>{
 public:
  typedef State mtState;
  typedef typename mtState::CovMat mtCovMat;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typename ModelBase<State,State,Meas,Noise>::mtJacInput F_;
  typename ModelBase<State,State,Meas,Noise>::mtJacNoise Fn_;
  typename mtNoise::CovMat prenoiP_;
  mtMeas meas_;
  Prediction(){
    prenoiP_.setIdentity();
  };
  Prediction(const mtMeas& meas){
    prenoiP_.setIdentity();
    setMeasurement(meas);
  };
  virtual ~Prediction(){};
  void setMeasurement(const mtMeas& meas){
    meas_ = meas;
  };
  int predictEKF(mtState& state, mtCovMat& cov, const double dt){
    F_ = this->jacInput(state,meas_,dt);
    Fn_ = this->jacNoise(state,meas_,dt);
    state = this->eval(state,meas_,dt);
    state.fix();
    cov = F_*cov*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    return 0;
  }
  int predictUKF(mtState& state, mtCovMat& cov, const double dt){
    return predictEKF(state,cov,dt);
  }
};

}

#endif /* PREDICTIONMODEL_HPP_ */
