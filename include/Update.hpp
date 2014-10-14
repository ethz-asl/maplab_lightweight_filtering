/*
 * Update.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef UPDATEMODEL_HPP_
#define UPDATEMODEL_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
#include "ModelBase.hpp"
#include "Prediction.hpp"

namespace LWF{

template<typename State>
class UpdateBase{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  UpdateBase(){};
  virtual ~UpdateBase(){};
  virtual int updateEKF(mtState& state, mtCovMat& cov) = 0;
  virtual int updateUKF(mtState& state, mtCovMat& cov) = 0;
};

template<typename Innovation, typename State, typename Meas, typename Noise>
class Update: public UpdateBase<State>, public ModelBase<State,Innovation,Meas,Noise>{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typename ModelBase<State,Innovation,Meas,Noise>::mtJacInput H_;
  typename ModelBase<State,Innovation,Meas,Noise>::mtJacNoise Hn_;
  typename mtNoise::mtCovMat updnoiP_;
  mtMeas meas_;
  mtInnovation y_;
  typename mtInnovation::mtCovMat Py_;
  typename mtInnovation::mtCovMat Pyinv_;
  typename mtInnovation::mtDiffVec innVector_;
  const mtInnovation yIdentity_;
  typename mtState::mtDiffVec updateVec_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> K_;
  Update(){
    updateVec_.setIdentity();
    updnoiP_.setIdentity();
  };
  Update(const mtMeas& meas){
    updateVec_.setIdentity();
    updnoiP_.setIdentity();
    setMeasurement(meas);
  };
  virtual ~Update(){};
  void setMeasurement(const mtMeas& meas){
    meas_ = meas;
  };
  int updateEKF(mtState& state, mtCovMat& cov){
    H_ = this->jacInput(state,meas_);
    Hn_ = this->jacNoise(state,meas_);
    y_ = this->eval(state,meas_);

    // Update
    Py_ = H_*cov*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    y_.boxMinus(yIdentity_,innVector_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = cov*H_.transpose()*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;
    state.boxPlus(updateVec_,state);
    state.fix();
    return 0;
  }
  int updateUKF(mtState& state, mtCovMat& cov){
    return updateEKF(state,cov);
  }
};

template<typename Innovation, typename State, typename Meas, typename Noise, typename Prediction>
class PredictionUpdate: public UpdateBase<State>, public ModelBase<State,Innovation,Meas,Noise>{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef typename Prediction::mtMeas mtPredictionMeas;
  typedef Noise mtNoise;
  typedef typename Prediction::mtNoise mtPredictionNoise;
  typename ModelBase<State,Innovation,Meas,Noise>::mtJacInput H_;
  typename ModelBase<State,Innovation,Meas,Noise>::mtJacNoise Hn_;
  typename Prediction::mtJacInput F_;
  typename Prediction::mtJacNoise Fn_;
  typename mtNoise::mtCovMat updnoiP_;
  mtMeas meas_;
  mtInnovation y_;
  typename mtInnovation::mtCovMat Py_;
  typename mtInnovation::mtCovMat Pyinv_;
  typename mtInnovation::mtDiffVec innVector_;
  const mtInnovation yIdentity_;
  typename mtState::mtDiffVec updateVec_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> K_;
  Prediction* mpPrediction_;
  PredictionUpdate(){
    updateVec_.setIdentity();
    updnoiP_.setIdentity();
    mpPrediction_ = nullptr;
  };
  PredictionUpdate(const mtMeas& meas){
    updateVec_.setIdentity();
    updnoiP_.setIdentity();
    setMeasurement(meas);
  };
  virtual ~PredictionUpdate(){};
  void setMeasurement(const mtMeas& meas){
    meas_ = meas;
  };
  int predictAndUpdateEKF(mtState& state, mtCovMat& cov, PredictionBase<State>* mpPredictionBase, const double dt){ // TODO: change dt to reference (everywhere)
    // Predict
    mpPrediction_ = static_cast<Prediction>(mpPredictionBase); // TODO: Dangerous
    F_ = mpPrediction_->jacInput(state,mpPrediction_->meas_,dt);
    Fn_ = mpPrediction_->jacNoise(state,mpPrediction_->meas_,dt);
    state = mpPrediction_->eval(state,mpPrediction_->meas_,dt);
    state.fix();
    cov = F_*cov*F_.transpose() + Fn_*mpPrediction_->prenoiP_*Fn_.transpose();

    // Update
    H_ = this->jacInput(state,meas_);
    Hn_ = this->jacNoise(state,meas_);
    y_ = this->eval(state,meas_);
    Py_ = H_*cov*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    y_.boxMinus(yIdentity_,innVector_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = cov*H_.transpose()*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;
    state.boxPlus(updateVec_,state);
    state.fix();
    return 0;
  }
  int predictAndUpdateUKF(mtState& state, mtCovMat& cov, const double dt){
    return predictAndUpdateEKF(state,cov,dt);
  }
};

}

#endif /* UPDATEMODEL_HPP_ */
