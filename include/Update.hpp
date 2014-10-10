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

namespace LWF{

template<typename State>
class UpdateBase{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  UpdateBase(){};
  virtual ~UpdateBase(){};
  virtual int updateEKF(mtState& state, mtCovMat& cov, const double t) = 0;
  virtual int updateUKF(mtState& state, mtCovMat& cov, const double t) = 0;
};

template<typename Innovation, typename State, typename Meas, typename Noise>
class Update: public UpdateBase<State>, ModelBase<Innovation,State,Meas,Noise>{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typename ModelBase<Innovation,State,Meas,Noise>::mtJacInput H_;
  typename ModelBase<Innovation,State,Meas,Noise>::mtJacNoise Hn_;
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
  };
  Update(const mtMeas& meas){
    updateVec_.setIdentity();
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
    y_.boxminus(yIdentity_,innVector_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = cov*H_.transpose()*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;
    state.boxplus(updateVec_,state);
    state.fix();
    return 0;
  }
  int updateUKF(mtState& state, mtCovMat& cov){
    return updateEKF(state,cov);
  }
};

}

#endif /* UPDATEMODEL_HPP_ */
