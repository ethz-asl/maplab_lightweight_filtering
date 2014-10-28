/*
 * TestClasses.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef TestClasses_HPP_
#define TestClasses_HPP_

#include "State.hpp"
#include "Update.hpp"
#include "Prediction.hpp"

namespace LWFTest{

class State: public LWF::StateSVQ<0,4,1>{
 public:
  State(){};
  ~State(){};
};
class UpdateMeas: public LWF::StateSVQ<0,1,1>{
 public:
  UpdateMeas(){};
  ~UpdateMeas(){};
};
class UpdateNoise: public LWF::VectorState<6>{
 public:
  UpdateNoise(){};
  ~UpdateNoise(){};
};
class Innovation: public LWF::StateSVQ<0,1,1>{
 public:
  Innovation(){};
  ~Innovation(){};
};
class PredictionNoise: public LWF::VectorState<15>{
 public:
  PredictionNoise(){};
  ~PredictionNoise(){};
};
class PredictionMeas: public LWF::StateSVQ<0,2,0>{
 public:
  PredictionMeas(){};
  ~PredictionMeas(){};
};

class UpdateExample: public LWF::Update<Innovation,State,UpdateMeas,UpdateNoise>{
 public:
  using LWF::Update<Innovation,State,UpdateMeas,UpdateNoise>::eval;
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef UpdateMeas mtMeas;
  typedef UpdateNoise mtNoise;
  typedef Innovation mtInnovation;
  UpdateExample(){};
  ~UpdateExample(){};
  mtInnovation eval(const mtState& state, const mtMeas& meas, const mtNoise noise, double dt = 0.0) const{
    mtInnovation inn;
    inn.v(0) = state.q(0).rotate(state.v(0))-meas.v(0)+noise.block<3>(0);
    inn.q(0) = (state.q(0)*meas.q(0).inverted()).boxPlus(noise.block<3>(3));
    return inn;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacInput J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(inn.getId(inn.v(0)),state.getId(state.v(0))) = rot::RotationMatrixPD(state.q(0)).matrix();
    J.template block<3,3>(inn.getId(inn.v(0)),state.getId(state.q(0))) = kindr::linear_algebra::getSkewMatrixFromVector(state.q(0).rotate(state.v(0)));
    J.template block<3,3>(inn.getId(inn.q(0)),state.getId(state.q(0))) = Eigen::Matrix3d::Identity();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacNoise J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(inn.getId(inn.v(0)),0) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(inn.getId(inn.q(0)),3) = Eigen::Matrix3d::Identity();
    return J;
  }
};

class PredictionExample: public LWF::Prediction<State,PredictionMeas,PredictionNoise>{
 public:
  using LWF::Prediction<State,PredictionMeas,PredictionNoise>::eval;
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef PredictionMeas mtMeas;
  typedef PredictionNoise mtNoise;
  PredictionExample(){};
  ~PredictionExample(){};
  mtState eval(const mtState& state, const mtMeas& meas, const mtNoise noise, double dt) const{
    mtState output;
    Eigen::Vector3d g_(0,0,-9.81);
    Eigen::Vector3d dOmega = -dt*(meas.v(1)-state.v(3)+noise.block<3>(6)/sqrt(dt));
    rot::RotationQuaternionPD dQ = dQ.exponentialMap(dOmega);
    output.q(0) = state.q(0)*dQ;
    output.q(0).fix();
    output.v(0) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega))*state.v(0)-dt*state.v(1)+noise.block<3>(0)*sqrt(dt);
    output.v(1) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega))*state.v(1)
        -dt*(meas.v(0)-state.v(2)+state.q(0).inverseRotate(g_)+noise.block<3>(3)/sqrt(dt));
    output.v(2) = state.v(2)+noise.block<3>(9)*sqrt(dt);
    output.v(3) = state.v(3)+noise.block<3>(12)*sqrt(dt);
    return output;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt) const{
    mtJacInput J;
    Eigen::Vector3d g_(0,0,-9.81);
    Eigen::Vector3d dOmega = -dt*(meas.v(1)-state.v(3));
    J.setZero();
    J.template block<3,3>(state.getId(state.v(0)),state.getId(state.v(0))) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega));
    J.template block<3,3>(state.getId(state.v(0)),state.getId(state.v(1))) = -dt*Eigen::Matrix3d::Identity();
    J.template block<3,3>(state.getId(state.v(0)),state.getId(state.v(3))) = -dt*kindr::linear_algebra::getSkewMatrixFromVector(state.v(0));
    J.template block<3,3>(state.getId(state.v(1)),state.getId(state.v(1))) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega));
    J.template block<3,3>(state.getId(state.v(1)),state.getId(state.v(2))) = dt*Eigen::Matrix3d::Identity();
    J.template block<3,3>(state.getId(state.v(1)),state.getId(state.v(3))) = -dt*kindr::linear_algebra::getSkewMatrixFromVector(state.v(1));
    J.template block<3,3>(state.getId(state.v(1)),state.getId(state.q(0))) = dt*rot::RotationMatrixPD(state.q(0)).matrix().transpose()*kindr::linear_algebra::getSkewMatrixFromVector(g_);
    J.template block<3,3>(state.getId(state.v(2)),state.getId(state.v(2))) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(state.getId(state.v(3)),state.getId(state.v(3))) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(state.getId(state.q(0)),state.getId(state.v(3))) = dt*rot::RotationMatrixPD(state.q(0)).matrix()*LWF::Lmat(dOmega);
    J.template block<3,3>(state.getId(state.q(0)),state.getId(state.q(0))) = Eigen::Matrix3d::Identity();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt) const{
    mtJacNoise J;
    Eigen::Vector3d g_(0,0,-9.81);
    Eigen::Vector3d dOmega = -dt*(meas.v(1)-state.v(3));
    J.setZero();
    J.template block<3,3>(state.getId(state.v(0)),0) = Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(state.getId(state.v(0)),6) = kindr::linear_algebra::getSkewMatrixFromVector(state.v(0))*sqrt(dt);
    J.template block<3,3>(state.getId(state.v(1)),3) = -Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(state.getId(state.v(1)),6) = kindr::linear_algebra::getSkewMatrixFromVector(state.v(1))*sqrt(dt);
    J.template block<3,3>(state.getId(state.v(2)),9) = Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(state.getId(state.v(3)),12) = Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(state.getId(state.q(0)),6) = -rot::RotationMatrixPD(state.q(0)).matrix()*LWF::Lmat(dOmega)*sqrt(dt);
    return J;
  }
};

class PredictAndUpdateExample: public LWF::PredictionUpdate<Innovation,State,UpdateMeas,UpdateNoise,PredictionExample>{
 public:
  using LWF::PredictionUpdate<Innovation,State,UpdateMeas,UpdateNoise,PredictionExample>::eval;
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef UpdateMeas mtMeas;
  typedef UpdateNoise mtNoise;
  typedef Innovation mtInnovation;
  PredictAndUpdateExample(){};
  ~PredictAndUpdateExample(){};
  mtInnovation eval(const mtState& state, const mtMeas& meas, const mtNoise noise, double dt = 0.0) const{
    mtInnovation inn;
    inn.v(0) = state.q(0).rotate(state.v(0))-meas.v(0)+noise.block<3>(0);
    inn.q(0) = (state.q(0)*meas.q(0).inverted()).boxPlus(noise.block<3>(3));
    return inn;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacInput J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(inn.getId(inn.v(0)),state.getId(state.v(0))) = rot::RotationMatrixPD(state.q(0)).matrix();
    J.template block<3,3>(inn.getId(inn.v(0)),state.getId(state.q(0))) = kindr::linear_algebra::getSkewMatrixFromVector(state.q(0).rotate(state.v(0)));
    J.template block<3,3>(inn.getId(inn.q(0)),state.getId(state.q(0))) = Eigen::Matrix3d::Identity();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacNoise J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(inn.getId(inn.v(0)),0) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(inn.getId(inn.q(0)),3) = Eigen::Matrix3d::Identity();
    return J;
  }
};

}

#endif /* TestClasses_HPP_ */
