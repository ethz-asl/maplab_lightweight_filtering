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
  enum StateNames {
    POS,
    VEL,
    ACB,
    GYB,
    ATT
  };
  State(){};
  ~State(){};
};
class UpdateMeas: public LWF::StateSVQ<0,1,1>{
 public:
  enum StateNames {
    POS,
    ATT
  };
  UpdateMeas(){};
  ~UpdateMeas(){};
};
class UpdateNoise: public LWF::StateSVQ<0,2,0>{
 public:
  enum StateNames {
    POS,
    ATT
  };
  UpdateNoise(){};
  ~UpdateNoise(){};
};
class Innovation: public LWF::StateSVQ<0,1,1>{
 public:
  enum StateNames {
    POS,
    ATT
  };
  Innovation(){};
  ~Innovation(){};
};
class PredictionNoise: public LWF::StateSVQ<0,5,0>{
 public:
  enum StateNames {
    POS,
    VEL,
    ACB,
    GYB,
    ATT
  };
  PredictionNoise(){};
  ~PredictionNoise(){};
};
class PredictionMeas: public LWF::StateSVQ<0,2,0>{
 public:
  enum StateNames {
    ACC,
    GYR
  };
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
    inn.getValue<Innovation::POS>() = state.getValue<State::ATT>().rotate(state.getValue<State::POS>())-meas.getValue<UpdateMeas::POS>()+noise.getValue<UpdateNoise::POS>();
    inn.getValue<Innovation::ATT>() = (state.getValue<State::ATT>()*meas.getValue<UpdateMeas::ATT>().inverted()).boxPlus(noise.getValue<UpdateNoise::ATT>());
    return inn;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacInput J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtState::getId<State::POS>()) = rot::RotationMatrixPD(state.getValue<State::ATT>()).matrix();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtState::getId<State::ATT>()) = kindr::linear_algebra::getSkewMatrixFromVector(state.getValue<State::ATT>().rotate(state.getValue<State::POS>()));
    J.template block<3,3>(mtInnovation::getId<Innovation::ATT>(),mtState::getId<State::ATT>()) = Eigen::Matrix3d::Identity();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacNoise J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtNoise::getId<mtNoise::POS>()) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtInnovation::getId<Innovation::ATT>(),mtNoise::getId<mtNoise::ATT>()) = Eigen::Matrix3d::Identity();
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
    Eigen::Vector3d dOmega = -dt*(meas.getValue<PredictionMeas::GYR>()-state.getValue<State::GYB>()+noise.getValue<PredictionNoise::ATT>()/sqrt(dt));
    rot::RotationQuaternionPD dQ = dQ.exponentialMap(dOmega);
    output.q(0) = state.getValue<State::ATT>()*dQ;
    output.q(0).fix();
    output.v(0) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega))*state.getValue<State::POS>()-dt*state.getValue<State::VEL>()+noise.getValue<PredictionNoise::POS>()*sqrt(dt);
    output.v(1) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega))*state.getValue<State::VEL>()
        -dt*(meas.getValue<PredictionMeas::ACC>()-state.getValue<State::ACB>()+state.getValue<State::ATT>().inverseRotate(g_)+noise.getValue<PredictionNoise::VEL>()/sqrt(dt));
    output.v(2) = state.getValue<State::ACB>()+noise.getValue<PredictionNoise::ACB>()*sqrt(dt);
    output.v(3) = state.getValue<State::GYB>()+noise.getValue<PredictionNoise::GYB>()*sqrt(dt);
    return output;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt) const{
    mtJacInput J;
    Eigen::Vector3d g_(0,0,-9.81);
    Eigen::Vector3d dOmega = -dt*(meas.getValue<PredictionMeas::GYR>()-state.getValue<State::GYB>());
    J.setZero();
    J.template block<3,3>(mtState::getId<State::POS>(),mtState::getId<State::POS>()) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega));
    J.template block<3,3>(mtState::getId<State::POS>(),mtState::getId<State::VEL>()) = -dt*Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtState::getId<State::POS>(),mtState::getId<State::GYB>()) = -dt*kindr::linear_algebra::getSkewMatrixFromVector(state.getValue<State::POS>());
    J.template block<3,3>(mtState::getId<State::VEL>(),mtState::getId<State::VEL>()) = (Eigen::Matrix3d::Identity()+kindr::linear_algebra::getSkewMatrixFromVector(dOmega));
    J.template block<3,3>(mtState::getId<State::VEL>(),mtState::getId<State::ACB>()) = dt*Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtState::getId<State::VEL>(),mtState::getId<State::GYB>()) = -dt*kindr::linear_algebra::getSkewMatrixFromVector(state.getValue<State::VEL>());
    J.template block<3,3>(mtState::getId<State::VEL>(),mtState::getId<State::ATT>()) = dt*rot::RotationMatrixPD(state.getValue<State::ATT>()).matrix().transpose()*kindr::linear_algebra::getSkewMatrixFromVector(g_);
    J.template block<3,3>(mtState::getId<State::ACB>(),mtState::getId<State::ACB>()) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtState::getId<State::GYB>(),mtState::getId<State::GYB>()) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtState::getId<State::ATT>(),mtState::getId<State::GYB>()) = dt*rot::RotationMatrixPD(state.getValue<State::ATT>()).matrix()*LWF::Lmat(dOmega);
    J.template block<3,3>(mtState::getId<State::ATT>(),mtState::getId<State::ATT>()) = Eigen::Matrix3d::Identity();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt) const{
    mtJacNoise J;
    mtNoise noise;
    Eigen::Vector3d g_(0,0,-9.81);
    Eigen::Vector3d dOmega = -dt*(meas.getValue<PredictionMeas::GYR>()-state.getValue<State::GYB>());
    J.setZero();
    J.template block<3,3>(mtState::getId<State::POS>(),mtNoise::getId<mtNoise::POS>()) = Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(mtState::getId<State::POS>(),mtNoise::getId<mtNoise::ATT>()) = kindr::linear_algebra::getSkewMatrixFromVector(state.getValue<State::POS>())*sqrt(dt);
    J.template block<3,3>(mtState::getId<State::VEL>(),mtNoise::getId<mtNoise::VEL>()) = -Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(mtState::getId<State::VEL>(),mtNoise::getId<mtNoise::ATT>()) = kindr::linear_algebra::getSkewMatrixFromVector(state.getValue<State::VEL>())*sqrt(dt);
    J.template block<3,3>(mtState::getId<State::ACB>(),mtNoise::getId<mtNoise::ACB>()) = Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(mtState::getId<State::GYB>(),mtNoise::getId<mtNoise::GYB>()) = Eigen::Matrix3d::Identity()*sqrt(dt);
    J.template block<3,3>(mtState::getId<State::ATT>(),mtNoise::getId<mtNoise::ATT>()) = -rot::RotationMatrixPD(state.getValue<State::ATT>()).matrix()*LWF::Lmat(dOmega)*sqrt(dt);
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
    inn.getValue<Innovation::POS>() = state.getValue<State::ATT>().rotate(state.getValue<State::POS>())-meas.getValue<UpdateMeas::POS>()+noise.getValue<UpdateNoise::POS>();
    inn.getValue<Innovation::ATT>() = (state.getValue<State::ATT>()*meas.getValue<UpdateMeas::ATT>().inverted()).boxPlus(noise.getValue<UpdateNoise::ATT>());
    return inn;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacInput J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtState::getId<State::POS>()) = rot::RotationMatrixPD(state.getValue<State::ATT>()).matrix();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtState::getId<State::ATT>()) = kindr::linear_algebra::getSkewMatrixFromVector(state.getValue<State::ATT>().rotate(state.getValue<State::POS>()));
    J.template block<3,3>(mtInnovation::getId<Innovation::ATT>(),mtState::getId<State::ATT>()) = Eigen::Matrix3d::Identity();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacNoise J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtNoise::getId<mtNoise::POS>()) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtInnovation::getId<Innovation::ATT>(),mtNoise::getId<mtNoise::ATT>()) = Eigen::Matrix3d::Identity();
    return J;
  }
};

class NonlinearTest{
 public:
  typedef State mtState;
  typedef UpdateMeas mtUpdateMeas;
  typedef UpdateNoise mtUpdateNoise;
  typedef Innovation mtInnovation;
  typedef PredictionNoise mtPredictionNoise;
  typedef PredictionMeas mtPredictionMeas;
  typedef UpdateExample mtUpdateExample;
  typedef PredictionExample mtPredictionExample;
  typedef PredictAndUpdateExample mtPredictAndUpdateExample;
};

}

#endif /* TestClasses_HPP_ */
