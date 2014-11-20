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

namespace Nonlinear{

class State: public LWF::StateSVQ<0,4,1>{
 public:
  enum StateNames {
    POS,
    VEL,
    ACB,
    GYB,
    ATT
  };
  State(){
    getState<1>().getState<0>().name_ = "pos";
    getState<1>().getState<1>().name_ = "vel";
    getState<1>().getState<2>().name_ = "acb";
    getState<1>().getState<3>().name_ = "gyb";
    getState<2>().getState<0>().name_ = "att";
  }
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
class OutlierDetectionExample: public LWF::OutlierDetection<0,3>{
};

class UpdateExample: public LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,LWF::DummyPrediction,false,OutlierDetectionExample>{
 public:
  using LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,LWF::DummyPrediction,false,OutlierDetectionExample>::eval;
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

class PredictAndUpdateExample: public LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,PredictionExample,true,OutlierDetectionExample>{
 public:
  using LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,PredictionExample,true,OutlierDetectionExample>::eval;
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

}

namespace Linear{

class State: public LWF::StateSVQ<0,2,0>{
 public:
  enum StateNames {
    POS,
    VEL
  };
  State(){
    createDefaultNames("s");
  };
  ~State(){};
};
class UpdateMeas: public LWF::StateSVQ<1,1,0>{
 public:
  enum StateNames {
    HEI,
    POS
  };
  UpdateMeas(){};
  ~UpdateMeas(){};
};
class UpdateNoise: public LWF::StateSVQ<1,1,0>{
 public:
  enum StateNames {
    HEI,
    POS
  };
  UpdateNoise(){};
  ~UpdateNoise(){};
};
class Innovation: public LWF::StateSVQ<1,1,0>{
 public:
  enum StateNames {
    HEI,
    POS
  };
  Innovation(){};
  ~Innovation(){};
};
class PredictionNoise: public LWF::StateSVQ<0,1,0>{
 public:
  enum StateNames {
    VEL
  };
  PredictionNoise(){};
  ~PredictionNoise(){};
};
class PredictionMeas: public LWF::StateSVQ<0,1,0>{
 public:
  enum StateNames {
    ACC
  };
  PredictionMeas(){};
  ~PredictionMeas(){};
};
class OutlierDetectionExample: public LWF::OutlierDetection<0,3>{
};

class UpdateExample: public LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,LWF::DummyPrediction,false,OutlierDetectionExample>{
 public:
  using LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,LWF::DummyPrediction,false,OutlierDetectionExample>::eval;
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef UpdateMeas mtMeas;
  typedef UpdateNoise mtNoise;
  typedef Innovation mtInnovation;
  UpdateExample(){};
  ~UpdateExample(){};
  mtInnovation eval(const mtState& state, const mtMeas& meas, const mtNoise noise, double dt = 0.0) const{
    mtInnovation inn;
    inn.getValue<Innovation::POS>() = state.getValue<State::POS>()-meas.getValue<UpdateMeas::POS>()+noise.getValue<UpdateNoise::POS>();
    inn.getValue<Innovation::HEI>() = Eigen::Vector3d(0,0,1).dot(state.getValue<State::POS>())-meas.getValue<UpdateMeas::HEI>()+noise.getValue<UpdateNoise::HEI>();;
    return inn;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacInput J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtState::getId<State::POS>()) = Eigen::Matrix3d::Identity();
    J.template block<1,3>(mtInnovation::getId<Innovation::HEI>(),mtState::getId<State::POS>()) = Eigen::Vector3d(0,0,1).transpose();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacNoise J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtNoise::getId<mtNoise::POS>()) = Eigen::Matrix3d::Identity();
    J(mtInnovation::getId<Innovation::HEI>(),mtNoise::getId<mtNoise::HEI>()) = 1.0;
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
    output.getValue<mtState::POS>() = state.getValue<mtState::POS>()+dt*state.getValue<mtState::VEL>();
    output.getValue<mtState::VEL>()  = state.getValue<mtState::VEL>()+dt*(meas.getValue<mtMeas::ACC>()+noise.getValue<PredictionNoise::VEL>()/sqrt(dt));
    return output;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt) const{
    mtJacInput J;
    J.setZero();
    J.template block<3,3>(mtState::getId<State::POS>(),mtState::getId<State::POS>()) = Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtState::getId<State::POS>(),mtState::getId<State::VEL>()) = dt*Eigen::Matrix3d::Identity();
    J.template block<3,3>(mtState::getId<State::VEL>(),mtState::getId<State::VEL>()) = Eigen::Matrix3d::Identity();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt) const{
    mtJacNoise J;
    J.setZero();
    J.template block<3,3>(mtState::getId<State::VEL>(),mtNoise::getId<mtNoise::VEL>()) = Eigen::Matrix3d::Identity()*sqrt(dt);
    return J;
  }
};

class PredictAndUpdateExample: public LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,PredictionExample,true,OutlierDetectionExample>{
 public:
  using LWF::Update<Innovation,State,UpdateMeas,UpdateNoise,PredictionExample,true,OutlierDetectionExample>::eval;
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef UpdateMeas mtMeas;
  typedef UpdateNoise mtNoise;
  typedef Innovation mtInnovation;
  PredictAndUpdateExample(){};
  ~PredictAndUpdateExample(){};
  mtInnovation eval(const mtState& state, const mtMeas& meas, const mtNoise noise, double dt = 0.0) const{
    mtInnovation inn;
    inn.getValue<Innovation::POS>() = state.getValue<State::POS>()-meas.getValue<UpdateMeas::POS>()+noise.getValue<UpdateNoise::POS>();
    inn.getValue<Innovation::HEI>() = Eigen::Vector3d(0,0,1).dot(state.getValue<State::POS>())-meas.getValue<UpdateMeas::HEI>()+noise.getValue<UpdateNoise::HEI>();;
    return inn;
  }
  mtJacInput jacInput(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacInput J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtState::getId<State::POS>()) = Eigen::Matrix3d::Identity();
    J.template block<1,3>(mtInnovation::getId<Innovation::HEI>(),mtState::getId<State::POS>()) = Eigen::Vector3d(0,0,1).transpose();
    return J;
  }
  mtJacNoise jacNoise(const mtState& state, const mtMeas& meas, double dt = 0.0) const{
    mtJacNoise J;
    mtInnovation inn;
    J.setZero();
    J.template block<3,3>(mtInnovation::getId<Innovation::POS>(),mtNoise::getId<mtNoise::POS>()) = Eigen::Matrix3d::Identity();
    J(mtInnovation::getId<Innovation::HEI>(),mtNoise::getId<mtNoise::HEI>()) = 1.0;
    return J;
  }
};

}

class NonlinearTest{
 public:
  static const int id_ = 0;
  typedef Nonlinear::State mtState;
  typedef Nonlinear::UpdateMeas mtUpdateMeas;
  typedef Nonlinear::UpdateNoise mtUpdateNoise;
  typedef Nonlinear::Innovation mtInnovation;
  typedef Nonlinear::PredictionNoise mtPredictionNoise;
  typedef Nonlinear::PredictionMeas mtPredictionMeas;
  typedef Nonlinear::UpdateExample mtUpdateExample;
  typedef Nonlinear::PredictionExample mtPredictionExample;
  typedef Nonlinear::PredictAndUpdateExample mtPredictAndUpdateExample;
  typedef Nonlinear::OutlierDetectionExample mtOutlierDetectionExample;
  void init(mtState& state,mtUpdateMeas& updateMeas,mtPredictionMeas& predictionMeas){
    state.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    state.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    state.v(2) = Eigen::Vector3d(0.3,10.9,2.3);
    state.v(3) = Eigen::Vector3d(0.3,10.9,2.3);
    state.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    updateMeas.v(0) = Eigen::Vector3d(-1.5,12,5.23);
    updateMeas.q(0) = rot::RotationQuaternionPD(3.0/sqrt(15.0),-1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
    predictionMeas.v(0) = Eigen::Vector3d(-5,2,17.3);
    predictionMeas.v(1) = Eigen::Vector3d(15.7,0.45,-2.3);
  }
};

class LinearTest{
 public:
  static const int id_ = 1;
  typedef Linear::State mtState;
  typedef Linear::UpdateMeas mtUpdateMeas;
  typedef Linear::UpdateNoise mtUpdateNoise;
  typedef Linear::Innovation mtInnovation;
  typedef Linear::PredictionNoise mtPredictionNoise;
  typedef Linear::PredictionMeas mtPredictionMeas;
  typedef Linear::UpdateExample mtUpdateExample;
  typedef Linear::PredictionExample mtPredictionExample;
  typedef Linear::PredictAndUpdateExample mtPredictAndUpdateExample;
  typedef Linear::OutlierDetectionExample mtOutlierDetectionExample;
  void init(mtState& state,mtUpdateMeas& updateMeas,mtPredictionMeas& predictionMeas){
    state.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    state.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    updateMeas.v(0) = Eigen::Vector3d(-1.5,12,5.23);
    updateMeas.s(0) = 0.5;
    predictionMeas.v(0) = Eigen::Vector3d(-5,2,17.3);
  }
};

}

#endif /* TestClasses_HPP_ */
