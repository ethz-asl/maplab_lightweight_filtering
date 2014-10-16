#include "Prediction.hpp"
#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

class State: public LWF::StateSVQ<0,4,1>{
 public:
  State(){};
  ~State(){};
};
class Meas: public LWF::StateSVQ<0,2,0>{
 public:
  Meas(){};
  ~Meas(){};
};
class Noise: public LWF::VectorState<15>{
 public:
  Noise(){};
  ~Noise(){};
};

class PredictionExample: public LWF::Prediction<State,Meas,Noise>{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  PredictionExample(){};
  PredictionExample(const mtMeas& meas): LWF::Prediction<State,Meas,Noise>(meas){};
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

// The fixture for testing class PredictionModel
class PredictionModelTest : public ::testing::Test {
 protected:
  PredictionModelTest() {
    testState_.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    testState_.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(2) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(3) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testMeas_.v(0) = Eigen::Vector3d(-1.5,12,1785.23);
    testMeas_.v(1) = Eigen::Vector3d(-1.5,12,1785.23);
  }
  virtual ~PredictionModelTest() {
  }
  PredictionExample testPrediction_;
  State testState_;
  Meas testMeas_;
  const double dt_ = 0.1;
};

// Test constructors
TEST_F(PredictionModelTest, constructors) {
  PredictionExample testPrediction;
  ASSERT_EQ((testPrediction.prenoiP_-PredictionExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
  PredictionExample testPrediction2(testMeas_);
  ASSERT_EQ((testPrediction2.prenoiP_-PredictionExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
  PredictionExample::mtMeas::mtDiffVec dif;
  testPrediction2.meas_.boxMinus(testMeas_,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
}

// Test finite difference Jacobians
TEST_F(PredictionModelTest, FDjacobians) {
  PredictionExample::mtJacInput F = testPrediction_.jacInputFD(testState_,testMeas_,dt_,0.0000001);
  ASSERT_NEAR((F-testPrediction_.jacInput(testState_,testMeas_,dt_)).norm(),0.0,1e-5);
  PredictionExample::mtJacNoise Fn = testPrediction_.jacNoiseFD(testState_,testMeas_,dt_,0.0000001);
  ASSERT_NEAR((Fn-testPrediction_.jacNoise(testState_,testMeas_,dt_)).norm(),0.0,1e-5);
}

// Test setMeasurement
TEST_F(PredictionModelTest, setMeasurement) {
  testPrediction_.setMeasurement(testMeas_);
  PredictionExample::mtMeas::mtDiffVec dif;
  testPrediction_.meas_.boxMinus(testMeas_,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
}

// Test predictEKF
TEST_F(PredictionModelTest, predictEKF) {
  testPrediction_.setMeasurement(testMeas_);
  PredictionExample::mtState::mtCovMat cov;
  cov.setIdentity();
  PredictionExample::mtJacInput F = testPrediction_.jacInput(testState_,testMeas_,dt_);
  PredictionExample::mtJacNoise Fn = testPrediction_.jacNoise(testState_,testMeas_,dt_);
  PredictionExample::mtState::mtCovMat predictedCov = F*cov*F.transpose() + Fn*testPrediction_.prenoiP_*Fn.transpose();
  PredictionExample::mtState state;
  state = testState_;
  testPrediction_.predictEKF(state,cov,dt_);
  PredictionExample::mtState::mtDiffVec dif;
  PredictionExample::mtNoise noise;
  state.boxMinus(testPrediction_.eval(testState_,testMeas_,noise,dt_),dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
}

// Test comparePredict
TEST_F(PredictionModelTest, comparePredict) {
  testPrediction_.setMeasurement(testMeas_);
  PredictionExample::mtState::mtCovMat cov1 = PredictionExample::mtState::mtCovMat::Identity()*0.000001;
  PredictionExample::mtState::mtCovMat cov2 = PredictionExample::mtState::mtCovMat::Identity()*0.000001;
  PredictionExample::mtState state1 = testState_;
  PredictionExample::mtState state2 = testState_;
  testPrediction_.predictEKF(state1,cov1,dt_);
  testPrediction_.predictUKF(state2,cov2,dt_);
  PredictionExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
