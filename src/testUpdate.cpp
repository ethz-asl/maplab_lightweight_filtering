#include "Update.hpp"
#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

class State: public LWF::StateSVQ<0,4,1>{
 public:
  State(){};
  ~State(){};
};
class Meas: public LWF::StateSVQ<0,1,1>{
 public:
  Meas(){};
  ~Meas(){};
};
class Noise: public LWF::VectorState<6>{
 public:
  Noise(){};
  ~Noise(){};
};
class Innovation: public LWF::StateSVQ<0,1,1>{
 public:
  Innovation(){};
  ~Innovation(){};
};

class UpdateExample: public LWF::Update<Innovation,State,Meas,Noise>{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef Innovation mtInnovation;
  UpdateExample(){};
  UpdateExample(const mtMeas& meas): LWF::Update<Innovation,State,Meas,Noise>(meas){};
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

// The fixture for testing class UpdateModel
class UpdateModelTest : public ::testing::Test {
 protected:
  UpdateModelTest() {
    testState_.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    testState_.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(2) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(3) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testMeas_.v(0) = Eigen::Vector3d(-1.5,12,1785.23);
    testMeas_.q(0) = rot::RotationQuaternionPD(3.0/sqrt(15.0),-1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
  }
  virtual ~UpdateModelTest() {
  }
  UpdateExample testUpdate_;
  State testState_;
  Meas testMeas_;
  const double dt_ = 0.1;
};

// Test constructors
TEST_F(UpdateModelTest, constructors) {
  UpdateExample testUpdate;
  ASSERT_EQ((testUpdate.updnoiP_-UpdateExample::mtNoise::mtCovMat::Identity()).norm(),0.0);
  UpdateExample testUpdate2(testMeas_);
  ASSERT_EQ((testUpdate2.updnoiP_-UpdateExample::mtNoise::mtCovMat::Identity()).norm(),0.0);
  UpdateExample::mtMeas::mtDiffVec dif;
  testUpdate2.meas_.boxMinus(testMeas_,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
}

// Test finite difference Jacobians
TEST_F(UpdateModelTest, FDjacobians) {
  UpdateExample::mtJacInput F = testUpdate_.jacInputFD(testState_,testMeas_,dt_,0.0000001);
  ASSERT_NEAR((F-testUpdate_.jacInput(testState_,testMeas_,dt_)).norm(),0.0,1e-5);
  UpdateExample::mtJacNoise Fn = testUpdate_.jacNoiseFD(testState_,testMeas_,dt_,0.0000001);
  ASSERT_NEAR((Fn-testUpdate_.jacNoise(testState_,testMeas_,dt_)).norm(),0.0,1e-5);
}

// Test setMeasurement
TEST_F(UpdateModelTest, setMeasurement) {
  testUpdate_.setMeasurement(testMeas_);
  UpdateExample::mtMeas::mtDiffVec dif;
  testUpdate_.meas_.boxMinus(testMeas_,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
}

// Test updateEKF
TEST_F(UpdateModelTest, updateEKF) {
  testUpdate_.setMeasurement(testMeas_);
  UpdateExample::mtState::mtCovMat cov;
  UpdateExample::mtState::mtCovMat updateCov;
  UpdateExample::mtNoise noise;
  cov.setIdentity();
  UpdateExample::mtJacInput H = testUpdate_.jacInput(testState_,testMeas_,dt_);
  UpdateExample::mtJacNoise Hn = testUpdate_.jacNoise(testState_,testMeas_,dt_);

  UpdateExample::mtInnovation y = testUpdate_.eval(testState_,testUpdate_.meas_,noise);
  UpdateExample::mtInnovation yIdentity;
  UpdateExample::mtInnovation::mtDiffVec innVector;

  UpdateExample::mtState state;
  UpdateExample::mtState stateUpdated;
  state = testState_;

  // Update
  UpdateExample::mtInnovation::mtCovMat Py = H*cov*H.transpose() + Hn*testUpdate_.updnoiP_*Hn.transpose();
  y.boxMinus(yIdentity,innVector);
  UpdateExample::mtInnovation::mtCovMat Pyinv = Py.inverse();

  // Kalman Update
  Eigen::Matrix<double,UpdateExample::mtState::D_,UpdateExample::mtInnovation::D_> K = cov*H.transpose()*Pyinv;
  updateCov = cov - K*Py*K.transpose();
  UpdateExample::mtState::mtDiffVec updateVec;
  updateVec = -K*innVector;
  state.boxPlus(updateVec,stateUpdated);

  testUpdate_.updateEKF(state,cov);
  UpdateExample::mtState::mtDiffVec dif;
  state.boxMinus(stateUpdated,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-updateCov).norm(),0.0,1e-6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
