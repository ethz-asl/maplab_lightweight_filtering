#include "Update.hpp"
#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

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

// The fixture for testing class UpdateModel
class UpdateModelTest : public ::testing::Test {
 protected:
  UpdateModelTest() {
    testState_.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    testState_.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(2) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(3) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testUpdateMeas_.v(0) = Eigen::Vector3d(-1.5,12,5.23);
    testUpdateMeas_.q(0) = rot::RotationQuaternionPD(3.0/sqrt(15.0),-1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
    testPredictionMeas_.v(0) = Eigen::Vector3d(-5,2,17.3);
    testPredictionMeas_.v(1) = Eigen::Vector3d(15.7,0.45,-2.3);
  }
  virtual ~UpdateModelTest() {
  }
  UpdateExample testUpdate_;
  PredictAndUpdateExample testPredictAndUpdate_;
  PredictionExample testPrediction_;
  State testState_;
  UpdateMeas testUpdateMeas_;
  PredictionMeas testPredictionMeas_;
  const double dt_ = 0.1;
};

// Test constructors
TEST_F(UpdateModelTest, constructors) {
  UpdateExample testUpdate;
  ASSERT_EQ((testUpdate.updnoiP_-UpdateExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
}

// Test finite difference Jacobians
TEST_F(UpdateModelTest, FDjacobians) {
  UpdateExample::mtJacInput F = testUpdate_.jacInputFD(testState_,testUpdateMeas_,dt_,0.0000001);
  ASSERT_NEAR((F-testUpdate_.jacInput(testState_,testUpdateMeas_,dt_)).norm(),0.0,1e-5);
  UpdateExample::mtJacNoise Fn = testUpdate_.jacNoiseFD(testState_,testUpdateMeas_,dt_,0.0000001);
  ASSERT_NEAR((Fn-testUpdate_.jacNoise(testState_,testUpdateMeas_,dt_)).norm(),0.0,1e-5);
}

// Test updateEKF
TEST_F(UpdateModelTest, updateEKF) {
  UpdateExample::mtState::mtCovMat cov;
  UpdateExample::mtState::mtCovMat updateCov;
  cov.setIdentity();
  UpdateExample::mtJacInput H = testUpdate_.jacInput(testState_,testUpdateMeas_,dt_);
  UpdateExample::mtJacNoise Hn = testUpdate_.jacNoise(testState_,testUpdateMeas_,dt_);

  UpdateExample::mtInnovation y = testUpdate_.eval(testState_,testUpdateMeas_);
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

  testUpdate_.updateEKF(state,cov,testUpdateMeas_);
  UpdateExample::mtState::mtDiffVec dif;
  state.boxMinus(stateUpdated,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-updateCov).norm(),0.0,1e-6);
}

// Test compareUpdate
TEST_F(UpdateModelTest, compareUpdate) {
  UpdateExample::mtState::mtCovMat cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState::mtCovMat cov2 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState state1 = testState_;
  UpdateExample::mtState state2 = testState_;
  testUpdate_.updateEKF(state1,cov1,testUpdateMeas_);
  testUpdate_.updateUKF(state2,cov2,testUpdateMeas_);
  UpdateExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-5); // Careful, will differ depending on the magnitude of the covariance
}

// Test predictAndUpdateEKF
TEST_F(UpdateModelTest, predictAndUpdateEKF) {
  UpdateExample::mtState::mtCovMat cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState::mtCovMat cov2 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState state1 = testState_;
  UpdateExample::mtState state2 = testState_;
  testPrediction_.predictEKF(state1,cov1,testPredictionMeas_,dt_);
  testUpdate_.updateEKF(state1,cov1,testUpdateMeas_);
  testPredictAndUpdate_.predictAndUpdateEKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  UpdateExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-5); // Careful, will differ depending on the magnitude of the covariance
}

// Test predictAndUpdateUKF
TEST_F(UpdateModelTest, predictAndUpdateUKF) {
  UpdateExample::mtState::mtCovMat cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState::mtCovMat cov2 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState state1 = testState_;
  UpdateExample::mtState state2 = testState_;
  testPrediction_.predictUKF(state1,cov1,testPredictionMeas_,dt_);
  testUpdate_.updateUKF(state1,cov1,testUpdateMeas_);
  testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  state1.print();
  state2.print();
  UpdateExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-4); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
