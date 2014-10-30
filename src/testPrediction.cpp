#include "TestClasses.hpp"
#include "gtest/gtest.h"
#include <assert.h>

using namespace LWFTest;

// The fixture for testing class PredictionModel
class PredictionModelTest : public ::testing::Test {
 protected:
  PredictionModelTest() {
    testState_.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    testState_.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(2) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.v(3) = Eigen::Vector3d(0.3,10.9,2.3);
    testState_.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testMeas_.v(0) = Eigen::Vector3d(2.3,5.6,3.1);
    testMeas_.v(1) = Eigen::Vector3d(-2,2,-2);
    measMap_[0.1] = testMeas_;
    testMeas_.v(0) = Eigen::Vector3d(-1.5,12,15.23);
    testMeas_.v(1) = Eigen::Vector3d(-5,2,5.2);
    measMap_[0.2] = testMeas_;
    testMeas_.v(0) = Eigen::Vector3d(3,-4.5,0.0);
    testMeas_.v(1) = Eigen::Vector3d(-1.2,2.1,1.1);
    measMap_[0.4] = testMeas_;
  }
  virtual ~PredictionModelTest() {
  }
  PredictionExample testPrediction_;
  State testState_;
  PredictionMeas testMeas_;
  const double dt_ = 0.1;
  std::map<double,PredictionMeas> measMap_;
};

// Test constructors
TEST_F(PredictionModelTest, constructors) {
  PredictionExample testPrediction;
  ASSERT_EQ((testPrediction.prenoiP_-PredictionExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
  typename PredictionExample::mtNoise::mtDiffVec dif;
  testPrediction.stateSigmaPointsNoi_.getMean().boxMinus(typename PredictionExample::mtNoise(),dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((testPrediction.prenoiP_-testPrediction.stateSigmaPointsNoi_.getCovarianceMatrix()).norm(),0.0,1e-8);
}

// Test finite difference Jacobians
TEST_F(PredictionModelTest, FDjacobians) {
  PredictionExample::mtJacInput F = testPrediction_.jacInputFD(testState_,testMeas_,dt_,0.0000001);
  ASSERT_NEAR((F-testPrediction_.jacInput(testState_,testMeas_,dt_)).norm(),0.0,1e-5);
  PredictionExample::mtJacNoise Fn = testPrediction_.jacNoiseFD(testState_,testMeas_,dt_,0.0000001);
  ASSERT_NEAR((Fn-testPrediction_.jacNoise(testState_,testMeas_,dt_)).norm(),0.0,1e-5);
}

// Test predictEKF
TEST_F(PredictionModelTest, predictEKF) {
  PredictionExample::mtState::mtCovMat cov;
  cov.setIdentity();
  PredictionExample::mtJacInput F = testPrediction_.jacInput(testState_,testMeas_,dt_);
  PredictionExample::mtJacNoise Fn = testPrediction_.jacNoise(testState_,testMeas_,dt_);
  PredictionExample::mtState::mtCovMat predictedCov = F*cov*F.transpose() + Fn*testPrediction_.prenoiP_*Fn.transpose();
  PredictionExample::mtState state;
  state = testState_;
  testPrediction_.predictEKF(state,cov,testMeas_,dt_);
  PredictionExample::mtState::mtDiffVec dif;
  state.boxMinus(testPrediction_.eval(testState_,testMeas_,dt_),dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
}

// Test comparePredict
TEST_F(PredictionModelTest, comparePredict) {
  PredictionExample::mtState::mtCovMat cov1 = PredictionExample::mtState::mtCovMat::Identity()*0.000001;
  PredictionExample::mtState::mtCovMat cov2 = PredictionExample::mtState::mtCovMat::Identity()*0.000001;
  PredictionExample::mtState state1 = testState_;
  PredictionExample::mtState state2 = testState_;
  testPrediction_.predictEKF(state1,cov1,testMeas_,dt_);
  testPrediction_.predictUKF(state2,cov2,testMeas_,dt_);
  PredictionExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-5); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance
}

// Test predictMergedEKF
TEST_F(PredictionModelTest, predictMergedEKF) {
  PredictionExample::mtState::mtCovMat cov;
  cov.setIdentity();
  double t = 0;
  double dt = measMap_.rbegin()->first-t;

  PredictionExample::mtMeas meanMeas;
  typename PredictionExample::mtMeas::mtDiffVec vec;
  typename PredictionExample::mtMeas::mtDiffVec difVec;
  vec.setZero();
  for(std::map<double,PredictionMeas>::iterator it = next(measMap_.begin());it != measMap_.end();it++){
    measMap_.begin()->second.boxMinus(it->second,difVec);
    vec = vec + difVec;
  }
  vec = vec/measMap_.size();
  measMap_.begin()->second.boxPlus(vec,meanMeas);

  PredictionExample::mtJacInput F = testPrediction_.jacInput(testState_,meanMeas,dt);
  PredictionExample::mtJacNoise Fn = testPrediction_.jacNoise(testState_,meanMeas,dt);
  PredictionExample::mtState::mtCovMat predictedCov = F*cov*F.transpose() + Fn*testPrediction_.prenoiP_*Fn.transpose();
  PredictionExample::mtState state1;
  state1 = testState_;
  testPrediction_.predictMergedEKF(state1,cov,0.0,measMap_.begin(),measMap_.size());
  PredictionExample::mtState state2;
  state2 = testState_;
  for(std::map<double,PredictionMeas>::iterator it = measMap_.begin();it != measMap_.end();it++){
    state2 = testPrediction_.eval(state2,it->second,it->first-t);
    t = it->first;
  }
  PredictionExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
}

// Test predictMergedUKF
TEST_F(PredictionModelTest, predictMergedUKF) {
  PredictionExample::mtState::mtCovMat cov;
  cov.setIdentity();
  double t = 0;
  double dt = measMap_.rbegin()->first-t;

  PredictionExample::mtMeas meanMeas;
  typename PredictionExample::mtMeas::mtDiffVec vec;
  typename PredictionExample::mtMeas::mtDiffVec difVec;
  vec.setZero();
  for(std::map<double,PredictionMeas>::iterator it = next(measMap_.begin());it != measMap_.end();it++){
    measMap_.begin()->second.boxMinus(it->second,difVec);
    vec = vec + difVec;
  }
  vec = vec/measMap_.size();
  measMap_.begin()->second.boxPlus(vec,meanMeas);

  testPrediction_.stateSigmaPoints_.computeFromGaussian(testState_,cov);
  for(unsigned int i=0;i<testPrediction_.stateSigmaPoints_.L_;i++){
    testPrediction_.stateSigmaPointsPre_(i) = testPrediction_.eval(testPrediction_.stateSigmaPoints_(i),meanMeas,testPrediction_.stateSigmaPointsNoi_(i),dt);
  }
  PredictionExample::mtState state1 = testPrediction_.stateSigmaPointsPre_.getMean();
  PredictionExample::mtState::mtCovMat predictedCov = testPrediction_.stateSigmaPointsPre_.getCovarianceMatrix(state1);
  PredictionExample::mtState state2;
  state2 = testState_;
  testPrediction_.predictMergedUKF(state2,cov,0.0,measMap_.begin(),measMap_.size());
  PredictionExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
