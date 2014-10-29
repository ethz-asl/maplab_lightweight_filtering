#include "TestClasses.hpp"
#include "gtest/gtest.h"
#include <assert.h>

using namespace LWFTest;

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
  typename UpdateExample::mtNoise::mtDiffVec dif;
  testUpdate.stateSigmaPointsNoi_.getMean().boxMinus(typename UpdateExample::mtNoise(),dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((testUpdate.updnoiP_-testUpdate.stateSigmaPointsNoi_.getCovarianceMatrix()).norm(),0.0,1e-8);
  PredictAndUpdateExample testPredictAndUpdate;
  ASSERT_EQ((testPredictAndUpdate.updnoiP_-PredictAndUpdateExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
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

// Test updateEKFWithOutlier
TEST_F(UpdateModelTest, updateEKFWithOutlier) {
  testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<UpdateExample::mtInnovation>(0,2,7.21));
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
  Py.block(0,0,6,3).setZero();
  Py.block(0,0,3,6).setZero();
  Py.block(0,0,3,3).setIdentity();
  H.block(0,0,3,UpdateExample::mtState::D_).setZero();
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

  // Test with outlier detection
  cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  state1 = testState_;
  state2 = testState_;
  testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<UpdateExample::mtInnovation>(0,2,7.21));
  testUpdate_.updateEKF(state1,cov1,testUpdateMeas_);
  testUpdate_.updateUKF(state2,cov2,testUpdateMeas_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-5);
}

// Test predictAndUpdateEKF
TEST_F(UpdateModelTest, predictAndUpdateEKF) {
  UpdateExample::mtState::mtCovMat cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState::mtCovMat cov2 = cov1;
  UpdateExample::mtState state1 = testState_;
  UpdateExample::mtState state2 = testState_;
  testPrediction_.predictEKF(state1,cov1,testPredictionMeas_,dt_);
  testUpdate_.updateEKF(state1,cov1,testUpdateMeas_);
  testPredictAndUpdate_.predictAndUpdateEKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  UpdateExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-8);
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-8);

  // With outlier
  testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<UpdateExample::mtInnovation>(0,2,7.21));
  testPredictAndUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<UpdateExample::mtInnovation>(0,2,7.21));
  cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = cov1;
  state1 = testState_;
  state2 = testState_;
  testPrediction_.predictEKF(state1,cov1,testPredictionMeas_,dt_);
  testUpdate_.updateEKF(state1,cov1,testUpdateMeas_);
  testPredictAndUpdate_.predictAndUpdateEKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-8);
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-8);
}

// Test predictAndUpdateUKF
TEST_F(UpdateModelTest, predictAndUpdateUKF) {
  UpdateExample::mtState::mtCovMat cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState::mtCovMat cov2 = cov1;
  UpdateExample::mtState state1 = testState_;
  UpdateExample::mtState state2 = testState_;
  testPrediction_.predictUKF(state1,cov1,testPredictionMeas_,dt_);
  testUpdate_.updateUKF(state1,cov1,testUpdateMeas_);
  testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  UpdateExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-4); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance

  // With outlier
  testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<UpdateExample::mtInnovation>(0,2,7.21));
  testPredictAndUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<UpdateExample::mtInnovation>(0,2,7.21));
  cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = cov1;
  state1 = testState_;
  state2 = testState_;
  testPrediction_.predictUKF(state1,cov1,testPredictionMeas_,dt_);
  testUpdate_.updateUKF(state1,cov1,testUpdateMeas_);
  testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-4); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance

  // NB: Also tested with hack in update, difference comes because of different sigma points
}

// Test comparePredictAndUpdate (including correlated noise)
TEST_F(UpdateModelTest, comparePredictAndUpdate) {
  UpdateExample::mtState::mtCovMat cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  UpdateExample::mtState::mtCovMat cov2 = cov1;
  UpdateExample::mtState state1 = testState_;
  UpdateExample::mtState state2 = testState_;
  testPredictAndUpdate_.predictAndUpdateEKF(state1,cov1,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  UpdateExample::mtState::mtDiffVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-3); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-4); // Careful, will differ depending on the magnitude of the covariance

  // Negativ Control (Based on above)
  cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = cov1;
  state1 = testState_;
  state2 = testState_;
  testPredictAndUpdate_.predictAndUpdateEKF(state1,cov1,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  testPredictAndUpdate_.preupdnoiP_.block(6,3,3,3) = Eigen::Matrix3d::Identity()*0.00005;
  testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  state1.boxMinus(state2,dif);
  ASSERT_TRUE(dif.norm()>1e-3);
  ASSERT_TRUE((cov1-cov2).norm()>1e-4);


  testPredictAndUpdate_.preupdnoiP_.block(6,3,3,3) = Eigen::Matrix3d::Identity()*0.00005;
  cov1 = UpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = cov1;
  state1 = testState_;
  state2 = testState_;
  testPredictAndUpdate_.predictAndUpdateEKF(state1,cov1,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,testUpdateMeas_,testPrediction_,testPredictionMeas_,dt_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-3); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-4); // Careful, will differ depending on the magnitude of the covariance
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
