#include "TestClasses.hpp"
#include "gtest/gtest.h"
#include <assert.h>

using namespace LWFTest;

typedef ::testing::Types<
    NonlinearTest
> TestClasses;

// The fixture for testing class UpdateModel
template<typename TestClass>
class UpdateModelTest : public ::testing::Test, public TestClass {
 public:
  UpdateModelTest() {
    this->testState_.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    this->testState_.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    this->testState_.v(2) = Eigen::Vector3d(0.1,2.9,7.3);
    this->testState_.v(3) = Eigen::Vector3d(2,4,-2.3);
    this->testState_.v(2) = Eigen::Vector3d(0.3,10.9,2.3); // TODO
    this->testState_.v(3) = Eigen::Vector3d(0.3,10.9,2.3);
    this->testState_.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    this->testUpdateMeas_.v(0) = Eigen::Vector3d(-1.5,12,5.23);
    this->testUpdateMeas_.q(0) = rot::RotationQuaternionPD(3.0/sqrt(15.0),-1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
    this->testPredictionMeas_.v(0) = Eigen::Vector3d(-5,2,17.3);
    this->testPredictionMeas_.v(1) = Eigen::Vector3d(15.7,0.45,-2.3);
  }
  virtual ~UpdateModelTest() {
  }
  using typename TestClass::mtState;
  using typename TestClass::mtUpdateMeas;
  using typename TestClass::mtUpdateNoise;
  using typename TestClass::mtInnovation;
  using typename TestClass::mtPredictionNoise;
  using typename TestClass::mtPredictionMeas;
  using typename TestClass::mtUpdateExample;
  using typename TestClass::mtPredictionExample;
  using typename TestClass::mtPredictAndUpdateExample;
  mtUpdateExample testUpdate_;
  mtPredictAndUpdateExample testPredictAndUpdate_;
  mtPredictionExample testPrediction_;
  mtState testState_;
  mtUpdateMeas testUpdateMeas_;
  mtPredictionMeas testPredictionMeas_;
  const double dt_ = 0.1;
};

TYPED_TEST_CASE(UpdateModelTest, TestClasses);

// Test constructors
TYPED_TEST(UpdateModelTest, constructors) {
  typename TestFixture::mtUpdateExample testUpdate;
  ASSERT_EQ((testUpdate.updnoiP_-TestFixture::mtUpdateExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
  typename TestFixture::mtUpdateExample::mtNoise::mtDifVec dif;
  testUpdate.stateSigmaPointsNoi_.getMean().boxMinus(typename TestFixture::mtUpdateExample::mtNoise(),dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((testUpdate.updnoiP_-testUpdate.stateSigmaPointsNoi_.getCovarianceMatrix()).norm(),0.0,1e-8);
  typename TestFixture::mtPredictAndUpdateExample testPredictAndUpdate;
  ASSERT_EQ((testPredictAndUpdate.updnoiP_-TestFixture::mtPredictAndUpdateExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
}

// Test finite difference Jacobians
TYPED_TEST(UpdateModelTest, FDjacobians) {
  typename TestFixture::mtUpdateExample::mtJacInput F = this->testUpdate_.jacInputFD(this->testState_,this->testUpdateMeas_,this->dt_,0.0000001);
  ASSERT_NEAR((F-this->testUpdate_.jacInput(this->testState_,this->testUpdateMeas_,this->dt_)).norm(),0.0,1e-5);
  typename TestFixture::mtUpdateExample::mtJacNoise Fn = this->testUpdate_.jacNoiseFD(this->testState_,this->testUpdateMeas_,this->dt_,0.0000001);
  ASSERT_NEAR((Fn-this->testUpdate_.jacNoise(this->testState_,this->testUpdateMeas_,this->dt_)).norm(),0.0,1e-5);
}

// Test updateEKF
TYPED_TEST(UpdateModelTest, updateEKF) {
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov;
  typename TestFixture::mtUpdateExample::mtState::mtCovMat updateCov;
  cov.setIdentity();
  typename TestFixture::mtUpdateExample::mtJacInput H = this->testUpdate_.jacInput(this->testState_,this->testUpdateMeas_,this->dt_);
  typename TestFixture::mtUpdateExample::mtJacNoise Hn = this->testUpdate_.jacNoise(this->testState_,this->testUpdateMeas_,this->dt_);

  typename TestFixture::mtUpdateExample::mtInnovation y = this->testUpdate_.eval(this->testState_,this->testUpdateMeas_);
  typename TestFixture::mtUpdateExample::mtInnovation yIdentity;
  yIdentity.setIdentity();
  typename TestFixture::mtUpdateExample::mtInnovation::mtDifVec innVector;

  typename TestFixture::mtUpdateExample::mtState state;
  typename TestFixture::mtUpdateExample::mtState stateUpdated;
  state = this->testState_;

  // Update
  typename TestFixture::mtUpdateExample::mtInnovation::mtCovMat Py = H*cov*H.transpose() + Hn*this->testUpdate_.updnoiP_*Hn.transpose();
  y.boxMinus(yIdentity,innVector);
  typename TestFixture::mtUpdateExample::mtInnovation::mtCovMat Pyinv = Py.inverse();

  // Kalman Update
  Eigen::Matrix<double,TestFixture::mtUpdateExample::mtState::D_,TestFixture::mtUpdateExample::mtInnovation::D_> K = cov*H.transpose()*Pyinv;
  updateCov = cov - K*Py*K.transpose();
  typename TestFixture::mtUpdateExample::mtState::mtDifVec updateVec;
  updateVec = -K*innVector;
  state.boxPlus(updateVec,stateUpdated);

  this->testUpdate_.updateEKF(state,cov,this->testUpdateMeas_);
  typename TestFixture::mtUpdateExample::mtState::mtDifVec dif;
  state.boxMinus(stateUpdated,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-updateCov).norm(),0.0,1e-6);
}

// Test updateEKFWithOutlier
TYPED_TEST(UpdateModelTest, updateEKFWithOutlier) {
  this->testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<typename TestFixture::mtUpdateExample::mtInnovation>(0,2,7.21));
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov;
  typename TestFixture::mtUpdateExample::mtState::mtCovMat updateCov;
  cov.setIdentity();
  typename TestFixture::mtUpdateExample::mtJacInput H = this->testUpdate_.jacInput(this->testState_,this->testUpdateMeas_,this->dt_);
  typename TestFixture::mtUpdateExample::mtJacNoise Hn = this->testUpdate_.jacNoise(this->testState_,this->testUpdateMeas_,this->dt_);

  typename TestFixture::mtUpdateExample::mtInnovation y = this->testUpdate_.eval(this->testState_,this->testUpdateMeas_);
  typename TestFixture::mtUpdateExample::mtInnovation yIdentity;
  yIdentity.setIdentity();
  typename TestFixture::mtUpdateExample::mtInnovation::mtDifVec innVector;

  typename TestFixture::mtUpdateExample::mtState state;
  typename TestFixture::mtUpdateExample::mtState stateUpdated;
  state = this->testState_;

  // Update
  typename TestFixture::mtUpdateExample::mtInnovation::mtCovMat Py = H*cov*H.transpose() + Hn*this->testUpdate_.updnoiP_*Hn.transpose();
  y.boxMinus(yIdentity,innVector);
  Py.block(0,0,6,3).setZero();
  Py.block(0,0,3,6).setZero();
  Py.block(0,0,3,3).setIdentity();
  H.block(0,0,3,TestFixture::mtUpdateExample::mtState::D_).setZero();
  typename TestFixture::mtUpdateExample::mtInnovation::mtCovMat Pyinv = Py.inverse();

  // Kalman Update
  Eigen::Matrix<double,TestFixture::mtUpdateExample::mtState::D_,TestFixture::mtUpdateExample::mtInnovation::D_> K = cov*H.transpose()*Pyinv;
  updateCov = cov - K*Py*K.transpose();
  typename TestFixture::mtUpdateExample::mtState::mtDifVec updateVec;
  updateVec = -K*innVector;
  state.boxPlus(updateVec,stateUpdated);

  this->testUpdate_.updateEKF(state,cov,this->testUpdateMeas_);
  typename TestFixture::mtUpdateExample::mtState::mtDifVec dif;
  state.boxMinus(stateUpdated,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov-updateCov).norm(),0.0,1e-6);
}

// Test compareUpdate
TYPED_TEST(UpdateModelTest, compareUpdate) {
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov2 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  typename TestFixture::mtUpdateExample::mtState state1 = this->testState_;
  typename TestFixture::mtUpdateExample::mtState state2 = this->testState_;
  this->testUpdate_.updateEKF(state1,cov1,this->testUpdateMeas_);
  this->testUpdate_.updateUKF(state2,cov2,this->testUpdateMeas_);
  typename TestFixture::mtUpdateExample::mtState::mtDifVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-5); // Careful, will differ depending on the magnitude of the covariance

  // Test with outlier detection
  cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  state1 = this->testState_;
  state2 = this->testState_;
  this->testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<typename TestFixture::mtUpdateExample::mtInnovation>(0,2,7.21));
  this->testUpdate_.updateEKF(state1,cov1,this->testUpdateMeas_);
  this->testUpdate_.updateUKF(state2,cov2,this->testUpdateMeas_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-5);
}

// Test predictAndUpdateEKF
TYPED_TEST(UpdateModelTest, predictAndUpdateEKF) {
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov2 = cov1;
  typename TestFixture::mtUpdateExample::mtState state1 = this->testState_;
  typename TestFixture::mtUpdateExample::mtState state2 = this->testState_;
  this->testPrediction_.predictEKF(state1,cov1,this->testPredictionMeas_,this->dt_);
  this->testUpdate_.updateEKF(state1,cov1,this->testUpdateMeas_);
  this->testPredictAndUpdate_.predictAndUpdateEKF(state2,cov2,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  typename TestFixture::mtUpdateExample::mtState::mtDifVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-8);
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-8);

  // With outlier
  this->testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<typename TestFixture::mtUpdateExample::mtInnovation>(0,2,7.21));
  this->testPredictAndUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<typename TestFixture::mtUpdateExample::mtInnovation>(0,2,7.21));
  cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = cov1;
  state1 = this->testState_;
  state2 = this->testState_;
  this->testPrediction_.predictEKF(state1,cov1,this->testPredictionMeas_,this->dt_);
  this->testUpdate_.updateEKF(state1,cov1,this->testUpdateMeas_);
  this->testPredictAndUpdate_.predictAndUpdateEKF(state2,cov2,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-8);
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-8);
}

// Test predictAndUpdateUKF
TYPED_TEST(UpdateModelTest, predictAndUpdateUKF) {
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov2 = cov1;
  typename TestFixture::mtUpdateExample::mtState state1 = this->testState_;
  typename TestFixture::mtUpdateExample::mtState state2 = this->testState_;
  this->testPrediction_.predictUKF(state1,cov1,this->testPredictionMeas_,this->dt_);
  this->testUpdate_.updateUKF(state1,cov1,this->testUpdateMeas_);
  this->testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  typename TestFixture::mtUpdateExample::mtState::mtDifVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-4); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance

  // With outlier
  this->testUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<typename TestFixture::mtUpdateExample::mtInnovation>(0,2,7.21));
  this->testPredictAndUpdate_.outlierDetectionVector_.push_back(LWF::UpdateOutlierDetection<typename TestFixture::mtUpdateExample::mtInnovation>(0,2,7.21));
  cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.000001;
  cov2 = cov1;
  state1 = this->testState_;
  state2 = this->testState_;
  this->testPrediction_.predictUKF(state1,cov1,this->testPredictionMeas_,this->dt_);
  this->testUpdate_.updateUKF(state1,cov1,this->testUpdateMeas_);
  this->testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-4); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6); // Careful, will differ depending on the magnitude of the covariance

  // NB: Also tested with hack in update, difference comes because of different sigma points
}

// Test comparePredictAndUpdate (including correlated noise)
TYPED_TEST(UpdateModelTest, comparePredictAndUpdate) {
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.0001;
  typename TestFixture::mtUpdateExample::mtState::mtCovMat cov2 = cov1;
  typename TestFixture::mtUpdateExample::mtState state1 = this->testState_;
  typename TestFixture::mtUpdateExample::mtState state2 = this->testState_;
  this->testPredictAndUpdate_.predictAndUpdateEKF(state1,cov1,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  this->testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  typename TestFixture::mtUpdateExample::mtState::mtDifVec dif;
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,2e-2); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,8e-5); // Careful, will differ depending on the magnitude of the covariance

  // Negativ Control (Based on above)
  cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.0001;
  cov2 = cov1;
  state1 = this->testState_;
  state2 = this->testState_;
  this->testPredictAndUpdate_.predictAndUpdateEKF(state1,cov1,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  this->testPredictAndUpdate_.preupdnoiP_.block(6,3,3,3) = Eigen::Matrix3d::Identity()*0.00009;
  this->testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  state1.boxMinus(state2,dif);
  ASSERT_TRUE(dif.norm()>1e-1); // TODO: retest with linear model
  ASSERT_TRUE((cov1-cov2).norm()>8e-5);


  this->testPredictAndUpdate_.preupdnoiP_.block(6,3,3,3) = Eigen::Matrix3d::Identity()*0.00009;
  cov1 = TestFixture::mtUpdateExample::mtState::mtCovMat::Identity()*0.0001;
  cov2 = cov1;
  state1 = this->testState_;
  state2 = this->testState_;
  this->testPredictAndUpdate_.predictAndUpdateEKF(state1,cov1,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  this->testPredictAndUpdate_.predictAndUpdateUKF(state2,cov2,this->testUpdateMeas_,this->testPrediction_,this->testPredictionMeas_,this->dt_);
  state1.boxMinus(state2,dif);
  ASSERT_NEAR(dif.norm(),0.0,2e-2); // Careful, will differ depending on the magnitude of the covariance
  ASSERT_NEAR((cov1-cov2).norm(),0.0,8e-5); // Careful, will differ depending on the magnitude of the covariance
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
