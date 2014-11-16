#include "TestClasses.hpp"
#include "gtest/gtest.h"
#include <assert.h>

using namespace LWFTest;

typedef ::testing::Types<
    NonlinearTest,
    LinearTest
> TestClasses;

// The fixture for testing class PredictionModel
template<typename TestClass>
class PredictionModelTest : public ::testing::Test, public TestClass {
 protected:
  PredictionModelTest() {
    this->init(this->testState_,this->testUpdateMeas_,this->testPredictionMeas_);
    this->measMap_[0.1] = this->testPredictionMeas_;
    this->measMap_[0.2] = this->testPredictionMeas_;
    this->measMap_[0.4] = this->testPredictionMeas_;
  }
  virtual ~PredictionModelTest() {
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
  mtPredictionExample testPrediction_;
  mtState testState_;
  mtPredictionMeas testPredictionMeas_;
  mtUpdateMeas testUpdateMeas_;
  const double dt_ = 0.1;
  std::map<double,mtPredictionMeas> measMap_;
};

TYPED_TEST_CASE(PredictionModelTest, TestClasses);

// Test constructors
TYPED_TEST(PredictionModelTest, constructors) {
  typename TestFixture::mtPredictionExample testPrediction;
  ASSERT_EQ(testPrediction.mode_,LWF::PredictionEKF);
  ASSERT_EQ((testPrediction.prenoiP_-TestFixture::mtPredictionExample::mtNoise::mtCovMat::Identity()*0.0001).norm(),0.0);
  typename TestFixture::mtPredictionExample::mtNoise::mtDifVec dif;
  typename TestFixture::mtPredictionExample::mtNoise noise;
  noise.setIdentity();
  testPrediction.stateSigmaPointsNoi_.getMean().boxMinus(noise,dif);
  ASSERT_NEAR(dif.norm(),0.0,1e-6);
  ASSERT_NEAR((testPrediction.prenoiP_-testPrediction.stateSigmaPointsNoi_.getCovarianceMatrix()).norm(),0.0,1e-8);
}

// Test finite difference Jacobians
TYPED_TEST(PredictionModelTest, FDjacobians) {
  typename TestFixture::mtPredictionExample::mtJacInput F = this->testPrediction_.jacInputFD(this->testState_,this->testPredictionMeas_,this->dt_,0.0000001);
  typename TestFixture::mtPredictionExample::mtJacNoise Fn = this->testPrediction_.jacNoiseFD(this->testState_,this->testPredictionMeas_,this->dt_,0.0000001);
  switch(TestFixture::id_){
    case 0:
      ASSERT_NEAR((F-this->testPrediction_.jacInput(this->testState_,this->testPredictionMeas_,this->dt_)).norm(),0.0,1e-5);
      ASSERT_NEAR((Fn-this->testPrediction_.jacNoise(this->testState_,this->testPredictionMeas_,this->dt_)).norm(),0.0,1e-5);
      break;
    case 1:
      ASSERT_NEAR((F-this->testPrediction_.jacInput(this->testState_,this->testPredictionMeas_,this->dt_)).norm(),0.0,1e-8);
      ASSERT_NEAR((Fn-this->testPrediction_.jacNoise(this->testState_,this->testPredictionMeas_,this->dt_)).norm(),0.0,1e-8);
      break;
    default:
      ASSERT_NEAR((F-this->testPrediction_.jacInput(this->testState_,this->testPredictionMeas_,this->dt_)).norm(),0.0,1e-5);
      ASSERT_NEAR((Fn-this->testPrediction_.jacNoise(this->testState_,this->testPredictionMeas_,this->dt_)).norm(),0.0,1e-5);
  };
}

// Test predictEKF
TYPED_TEST(PredictionModelTest, predictEKF) {
  typename TestFixture::mtPredictionExample::mtState::mtCovMat cov;
  cov.setIdentity();
  typename TestFixture::mtPredictionExample::mtJacInput F = this->testPrediction_.jacInput(this->testState_,this->testPredictionMeas_,this->dt_);
  typename TestFixture::mtPredictionExample::mtJacNoise Fn = this->testPrediction_.jacNoise(this->testState_,this->testPredictionMeas_,this->dt_);
  typename TestFixture::mtPredictionExample::mtState::mtCovMat predictedCov = F*cov*F.transpose() + Fn*this->testPrediction_.prenoiP_*Fn.transpose();
  typename TestFixture::mtPredictionExample::mtState state;
  state = this->testState_;
  this->testPrediction_.predictEKF(state,cov,this->testPredictionMeas_,this->dt_);
  typename TestFixture::mtPredictionExample::mtState::mtDifVec dif;
  state.boxMinus(this->testPrediction_.eval(this->testState_,this->testPredictionMeas_,this->dt_),dif);
  switch(TestFixture::id_){
    case 0:
      ASSERT_NEAR(dif.norm(),0.0,1e-6);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
      break;
    case 1:
      ASSERT_NEAR(dif.norm(),0.0,1e-10);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-10);
      break;
    default:
      ASSERT_NEAR(dif.norm(),0.0,1e-6);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
  };
}

// Test comparePredict
TYPED_TEST(PredictionModelTest, comparePredict) {
  typename TestFixture::mtPredictionExample::mtState::mtCovMat cov1 = TestFixture::mtPredictionExample::mtState::mtCovMat::Identity()*0.000001;
  typename TestFixture::mtPredictionExample::mtState::mtCovMat cov2 = cov1;
  typename TestFixture::mtPredictionExample::mtState state1 = this->testState_;
  typename TestFixture::mtPredictionExample::mtState state2 = this->testState_;
  this->testPrediction_.predictEKF(state1,cov1,this->testPredictionMeas_,this->dt_);
  this->testPrediction_.predictUKF(state2,cov2,this->testPredictionMeas_,this->dt_);
  typename TestFixture::mtPredictionExample::mtState::mtDifVec dif;
  state1.boxMinus(state2,dif);
  switch(TestFixture::id_){
    case 0:
      ASSERT_NEAR(dif.norm(),0.0,1e-5);
      ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6);
      break;
    case 1:
      ASSERT_NEAR(dif.norm(),0.0,1e-9);
      ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-10);
      break;
    default:
      ASSERT_NEAR(dif.norm(),0.0,1e-5);
      ASSERT_NEAR((cov1-cov2).norm(),0.0,1e-6);
  };
}

// Test predictMergedEKF
TYPED_TEST(PredictionModelTest, predictMergedEKF) {
  typename TestFixture::mtPredictionExample::mtState::mtCovMat cov;
  cov.setIdentity();
  double t = 0;
  double dt = this->measMap_.rbegin()->first-t;

  typename TestFixture::mtPredictionExample::mtMeas meanMeas;
  typename TestFixture::mtPredictionExample::mtMeas::mtDifVec vec;
  typename TestFixture::mtPredictionExample::mtMeas::mtDifVec difVec;
  vec.setZero();
  for(typename std::map<double,typename TestFixture::mtPredictionExample::mtMeas>::iterator it = next(this->measMap_.begin());it != this->measMap_.end();it++){
    this->measMap_.begin()->second.boxMinus(it->second,difVec);
    vec = vec + difVec;
  }
  vec = vec/this->measMap_.size();
  this->measMap_.begin()->second.boxPlus(vec,meanMeas);

  typename TestFixture::mtPredictionExample::mtJacInput F = this->testPrediction_.jacInput(this->testState_,meanMeas,dt);
  typename TestFixture::mtPredictionExample::mtJacNoise Fn = this->testPrediction_.jacNoise(this->testState_,meanMeas,dt);
  typename TestFixture::mtPredictionExample::mtState::mtCovMat predictedCov = F*cov*F.transpose() + Fn*this->testPrediction_.prenoiP_*Fn.transpose();
  typename TestFixture::mtPredictionExample::mtState state1;
  state1 = this->testState_;
  this->testPrediction_.predictMergedEKF(state1,cov,0.0,this->measMap_.begin(),this->measMap_.size());
  typename TestFixture::mtPredictionExample::mtState state2;
  state2 = this->testState_;
  for(typename std::map<double,typename TestFixture::mtPredictionExample::mtMeas>::iterator it = this->measMap_.begin();it != this->measMap_.end();it++){
    state2 = this->testPrediction_.eval(state2,it->second,it->first-t);
    t = it->first;
  }
  typename TestFixture::mtPredictionExample::mtState::mtDifVec dif;
  state1.boxMinus(state2,dif);
  switch(TestFixture::id_){
    case 0:
      ASSERT_NEAR(dif.norm(),0.0,1e-6);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
      break;
    case 1:
      ASSERT_NEAR(dif.norm(),0.0,1e-10);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-10);
      break;
    default:
      ASSERT_NEAR(dif.norm(),0.0,1e-6);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
  };
}

// Test predictMergedUKF
TYPED_TEST(PredictionModelTest, predictMergedUKF) {
  typename TestFixture::mtPredictionExample::mtState::mtCovMat cov;
  cov.setIdentity();
  double t = 0;
  double dt = this->measMap_.rbegin()->first-t;

  typename TestFixture::mtPredictionExample::mtMeas meanMeas;
  typename TestFixture::mtPredictionExample::mtMeas::mtDifVec vec;
  typename TestFixture::mtPredictionExample::mtMeas::mtDifVec difVec;
  vec.setZero();
  for(typename std::map<double,typename TestFixture::mtPredictionExample::mtMeas>::iterator it = next(this->measMap_.begin());it != this->measMap_.end();it++){
    this->measMap_.begin()->second.boxMinus(it->second,difVec);
    vec = vec + difVec;
  }
  vec = vec/this->measMap_.size();
  this->measMap_.begin()->second.boxPlus(vec,meanMeas);

  this->testPrediction_.stateSigmaPoints_.computeFromGaussian(this->testState_,cov);
  for(unsigned int i=0;i<this->testPrediction_.stateSigmaPoints_.L_;i++){
    this->testPrediction_.stateSigmaPointsPre_(i) = this->testPrediction_.eval(this->testPrediction_.stateSigmaPoints_(i),meanMeas,this->testPrediction_.stateSigmaPointsNoi_(i),dt);
  }
  typename TestFixture::mtPredictionExample::mtState state1 = this->testPrediction_.stateSigmaPointsPre_.getMean();
  typename TestFixture::mtPredictionExample::mtState::mtCovMat predictedCov = this->testPrediction_.stateSigmaPointsPre_.getCovarianceMatrix(state1);
  typename TestFixture::mtPredictionExample::mtState state2;
  state2 = this->testState_;
  this->testPrediction_.predictMergedUKF(state2,cov,0.0,this->measMap_.begin(),this->measMap_.size());
  typename TestFixture::mtPredictionExample::mtState::mtDifVec dif;
  state1.boxMinus(state2,dif);
  switch(TestFixture::id_){
    case 0:
      ASSERT_NEAR(dif.norm(),0.0,1e-6);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
      break;
    case 1:
      ASSERT_NEAR(dif.norm(),0.0,1e-10);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-10);
      break;
    default:
      ASSERT_NEAR(dif.norm(),0.0,1e-6);
      ASSERT_NEAR((cov-predictedCov).norm(),0.0,1e-6);
  };
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
