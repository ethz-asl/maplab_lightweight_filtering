#include "TestClasses.hpp"
#include "FilterBase.hpp"
#include "gtest/gtest.h"
#include <assert.h>

using namespace LWFTest;

typedef ::testing::Types<
    NonlinearTest,
    LinearTest
> TestClasses;

// The fixture for testing class MeasurementTimeline
class MeasurementTimelineTest : public ::testing::Test {
 protected:
  MeasurementTimelineTest() {
    for(unsigned int i=0;i<N_;i++){
      times_[i] = i*i*0.1+i*0.3;
      values_[i] = (i*345665)%10+i*0.34;
    }
    timeline_.maxWaitTime_ = 1.0;
  }
  virtual ~MeasurementTimelineTest() {
  }
  LWF::MeasurementTimeline<double> timeline_;
  static const unsigned int N_ = 5;
  double times_[N_];
  double values_[N_];
};

// Test constructors
TEST_F(MeasurementTimelineTest, constructors) {
  LWF::MeasurementTimeline<double> timeline;
  ASSERT_EQ(timeline.maxWaitTime_,0.1);
  ASSERT_EQ(timeline.safeWarningTime_,0.0);
  ASSERT_EQ(timeline.frontWarningTime_,0.0);
  ASSERT_EQ(timeline.gotFrontWarning_,false);
}

// Test addMeas
TEST_F(MeasurementTimelineTest, addMeas) {
  timeline_.frontWarningTime_ = times_[N_-2];
  for(int i=N_-1;i>=0;i--){ // Reverse for detecting FrontWarning
    timeline_.addMeas(values_[i],times_[i]);
    for(int j=N_-1;j>=i;j--){
      ASSERT_EQ(timeline_.measMap_.at(times_[j]),values_[j]);
    }
    ASSERT_EQ(timeline_.gotFrontWarning_,i<=(N_-2));
  }
}

// Test clean
TEST_F(MeasurementTimelineTest, clean) {
  for(unsigned int i=0;i<N_;i++){
    timeline_.addMeas(values_[i],times_[i]);
  }
  timeline_.clean(times_[N_-2]);
  ASSERT_EQ(timeline_.measMap_.size(),1);
  ASSERT_EQ(timeline_.measMap_.at(times_[N_-1]),values_[N_-1]);
}

// Test getNextTime
TEST_F(MeasurementTimelineTest, getNextTime) {
  for(unsigned int i=0;i<N_;i++){
    timeline_.addMeas(values_[i],times_[i]);
  }
  bool r;
  double nextTime;
  r = timeline_.getNextTime(times_[0]-1e-6,nextTime);
  ASSERT_EQ(r,true);
  ASSERT_EQ(nextTime,times_[0]);
  for(unsigned int i=0;i<N_-1;i++){
    r = timeline_.getNextTime(0.5*(times_[i]+times_[i+1]),nextTime);
    ASSERT_EQ(r,true);
    ASSERT_EQ(nextTime,times_[i+1]);
  }
  r = timeline_.getNextTime(times_[N_-1]+1e-6,nextTime);
  ASSERT_EQ(r,false);
}

// Test waitTime
TEST_F(MeasurementTimelineTest, waitTime) {
  for(unsigned int i=0;i<N_;i++){
    timeline_.addMeas(values_[i],times_[i]);
  }
  double time;
  for(unsigned int i=0;i<N_;i++){
    time = times_[i];
    timeline_.waitTime(times_[i],time);
    ASSERT_EQ(time,times_[i]);
  }
  time = times_[N_-1]+2.0;
  timeline_.waitTime(time,time);
  ASSERT_EQ(time,times_[N_-1]+1.0);
  time = times_[N_-1]+0.5;
  timeline_.waitTime(time,time);
  ASSERT_EQ(time,times_[N_-1]);
  time = times_[N_-1]-0.5;
  timeline_.waitTime(time,time);
  ASSERT_EQ(time,times_[N_-1]-0.5);
}

// Test getLastTime
TEST_F(MeasurementTimelineTest, getLastTime) {
  bool r;
  double lastTime;
  r = timeline_.getLastTime(lastTime);
  ASSERT_EQ(r,false);
  for(unsigned int i=0;i<N_;i++){
    timeline_.addMeas(values_[i],times_[i]);
  }
  r = timeline_.getLastTime(lastTime);
  ASSERT_EQ(r,true);
  ASSERT_EQ(lastTime,times_[N_-1]);
}

// Test hasMeasurementAt
TEST_F(MeasurementTimelineTest, hasMeasurementAt) {
  for(unsigned int i=0;i<N_;i++){
    timeline_.addMeas(values_[i],times_[i]);
  }
  bool r;
  for(unsigned int i=0;i<N_;i++){
    r = timeline_.hasMeasurementAt(times_[i]);
    ASSERT_EQ(r,true);
  }
  r = timeline_.hasMeasurementAt(0.5*(times_[0]+times_[1]));
  ASSERT_EQ(r,false);
}

// The fixture for testing class FilterBase
template<typename TestClass>
class FilterBaseTest : public ::testing::Test, public TestClass {
 protected:
  FilterBaseTest() {
    this->init(this->testState_,this->testUpdateMeas_,this->testPredictionMeas_);
    this->predictionManager_.maxWaitTime_ = 1.0;
    this->predictionManager2_.maxWaitTime_ = 1.0;
    this->updateManager_.maxWaitTime_ = 1.0;
    this->updateManager2_.maxWaitTime_ = 1.0;
    this->predictionUpdateManager_.maxWaitTime_ = 0.0;
    this->predictionUpdateManager2_.maxWaitTime_ = 0.0;
    this->testFilter_.registerPredictionManager(this->predictionManager_,"Prediction");
    this->testFilter_.registerUpdateManager(this->updateManager_,"Update1");
    this->testFilter_.registerUpdateAndPredictManager(this->predictionUpdateManager_,"Update2");
    this->testFilter2_.registerPredictionManager(this->predictionManager2_,"Prediction");
    this->testFilter2_.registerUpdateManager(this->updateManager2_,"Update1");
    this->testFilter2_.registerUpdateAndPredictManager(this->predictionUpdateManager2_,"Update2");
    switch(id_){
      case 0:
        this->testFilter_.readFromInfo("test_nonlinear.info");
        break;
      case 1:
        this->testFilter_.readFromInfo("test_linear.info");
        break;
      default:
        this->testFilter_.readFromInfo("test_nonlinear.info");
    };
  }
  virtual ~FilterBaseTest() {
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
  using TestClass::id_;
  LWF::FilterBase<mtPredictionExample> testFilter_;
  LWF::FilterBase<mtPredictionExample> testFilter2_;
  mtState testState_;
  typename TestClass::mtState::mtCovMat testCov_;
  typename TestClass::mtState::mtDifVec difVec_;
  mtUpdateMeas testUpdateMeas_;
  mtPredictionMeas testPredictionMeas_;
  LWF::PredictionManager<mtPredictionExample> predictionManager_;
  LWF::UpdateManager<mtUpdateExample> updateManager_;
  LWF::UpdateAndPredictManager<mtPredictAndUpdateExample,mtPredictionExample> predictionUpdateManager_;
  LWF::PredictionManager<mtPredictionExample> predictionManager2_;
  LWF::UpdateManager<mtUpdateExample> updateManager2_;
  LWF::UpdateAndPredictManager<mtPredictAndUpdateExample,mtPredictionExample> predictionUpdateManager2_;
  const double dt_ = 0.1;
};

TYPED_TEST_CASE(FilterBaseTest, TestClasses);

// Test constructors
TYPED_TEST(FilterBaseTest, constructors) {
  LWF::FilterBase<typename TestFixture::mtPredictionExample> testFilter;
  LWF::UpdateManager<typename TestFixture::mtUpdateExample> updateManager;
  ASSERT_EQ(updateManager.coupledToPrediction_,false);
  ASSERT_EQ(updateManager.filteringMode_,LWF::UpdateEKF);
  LWF::UpdateAndPredictManager<typename TestFixture::mtPredictAndUpdateExample,typename TestFixture::mtPredictionExample> predictionUpdateManager;
  ASSERT_EQ(predictionUpdateManager.coupledToPrediction_,true);
  ASSERT_EQ(predictionUpdateManager.filteringMode_,LWF::UpdateEKF);
  LWF::PredictionManager<typename TestFixture::mtPredictionExample> predictionManager;
  ASSERT_EQ(predictionManager.filteringMode_,LWF::PredictionEKF);
}

// Test propertyHandler
TYPED_TEST(FilterBaseTest, propertyHandler) {
  // Generate parameters
  typename TestFixture::mtPredictionExample::mtNoise::mtCovMat prenoiP;
  prenoiP.setZero();
  for(unsigned int i=0;i<TestFixture::mtPredictionExample::mtNoise::D_;i++){
    prenoiP(i,i) = 0.1*i*i+3.4*i+1.2;
  }
  typename TestFixture::mtUpdateExample::mtNoise::mtCovMat updnoiP;
  updnoiP.setZero();
  for(unsigned int i=0;i<TestFixture::mtUpdateExample::mtNoise::D_;i++){
    updnoiP(i,i) = -0.1*i*i+3.2*i+1.1;
  }
  typename TestFixture::mtPredictAndUpdateExample::mtNoise::mtCovMat updnoiP2;
  updnoiP2.setZero();
  for(unsigned int i=0;i<TestFixture::mtPredictAndUpdateExample::mtNoise::D_;i++){
    updnoiP2(i,i) = -0.2*i*i+1.2*i+0.1;
  }
  Eigen::Matrix<double,TestFixture::mtPredictAndUpdateExample::mtPredictionNoise::D_,TestFixture::mtPredictAndUpdateExample::mtNoise::D_> preupdnoiP;
  preupdnoiP.setZero();
  for(unsigned int i=0;i<TestFixture::mtPredictAndUpdateExample::mtPredictionNoise::D_;i++){
    for(unsigned int j=0;j<TestFixture::mtPredictAndUpdateExample::mtNoise::D_;j++){
      preupdnoiP(i,j) = 0.3*i*i+0.2*i+3.1;
    }
  }
  typename TestFixture::mtState::mtCovMat initP;
  initP.setZero();
  for(unsigned int i=0;i<TestFixture::mtState::D_;i++){
    initP(i,i) = 0.5*i*i+3.1*i+3.2;
  }
  typename TestFixture::mtState initState;
  initState.setRandom(1);
  this->predictionManager_.prediction_.prenoiP_ = prenoiP;
  this->updateManager_.update_.updnoiP_ = updnoiP;
  this->predictionUpdateManager_.update_.updnoiP_ = updnoiP2;
  this->predictionUpdateManager_.update_.preupdnoiP_ = preupdnoiP;
  this->testFilter_.init_.cov_ = initP;
  this->testFilter_.init_.state_ = initState;

  // Write to file
  this->testFilter_.writeToInfo("test.info");

  // Set parameters zero
  this->predictionManager_.prediction_.prenoiP_.setZero();
  this->updateManager_.update_.updnoiP_.setZero();
  this->predictionUpdateManager_.update_.updnoiP_.setZero();
  this->predictionUpdateManager_.update_.preupdnoiP_.setZero();
  this->testFilter_.init_.cov_.setZero();
  this->testFilter_.init_.state_.setIdentity();

  // Read parameters from file and compare
  this->testFilter_.readFromInfo("test.info");
  ASSERT_NEAR((this->predictionManager_.prediction_.prenoiP_-prenoiP).norm(),0.0,1e-6);
  ASSERT_NEAR((this->updateManager_.update_.updnoiP_-updnoiP).norm(),0.0,1e-6);
  ASSERT_NEAR((this->predictionUpdateManager_.update_.updnoiP_-updnoiP2).norm(),0.0,1e-6);
  ASSERT_NEAR((this->predictionUpdateManager_.update_.preupdnoiP_-preupdnoiP).norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter_.init_.cov_-initP).norm(),0.0,1e-6);
  typename TestFixture::mtState::mtDifVec difVec;
  this->testFilter_.init_.state_.boxMinus(initState,difVec);
  ASSERT_NEAR(difVec.norm(),0.0,1e-6);
}

// Test updateSafe (Only for 1 update type (wait time set to zero for the other)), co-test getSafeTime() and setSafeWarningTime() and clean()
TYPED_TEST(FilterBaseTest, updateSafe) {
  double safeTime = 0.0;
  this->testFilter_.setSafeWarningTime(0.1); // makes warning -> check

  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager_.addMeas(this->testUpdateMeas_,0.1);
  ASSERT_TRUE(this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.1);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  ASSERT_EQ(this->predictionManager_.measMap_.size(),0);
  ASSERT_EQ(this->updateManager_.measMap_.size(),0);

  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1); // makes warning -> check
  this->updateManager_.addMeas(this->testUpdateMeas_,0.1);
  ASSERT_TRUE(!this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.1);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  ASSERT_EQ(this->predictionManager_.measMap_.size(),1);
  ASSERT_EQ(this->updateManager_.measMap_.size(),1);

  this->updateManager_.addMeas(this->testUpdateMeas_,0.2);
  ASSERT_TRUE(!this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.1);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  ASSERT_EQ(this->predictionManager_.measMap_.size(),1);
  ASSERT_EQ(this->updateManager_.measMap_.size(),2);

  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
  ASSERT_TRUE(this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.2);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.2);
  ASSERT_EQ(this->predictionManager_.measMap_.size(),1);
  ASSERT_EQ(this->updateManager_.measMap_.size(),0);

  this->updateManager_.addMeas(this->testUpdateMeas_,0.3);
  ASSERT_TRUE(this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.3);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.3);
  ASSERT_EQ(this->predictionManager_.measMap_.size(),0);
  ASSERT_EQ(this->updateManager_.measMap_.size(),0);
}

// Test updateFront
TYPED_TEST(FilterBaseTest, updateFront) {
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager_.addMeas(this->testUpdateMeas_,0.1);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==false);
  this->testFilter_.updateFront(0.5);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==false);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  ASSERT_EQ(this->testFilter_.front_.t_,0.5);

  this->updateManager_.addMeas(this->testUpdateMeas_,0.2);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==true);
  this->testFilter_.updateFront(0.2);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==false);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  ASSERT_EQ(this->testFilter_.front_.t_,0.2);

  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==true);
  this->testFilter_.updateFront(0.3);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==false);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.2);
  ASSERT_EQ(this->testFilter_.front_.t_,0.3);

  this->updateManager_.addMeas(this->testUpdateMeas_,0.3);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==true);
  this->testFilter_.updateFront(0.3);
  ASSERT_TRUE(this->testFilter_.checkFrontWarning()==false);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.3);
  ASSERT_EQ(this->testFilter_.front_.t_,0.3);
}

// Test high level logic
TYPED_TEST(FilterBaseTest, highlevel) {
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager_.addMeas(this->testUpdateMeas_,0.1);
  this->testFilter_.updateSafe();
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager2_.addMeas(this->testUpdateMeas_,0.1);
  this->testFilter2_.updateSafe();

  this->testFilter2_.safe_.state_.boxMinus(this->testFilter_.safe_.state_,this->difVec_);
  ASSERT_EQ(this->testFilter_.safe_.t_,this->testFilter2_.safe_.t_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter2_.safe_.cov_-this->testFilter_.safe_.cov_).norm(),0.0,1e-6);


  this->updateManager_.addMeas(this->testUpdateMeas_,0.2);
  this->testFilter_.updateFront(0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
  this->testFilter_.updateFront(0.3);
  this->updateManager_.addMeas(this->testUpdateMeas_,0.3);
  this->testFilter_.updateFront(0.3);

  this->updateManager2_.addMeas(this->testUpdateMeas_,0.2);
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.3);
  this->updateManager2_.addMeas(this->testUpdateMeas_,0.3);
  this->testFilter2_.updateSafe();

  this->testFilter2_.safe_.state_.boxMinus(this->testFilter_.safe_.state_,this->difVec_);
  ASSERT_EQ(this->testFilter_.safe_.t_,this->testFilter2_.safe_.t_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter2_.safe_.cov_-this->testFilter_.safe_.cov_).norm(),0.0,1e-6);
}

// Test high level logic 2: coupled
TYPED_TEST(FilterBaseTest, highlevel2) {
  this->predictionUpdateManager_.update_.preupdnoiP_.block(0,0,3,3) = Eigen::Matrix3d::Identity()*0.00009;
  this->predictionUpdateManager2_.update_.preupdnoiP_.block(0,0,3,3) = Eigen::Matrix3d::Identity()*0.00009;
  this->testState_ = this->testFilter_.safe_.state_;
  this->testCov_ = this->testFilter_.safe_.cov_;
  this->predictionUpdateManager_.maxWaitTime_ = 1.0;
  this->predictionUpdateManager2_.maxWaitTime_ = 1.0;
  this->updateManager_.maxWaitTime_ = 0.0;
  this->updateManager2_.maxWaitTime_ = 0.0;
  // TestFilter
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
  this->predictionUpdateManager_.addMeas(this->testUpdateMeas_,0.1);
  this->testFilter_.updateSafe();
  // TestFilter2
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.1);
  this->predictionUpdateManager2_.addMeas(this->testUpdateMeas_,0.1);
  this->testFilter2_.updateSafe();
  // Direct
  this->predictionUpdateManager_.update_.predictAndUpdateEKF(this->testState_,this->testCov_,this->testUpdateMeas_,this->predictionManager_.prediction_,this->testPredictionMeas_,0.1);

  // Compare
  this->testFilter2_.safe_.state_.boxMinus(this->testFilter_.safe_.state_,this->difVec_);
  ASSERT_EQ(this->testFilter_.safe_.t_,this->testFilter2_.safe_.t_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter2_.safe_.cov_-this->testFilter_.safe_.cov_).norm(),0.0,1e-6);
  this->testFilter_.safe_.state_.boxMinus(this->testState_,this->difVec_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter_.safe_.cov_-this->testCov_).norm(),0.0,1e-6);

  // TestFilter
  this->predictionUpdateManager_.addMeas(this->testUpdateMeas_,0.2);
  this->testFilter_.updateFront(0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
  this->testFilter_.updateFront(0.3);
  this->predictionUpdateManager_.addMeas(this->testUpdateMeas_,0.3);
  this->testFilter_.updateFront(0.3);
  // TestFilter2
  this->predictionUpdateManager2_.addMeas(this->testUpdateMeas_,0.2);
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.3);
  this->predictionUpdateManager2_.addMeas(this->testUpdateMeas_,0.3);
  this->testFilter2_.updateSafe();
  // Direct
  this->predictionUpdateManager_.update_.predictAndUpdateEKF(this->testState_,this->testCov_,this->testUpdateMeas_,this->predictionManager_.prediction_,this->testPredictionMeas_,0.1);
  this->predictionUpdateManager_.update_.predictAndUpdateEKF(this->testState_,this->testCov_,this->testUpdateMeas_,this->predictionManager_.prediction_,this->testPredictionMeas_,0.1);

  // Compare
  this->testFilter2_.safe_.state_.boxMinus(this->testFilter_.safe_.state_,this->difVec_);
  ASSERT_EQ(this->testFilter_.safe_.t_,this->testFilter2_.safe_.t_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter2_.safe_.cov_-this->testFilter_.safe_.cov_).norm(),0.0,1e-6);
  this->testFilter_.safe_.state_.boxMinus(this->testState_,this->difVec_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter_.safe_.cov_-this->testCov_).norm(),0.0,1e-6);
}

// Test high level logic 3: merged
TYPED_TEST(FilterBaseTest, highlevel3) {
  this->testState_ = this->testFilter_.safe_.state_;
  this->testCov_ = this->testFilter_.safe_.cov_;
  // TestFilter and direct method
  this->predictionManager_.prediction_.mbMergePredictions_ = true;
  this->predictionManager2_.prediction_.mbMergePredictions_ = true;
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager_.addMeas(this->testUpdateMeas_,0.1);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.4);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.5);
  this->updateManager_.addMeas(this->testUpdateMeas_,0.5);
    this->predictionManager_.prediction_.predictEKF(this->testState_,this->testCov_,this->testPredictionMeas_,0.1);
    this->updateManager_.update_.updateEKF(this->testState_,this->testCov_,this->testUpdateMeas_);
    this->predictionManager_.prediction_.predictMergedEKF(this->testState_,this->testCov_,0.1,next(this->predictionManager_.measMap_.begin(),1),3);
    this->predictionManager_.prediction_.predictEKF(this->testState_,this->testCov_,this->testPredictionMeas_,0.1);
    this->updateManager_.update_.updateEKF(this->testState_,this->testCov_,this->testUpdateMeas_);
  this->testFilter_.updateSafe();
  // TestFilter2
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager2_.addMeas(this->testUpdateMeas_,0.1);
  this->testFilter2_.updateSafe();
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.2);
  this->testFilter2_.updateSafe();
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.3);
  this->testFilter2_.updateSafe();
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.4);
  this->testFilter2_.updateSafe();
  this->predictionManager2_.addMeas(this->testPredictionMeas_,0.5);
  this->testFilter2_.updateSafe();
  this->updateManager2_.addMeas(this->testUpdateMeas_,0.5);
  this->testFilter2_.updateSafe();

  // Compare
  this->testFilter2_.safe_.state_.boxMinus(this->testFilter_.safe_.state_,this->difVec_);
  ASSERT_EQ(this->testFilter_.safe_.t_,this->testFilter2_.safe_.t_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter2_.safe_.cov_-this->testFilter_.safe_.cov_).norm(),0.0,1e-6);
  this->testFilter_.safe_.state_.boxMinus(this->testState_,this->difVec_);
  ASSERT_NEAR(this->difVec_.norm(),0.0,1e-6);
  ASSERT_NEAR((this->testFilter_.safe_.cov_-this->testCov_).norm(),0.0,1e-6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
