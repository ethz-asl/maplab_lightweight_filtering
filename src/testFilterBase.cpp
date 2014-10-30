#include "TestClasses.hpp"
#include "FilterBase.hpp"
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
    testUpdateMeas_.v(0) = Eigen::Vector3d(-1.5,12,15.23);
    testUpdateMeas_.q(0) = rot::RotationQuaternionPD(3.0/sqrt(15.0),-1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
    testPredictionMeas_.v(0) = Eigen::Vector3d(-5,2,17.3);
    testPredictionMeas_.v(1) = Eigen::Vector3d(15.7,0.45,-2.3);
    updateManager2_.maxWaitTime_ = 0.0;
    testFilter_.registerUpdateManager(updateManager1_,"Update1");
    testFilter_.registerUpdateManager(updateManager2_,"Update2");
    testFilter_.registerPredictionManager(predictionManager_,"Prediction");
    testFilter_.readFromInfo("test.info");
//    testFilter_.writeToInfo("test.info");
  }
  virtual ~UpdateModelTest() {
  }
  LWF::FilterBase<PredictionExample> testFilter_;
  State testState_;
  UpdateMeas testUpdateMeas_;
  PredictionMeas testPredictionMeas_;
  LWF::PredictionManager<PredictionExample> predictionManager_;
  LWF::UpdateManager<UpdateExample> updateManager1_;
  LWF::UpdateManager<UpdateExample> updateManager2_;
  const double dt_ = 0.1;
};

// Test constructors
TEST_F(UpdateModelTest, constructors) {
  LWF::FilterBase<PredictionExample> testFilter;
}
//
//// Test measurement adder
//TEST_F(UpdateModelTest, addMeasurement) {
//  PredictionExample* mpPrediction = new PredictionExample(testPredictionMeas_);
//  UpdateExample* mpUpdate = new UpdateExample(testUpdateMeas_);
//  UpdateExample* mpUpdate2 = new UpdateExample(testUpdateMeas_);
//  testFilter_.addPrediction(mpPrediction,0.2);
//  testFilter_.addUpdate(mpUpdate,0.3);
//  updateManager1_.addMeas(testUpdateMeas_,0.4,1);
//  PredictionExample::mtMeas::mtDiffVec predictionDiff;
//  static_cast<PredictionExample*>(testFilter_.predictionMap_[0.2])->meas_.boxMinus(testPredictionMeas_,predictionDiff);
//  ASSERT_NEAR(predictionDiff.norm(),0.0,1e-6);
//  UpdateExample::mtMeas::mtDiffVec updateDiff;
//  static_cast<UpdateExample*>(testFilter_.updateMap_[0][0.3])->meas_.boxMinus(testUpdateMeas_,updateDiff);
//  ASSERT_NEAR(updateDiff.norm(),0.0,1e-6);
//  static_cast<UpdateExample*>(testFilter_.updateMap_[1][0.4])->meas_.boxMinus(testUpdateMeas_,updateDiff);
//  ASSERT_NEAR(updateDiff.norm(),0.0,1e-6);
//}

// Test updateSafe (Only for 1 update type (wait time set to zero for the other)), co-test getSafeTime()
TEST_F(UpdateModelTest, updateSafe) {
  double safeTime = 0.0;
  predictionManager_.addMeas(testPredictionMeas_,0.1);
  updateManager1_.addMeas(testUpdateMeas_,0.1);
  ASSERT_TRUE(testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.1);
  testFilter_.updateSafe();
  ASSERT_EQ(testFilter_.safe_.t_,0.1);
  updateManager1_.addMeas(testUpdateMeas_,0.2);
  ASSERT_TRUE(!testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.1);
  testFilter_.updateSafe();
  ASSERT_EQ(testFilter_.safe_.t_,0.1);
  predictionManager_.addMeas(testPredictionMeas_,0.2);
  predictionManager_.addMeas(testPredictionMeas_,0.3);
  ASSERT_TRUE(testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.2);
  testFilter_.updateSafe();
  ASSERT_EQ(testFilter_.safe_.t_,0.2);
  updateManager1_.addMeas(testUpdateMeas_,0.3);
  ASSERT_TRUE(testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.3);
  testFilter_.updateSafe();
  ASSERT_EQ(testFilter_.safe_.t_,0.3);
}

// Test updateFront
TEST_F(UpdateModelTest, updateFront) {
  predictionManager_.addMeas(testPredictionMeas_,0.1);
  updateManager1_.addMeas(testUpdateMeas_,0.1);
  testFilter_.updateFront(0.5);
  ASSERT_EQ(testFilter_.safe_.t_,0.1);
  ASSERT_EQ(testFilter_.front_.t_,0.5);
  updateManager1_.addMeas(testUpdateMeas_,0.2);
  testFilter_.updateFront(0.2);
  ASSERT_EQ(testFilter_.safe_.t_,0.1);
  ASSERT_EQ(testFilter_.front_.t_,0.2);
  predictionManager_.addMeas(testPredictionMeas_,0.2);
  predictionManager_.addMeas(testPredictionMeas_,0.3);
  testFilter_.updateFront(0.3);
  ASSERT_EQ(testFilter_.safe_.t_,0.2);
  ASSERT_EQ(testFilter_.front_.t_,0.3);
  updateManager1_.addMeas(testUpdateMeas_,0.3);
  testFilter_.updateFront(0.3);
  ASSERT_EQ(testFilter_.safe_.t_,0.3);
  ASSERT_EQ(testFilter_.front_.t_,0.3);
}

//// Test cleaning
//TEST_F(UpdateModelTest, cleaning) {
//  predictionManager_.addMeas(testPredictionMeas_,0.1);
//  updateManager1_.addMeas(testUpdateMeas_,0.1);
//  updateManager1_.addMeas(testUpdateMeas_,0.2);
//  predictionManager_.addMeas(testPredictionMeas_,0.2);
//  predictionManager_.addMeas(testPredictionMeas_,0.3);
//  updateManager1_.addMeas(testUpdateMeas_,0.3,1);
//  ASSERT_EQ(testFilter_.predictionMap_.size(),3);
//  ASSERT_EQ(testFilter_.updateMap_[0].size(),2);
//  ASSERT_EQ(testFilter_.updateMap_[1].size(),1);
//  testFilter_.clean(0.2);
//  ASSERT_EQ(testFilter_.predictionMap_.size(),1);
//  ASSERT_EQ(testFilter_.updateMap_[0].size(),0);
//  ASSERT_EQ(testFilter_.updateMap_[1].size(),1);
//}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
