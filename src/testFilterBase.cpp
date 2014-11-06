#include "TestClasses.hpp"
#include "FilterBase.hpp"
#include "gtest/gtest.h"
#include <assert.h>

using namespace LWFTest;

typedef ::testing::Types<
    NonlinearTest,
    LinearTest
> TestClasses;

// The fixture for testing class FilterBase
template<typename TestClass>
class FilterBaseTest : public ::testing::Test, public TestClass {
 protected:
  FilterBaseTest() {
    this->init(this->testState_,this->testUpdateMeas_,this->testPredictionMeas_);
    this->updateManager2_.maxWaitTime_ = 0.0;
    this->testFilter_.registerUpdateManager(this->updateManager1_,"Update1");
    this->testFilter_.registerUpdateManager(this->updateManager2_,"Update2");
    this->testFilter_.registerPredictionManager(this->predictionManager_,"Prediction");
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
  mtState testState_;
  mtUpdateMeas testUpdateMeas_;
  mtPredictionMeas testPredictionMeas_;
  LWF::PredictionManager<mtPredictionExample> predictionManager_;
  LWF::UpdateManager<mtUpdateExample> updateManager1_;
  LWF::UpdateManager<mtUpdateExample> updateManager2_;
  const double dt_ = 0.1;
};

TYPED_TEST_CASE(FilterBaseTest, TestClasses);

// Test constructors
TYPED_TEST(FilterBaseTest, constructors) {
  LWF::FilterBase<typename TestFixture::mtPredictionExample> testFilter;
}
//
//// Test measurement adder
//TYPED_TEST(FilterBaseTest, addMeasurement) {
//  typename TestFixture::mtPredictionExample* mpPrediction = new typename TestFixture::mtPredictionExample(this->testPredictionMeas_);
//  UpdateExample* mpUpdate = new UpdateExample(this->testUpdateMeas_);
//  UpdateExample* mpUpdate2 = new UpdateExample(this->testUpdateMeas_);
//  this->testFilter_.addPrediction(mpPrediction,0.2);
//  this->testFilter_.addUpdate(mpUpdate,0.3);
//  this->updateManager1_.addMeas(this->testUpdateMeas_,0.4,1);
//  typename TestFixture::mtPredictionExample::mtMeas::mtDifVec predictionDiff;
//  static_cast<typename TestFixture::mtPredictionExample*>(this->testFilter_.predictionMap_[0.2])->meas_.boxMinus(this->testPredictionMeas_,predictionDiff);
//  ASSERT_NEAR(predictionDiff.norm(),0.0,1e-6);
//  UpdateExample::mtMeas::mtDifVec updateDiff;
//  static_cast<UpdateExample*>(this->testFilter_.updateMap_[0][0.3])->meas_.boxMinus(this->testUpdateMeas_,updateDiff);
//  ASSERT_NEAR(updateDiff.norm(),0.0,1e-6);
//  static_cast<UpdateExample*>(this->testFilter_.updateMap_[1][0.4])->meas_.boxMinus(this->testUpdateMeas_,updateDiff);
//  ASSERT_NEAR(updateDiff.norm(),0.0,1e-6);
//}

// Test updateSafe (Only for 1 update type (wait time set to zero for the other)), co-test getSafeTime()
TYPED_TEST(FilterBaseTest, updateSafe) {
  double safeTime = 0.0;
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager1_.addMeas(this->testUpdateMeas_,0.1);
  ASSERT_TRUE(this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.1);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  this->updateManager1_.addMeas(this->testUpdateMeas_,0.2);
  ASSERT_TRUE(!this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.1);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
  ASSERT_TRUE(this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.2);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.2);
  this->updateManager1_.addMeas(this->testUpdateMeas_,0.3);
  ASSERT_TRUE(this->testFilter_.getSafeTime(safeTime));
  ASSERT_EQ(safeTime,0.3);
  this->testFilter_.updateSafe();
  ASSERT_EQ(this->testFilter_.safe_.t_,0.3);
}

// Test updateFront
TYPED_TEST(FilterBaseTest, updateFront) {
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
  this->updateManager1_.addMeas(this->testUpdateMeas_,0.1);
  this->testFilter_.updateFront(0.5);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  ASSERT_EQ(this->testFilter_.front_.t_,0.5);
  this->updateManager1_.addMeas(this->testUpdateMeas_,0.2);
  this->testFilter_.updateFront(0.2);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.1);
  ASSERT_EQ(this->testFilter_.front_.t_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
  this->testFilter_.updateFront(0.3);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.2);
  ASSERT_EQ(this->testFilter_.front_.t_,0.3);
  this->updateManager1_.addMeas(this->testUpdateMeas_,0.3);
  this->testFilter_.updateFront(0.3);
  ASSERT_EQ(this->testFilter_.safe_.t_,0.3);
  ASSERT_EQ(this->testFilter_.front_.t_,0.3);
}

//// Test cleaning
//TYPED_TEST(FilterBaseTest, cleaning) {
//  this->predictionManager_.addMeas(this->testPredictionMeas_,0.1);
//  this->updateManager1_.addMeas(this->testUpdateMeas_,0.1);
//  this->updateManager1_.addMeas(this->testUpdateMeas_,0.2);
//  this->predictionManager_.addMeas(this->testPredictionMeas_,0.2);
//  this->predictionManager_.addMeas(this->testPredictionMeas_,0.3);
//  this->updateManager1_.addMeas(this->testUpdateMeas_,0.3,1);
//  ASSERT_EQ(this->testFilter_.predictionMap_.size(),3);
//  ASSERT_EQ(this->testFilter_.updateMap_[0].size(),2);
//  ASSERT_EQ(this->testFilter_.updateMap_[1].size(),1);
//  this->testFilter_.clean(0.2);
//  ASSERT_EQ(this->testFilter_.predictionMap_.size(),1);
//  ASSERT_EQ(this->testFilter_.updateMap_[0].size(),0);
//  ASSERT_EQ(this->testFilter_.updateMap_[1].size(),1);
//}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
