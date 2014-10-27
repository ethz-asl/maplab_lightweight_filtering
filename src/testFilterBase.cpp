#include "Update.hpp"
#include "Prediction.hpp"
#include "State.hpp"
#include "FilterBase.hpp"
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
    testFilter_.registerUpdateManager(updateManager1_);
    testFilter_.registerUpdateManager(updateManager2_);
    testFilter_.writeToInfo("test.info");
  }
  virtual ~UpdateModelTest() {
  }
  UpdateExample testUpdate_;
  PredictionExample testPrediction_;
  LWF::FilterBase<PredictionExample> testFilter_;
  State testState_;
  UpdateMeas testUpdateMeas_;
  PredictionMeas testPredictionMeas_;
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
  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.1);
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
  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.2);
  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.3);
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
  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.1);
  updateManager1_.addMeas(testUpdateMeas_,0.1);
  testFilter_.updateFront(0.5);
  ASSERT_EQ(testFilter_.safe_.t_,0.1);
  ASSERT_EQ(testFilter_.front_.t_,0.5);
  updateManager1_.addMeas(testUpdateMeas_,0.2);
  testFilter_.updateFront(0.2);
  ASSERT_EQ(testFilter_.safe_.t_,0.1);
  ASSERT_EQ(testFilter_.front_.t_,0.2);
  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.2);
  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.3);
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
//  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.1);
//  updateManager1_.addMeas(testUpdateMeas_,0.1);
//  updateManager1_.addMeas(testUpdateMeas_,0.2);
//  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.2);
//  testFilter_.predictionManager_.addMeas(testPredictionMeas_,0.3);
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
