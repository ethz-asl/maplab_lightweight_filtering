#include "PredictionModel.hpp"
#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

class State: public LWF::StateSVQ<4,3,2>{
 public:
  State(){};
  ~State(){};
};
class PredictionMeas: public LWF::StateSVQ<2,2,2>{
 public:
  PredictionMeas(){};
  ~PredictionMeas(){};
};
class PredictionModelExample: public LWF::PredictionModel<State,PredictionMeas,10>{
 public:
  PredictionModelExample(){};
  ~PredictionModelExample(){};
};

// The fixture for testing class PredictionModel
class PredictionModelTest : public ::testing::Test {
 protected:
  PredictionModelTest() {
    assert(State::V_>=State::Q_-1);
    testScalar1_[0] = 4.5;
    testScalar2_[0] = -17.34;
    for(int i=1;i<State::S_;i++){
      testScalar1_[i] = testScalar1_[i-1] + i*i*46.2;
      testScalar2_[i] = testScalar2_[i-1] - i*i*0.01;
    }
    testVector1_[0] << 2.1, -0.2, -1.9;
    testVector2_[0] << -10.6, 0.2, -105.2;
    for(int i=1;i<State::V_;i++){
      testVector1_[i] = testVector1_[i-1] + Eigen::Vector3d(0.3,10.9,2.3);
      testVector2_[i] = testVector2_[i-1] + Eigen::Vector3d(-1.5,12,1785.23);
    }
    testQuat1_[0] = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testQuat2_[0] = rot::RotationQuaternionPD(0.0,0.36,0.48,0.8);
    for(int i=1;i<State::Q_;i++){
      testQuat1_[i] = testQuat1_[i-1].boxPlus(testVector1_[i-1]);
      testQuat2_[i] = testQuat2_[i-1].boxPlus(testVector2_[i-1]);
    }
  }
  virtual ~PredictionModelTest() {
  }
  State testState1_;
  State testState2_;
  State::DiffVec difVec_;
  double testScalar1_[State::S_];
  double testScalar2_[State::S_];
  Eigen::Vector3d testVector1_[State::V_];
  Eigen::Vector3d testVector2_[State::V_];
  rot::RotationQuaternionPD testQuat1_[State::Q_];
  rot::RotationQuaternionPD testQuat2_[State::Q_];
};

// Test constructors
TEST_F(PredictionModelTest, constructors) {
  State testState1;
  for(int i=0;i<State::S_;i++){
    ASSERT_EQ(testState1.scalarList[i],0.0);
  }
  for(int i=0;i<State::V_;i++){
    ASSERT_EQ(testState1.vectorList[i](0),0.0);
    ASSERT_EQ(testState1.vectorList[i](1),0.0);
    ASSERT_EQ(testState1.vectorList[i](2),0.0);
  }
  for(int i=0;i<State::Q_;i++){
    ASSERT_EQ(testState1.quaternionList[i].w(),1.0);
    ASSERT_EQ(testState1.quaternionList[i].x(),0.0);
    ASSERT_EQ(testState1.quaternionList[i].y(),0.0);
    ASSERT_EQ(testState1.quaternionList[i].z(),0.0);
  }
}

// Test setIdentity
TEST_F(PredictionModelTest, setIdentity) {
  testState1_.setIdentity();
  for(int i=0;i<State::S_;i++){
    ASSERT_EQ(testState1_.scalarList[i],0.0);
  }
  for(int i=0;i<State::V_;i++){
    ASSERT_EQ(testState1_.vectorList[i](0),0.0);
    ASSERT_EQ(testState1_.vectorList[i](1),0.0);
    ASSERT_EQ(testState1_.vectorList[i](2),0.0);
  }
  for(int i=0;i<State::Q_;i++){
    ASSERT_EQ(testState1_.quaternionList[i].w(),1.0);
    ASSERT_EQ(testState1_.quaternionList[i].x(),0.0);
    ASSERT_EQ(testState1_.quaternionList[i].y(),0.0);
    ASSERT_EQ(testState1_.quaternionList[i].z(),0.0);
  }
}

// Test plus and minus
TEST_F(PredictionModelTest, plusAndMinus) {
  for(int i=0;i<State::S_;i++){
    testState1_.scalarList[i] = testScalar1_[i];
  }
  for(int i=0;i<State::V_;i++){
    testState1_.vectorList[i] = testVector1_[i];
  }
  for(int i=0;i<State::Q_;i++){
    testState1_.quaternionList[i] = testQuat1_[i];
  }
  for(int i=0;i<State::S_;i++){
    testState2_.scalarList[i] = testScalar2_[i];
  }
  for(int i=0;i<State::V_;i++){
    testState2_.vectorList[i] = testVector2_[i];
  }
  for(int i=0;i<State::Q_;i++){
    testState2_.quaternionList[i] = testQuat2_[i];
  }
  testState2_.boxMinus(testState1_,difVec_);
  unsigned int index=0;
  for(int i=0;i<State::S_;i++){
    ASSERT_EQ(difVec_(index),testScalar2_[i]-testScalar1_[i]);
    index ++;
  }
  for(int i=0;i<State::V_;i++){
    ASSERT_EQ(difVec_(index),testVector2_[i](0)-testVector1_[i](0));
    index ++;
    ASSERT_EQ(difVec_(index),testVector2_[i](1)-testVector1_[i](1));
    index ++;
    ASSERT_EQ(difVec_(index),testVector2_[i](2)-testVector1_[i](2));
    index ++;
  }
  for(int i=0;i<State::Q_;i++){
    ASSERT_EQ(difVec_(index),testQuat2_[i].boxMinus(testQuat1_[i])(0));
    index ++;
    ASSERT_EQ(difVec_(index),testQuat2_[i].boxMinus(testQuat1_[i])(1));
    index ++;
    ASSERT_EQ(difVec_(index),testQuat2_[i].boxMinus(testQuat1_[i])(2));
    index ++;
  }
  testState1_.boxPlus(difVec_,testState2_);
  for(int i=0;i<State::S_;i++){
    ASSERT_NEAR(testState2_.scalarList[i],testScalar2_[i],1e-6);
  }
  for(int i=0;i<State::V_;i++){
    ASSERT_NEAR(testState2_.vectorList[i](0),testVector2_[i](0),1e-6);
    ASSERT_NEAR(testState2_.vectorList[i](1),testVector2_[i](1),1e-6);
    ASSERT_NEAR(testState2_.vectorList[i](2),testVector2_[i](2),1e-6);
  }
  for(int i=0;i<State::Q_;i++){
    ASSERT_TRUE(testState2_.quaternionList[i].isNear(testQuat2_[i],1e-6));
  }
}

// Test accessors
TEST_F(PredictionModelTest, accessors) {
  for(int i=0;i<State::S_;i++){
    testState1_.scalarList[i] = testScalar1_[i];
  }
  for(int i=0;i<State::V_;i++){
    testState1_.vectorList[i] = testVector1_[i];
  }
  for(int i=0;i<State::Q_;i++){
    testState1_.quaternionList[i] = testQuat1_[i];
  }
  for(int i=0;i<State::S_;i++){
    testState1_.s(i) = testScalar1_[i];
  }
  for(int i=0;i<State::V_;i++){
    testState1_.v(i)(0) = testVector1_[i](0);
    testState1_.v(i)(1) = testVector1_[i](1);
    testState1_.v(i)(2) = testVector1_[i](2);
  }
  for(int i=0;i<State::Q_;i++){
    ASSERT_TRUE(testState1_.q(i).isNear(testQuat1_[i],1e-6));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
