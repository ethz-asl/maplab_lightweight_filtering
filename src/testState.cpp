#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

// The fixture for testing class VectorState
class VectorStateTest : public ::testing::Test {
 protected:
  VectorStateTest() {
    testVector1_ << 2.1, -0.2, -1.9, 0.2;
    testVector2_ << -10.6, 0.2, 25, -105.2;
  }
  virtual ~VectorStateTest() {
  }
  static const unsigned int D_ = 4;
  LWF::VectorState<D_> testVectorState1_;
  LWF::VectorState<D_> testVectorState2_;
  LWF::VectorState<D_>::DiffVec difVec_;
  Eigen::Matrix<double,D_,1> testVector1_;
  Eigen::Matrix<double,D_,1> testVector2_;
};

// Test constructors
TEST_F(VectorStateTest, constructors) {
  LWF::VectorState<D_> testVectorState1;
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState1[i],0.0);
  }
}

// Test setIdentity
TEST_F(VectorStateTest, setIdentity) {
  testVectorState1_.setIdentity();
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState1_[i],0.0);
  }
}

// Test plus and minus
TEST_F(VectorStateTest, plusAndMinus) {
  testVectorState1_.vector_ = testVector1_;
  difVec_ = testVector2_;
  testVectorState1_.boxPlus(difVec_,testVectorState2_);
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState2_[i],testVector1_[i]+testVector2_[i]);
  }
  testVectorState2_.boxMinus(testVectorState1_,difVec_);
  for(int i=0;i<D_;i++){
    ASSERT_NEAR(difVec_[i],testVector2_[i],1e-6);
  }
}

// The fixture for testing class StateSVQ
class StateSVQTest : public ::testing::Test {
 protected:
  StateSVQTest() {
    assert(V_>=Q_-1);
    testScalar1_[0] = 4.5;
    testScalar2_[0] = -17.34;
    for(int i=1;i<S_;i++){
      testScalar1_[i] = testScalar1_[i-1] + i*i*46.2;
      testScalar2_[i] = testScalar2_[i-1] - i*i*0.01;
    }
    testVector1_[0] << 2.1, -0.2, -1.9;
    testVector2_[0] << -10.6, 0.2, -105.2;
    for(int i=1;i<V_;i++){
      testVector1_[i] = testVector1_[i-1] + Eigen::Vector3d(0.3,10.9,2.3);
      testVector2_[i] = testVector2_[i-1] + Eigen::Vector3d(-1.5,12,1785.23);
    }
    testQuat1_[0] = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testQuat2_[0] = rot::RotationQuaternionPD(0.0,0.36,0.48,0.8);
    for(int i=1;i<Q_;i++){
      testQuat1_[i] = testQuat1_[i-1].boxPlus(testVector1_[i-1]);
      testQuat2_[i] = testQuat2_[i-1].boxPlus(testVector2_[i-1]);
    }
  }
  virtual ~StateSVQTest() {
  }
  static const unsigned int S_ = 4;
  static const unsigned int V_ = 3;
  static const unsigned int Q_ = 2;
  LWF::StateSVQ<S_,V_,Q_> testState1_;
  LWF::StateSVQ<S_,V_,Q_> testState2_;
  LWF::StateSVQ<S_,V_,Q_>::DiffVec difVec_;
  double testScalar1_[S_];
  double testScalar2_[S_];
  Eigen::Vector3d testVector1_[V_];
  Eigen::Vector3d testVector2_[V_];
  rot::RotationQuaternionPD testQuat1_[Q_];
  rot::RotationQuaternionPD testQuat2_[Q_];
};

// Test constructors
TEST_F(StateSVQTest, constructors) {
  LWF::StateSVQ<S_,V_,Q_> testState1;
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testState1.scalarList[i],0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testState1.vectorList[i](0),0.0);
    ASSERT_EQ(testState1.vectorList[i](1),0.0);
    ASSERT_EQ(testState1.vectorList[i](2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testState1.quaternionList[i].w(),1.0);
    ASSERT_EQ(testState1.quaternionList[i].x(),0.0);
    ASSERT_EQ(testState1.quaternionList[i].y(),0.0);
    ASSERT_EQ(testState1.quaternionList[i].z(),0.0);
  }
}

// Test setIdentity
TEST_F(StateSVQTest, setIdentity) {
  testState1_.setIdentity();
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testState1_.scalarList[i],0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testState1_.vectorList[i](0),0.0);
    ASSERT_EQ(testState1_.vectorList[i](1),0.0);
    ASSERT_EQ(testState1_.vectorList[i](2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testState1_.quaternionList[i].w(),1.0);
    ASSERT_EQ(testState1_.quaternionList[i].x(),0.0);
    ASSERT_EQ(testState1_.quaternionList[i].y(),0.0);
    ASSERT_EQ(testState1_.quaternionList[i].z(),0.0);
  }
}

// Test plus and minus
TEST_F(StateSVQTest, plusAndMinus) {
  for(int i=0;i<S_;i++){
    testState1_.scalarList[i] = testScalar1_[i];
  }
  for(int i=0;i<V_;i++){
    testState1_.vectorList[i] = testVector1_[i];
  }
  for(int i=0;i<Q_;i++){
    testState1_.quaternionList[i] = testQuat1_[i];
  }
  for(int i=0;i<S_;i++){
    testState2_.scalarList[i] = testScalar2_[i];
  }
  for(int i=0;i<V_;i++){
    testState2_.vectorList[i] = testVector2_[i];
  }
  for(int i=0;i<Q_;i++){
    testState2_.quaternionList[i] = testQuat2_[i];
  }
  testState2_.boxMinus(testState1_,difVec_);
  unsigned int index=0;
  for(int i=0;i<S_;i++){
    ASSERT_EQ(difVec_(index),testScalar2_[i]-testScalar1_[i]);
    index ++;
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(difVec_(index),testVector2_[i](0)-testVector1_[i](0));
    index ++;
    ASSERT_EQ(difVec_(index),testVector2_[i](1)-testVector1_[i](1));
    index ++;
    ASSERT_EQ(difVec_(index),testVector2_[i](2)-testVector1_[i](2));
    index ++;
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(difVec_(index),testQuat2_[i].boxMinus(testQuat1_[i])(0));
    index ++;
    ASSERT_EQ(difVec_(index),testQuat2_[i].boxMinus(testQuat1_[i])(1));
    index ++;
    ASSERT_EQ(difVec_(index),testQuat2_[i].boxMinus(testQuat1_[i])(2));
    index ++;
  }
  testState1_.boxPlus(difVec_,testState2_);
  for(int i=0;i<S_;i++){
    ASSERT_NEAR(testState2_.scalarList[i],testScalar2_[i],1e-6);
  }
  for(int i=0;i<V_;i++){
    ASSERT_NEAR(testState2_.vectorList[i](0),testVector2_[i](0),1e-6);
    ASSERT_NEAR(testState2_.vectorList[i](1),testVector2_[i](1),1e-6);
    ASSERT_NEAR(testState2_.vectorList[i](2),testVector2_[i](2),1e-6);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState2_.quaternionList[i].isNear(testQuat2_[i],1e-6));
  }
}

// Test accessors
TEST_F(StateSVQTest, accessors) {
  for(int i=0;i<S_;i++){
    testState1_.scalarList[i] = testScalar1_[i];
  }
  for(int i=0;i<V_;i++){
    testState1_.vectorList[i] = testVector1_[i];
  }
  for(int i=0;i<Q_;i++){
    testState1_.quaternionList[i] = testQuat1_[i];
  }
  for(int i=0;i<S_;i++){
    testState1_.s(i) = testScalar1_[i];
  }
  for(int i=0;i<V_;i++){
    testState1_.v(i)(0) = testVector1_[i](0);
    testState1_.v(i)(1) = testVector1_[i](1);
    testState1_.v(i)(2) = testVector1_[i](2);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState1_.q(i).isNear(testQuat1_[i],1e-6));
  }
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testState1_.getId(testState1_.s(i)) == i);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testState1_.getId(testState1_.v(i)) == S_+3*i);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState1_.getId(testState1_.q(i)) == S_+3*(V_+i));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
