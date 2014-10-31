#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

// The fixture for testing class VectorState
class VectorStateTest : public virtual ::testing::Test {
 protected:
  VectorStateTest() {
    testVectorVector1_ << 2.1, -0.2, -1.9, 0.2;
    testVectorVector2_ << -10.6, 0.2, 25, -105.2;
    testVectorState1_ = testVectorVector1_;
    testVectorState2_ = testVectorVector2_;
  }
  virtual ~VectorStateTest() {
  }
  static const unsigned int D_ = 4;
  LWF::VectorState<D_> testVectorState1_;
  LWF::VectorState<D_> testVectorState2_;
  LWF::VectorState<D_>::mtDifVec difVecVector_;
  Eigen::Matrix<double,D_,1> testVectorVector1_;
  Eigen::Matrix<double,D_,1> testVectorVector2_;
};

// Test constructors
TEST_F(VectorStateTest, constructors) {
  LWF::VectorState<D_> testVectorState1;
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState1[i],0.0);
  }
  LWF::VectorState<D_> testVectorState2(testVectorState2_);
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState2[i],testVectorState2_[i]);
  }
}

// Test operator= and [] accessor
TEST_F(VectorStateTest, opertorAss) {
  testVectorState1_.setIdentity();
  testVectorState1_ = testVectorVector1_;
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState1_[i],testVectorVector1_(i));
  }
  testVectorState1_ = testVectorState2_;
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState1_[i],testVectorState2_[i]);
  }
}

// Test setIdentity and Identity
TEST_F(VectorStateTest, setIdentity) {
  testVectorState1_ = testVectorVector1_;
  testVectorState1_.setIdentity();
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState1_[i],0.0);
  }
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState1_.Identity()[i],0.0);
  }
}

// Test plus and minus
TEST_F(VectorStateTest, plusAndMinus) {
  testVectorState1_ = testVectorVector1_;
  difVecVector_ = testVectorVector2_;
  testVectorState1_.boxPlus(difVecVector_,testVectorState2_);
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testVectorState2_[i],testVectorVector1_[i]+testVectorVector2_[i]);
  }
  testVectorState2_.boxMinus(testVectorState1_,difVecVector_);
  for(int i=0;i<D_;i++){
    ASSERT_NEAR(difVecVector_[i],testVectorVector2_[i],1e-6);
  }
}

// Test block
TEST_F(VectorStateTest, block) {
  testVectorState1_ = testVectorVector1_;
  ASSERT_EQ(testVectorState1_.block<2>(0)(0),testVectorVector1_(0));
  ASSERT_EQ(testVectorState1_.block<2>(0)(1),testVectorVector1_(1));
  ASSERT_EQ(testVectorState1_.block<2>(1)(0),testVectorVector1_(1));
  ASSERT_EQ(testVectorState1_.block<2>(1)(1),testVectorVector1_(2));
  ASSERT_EQ(testVectorState1_.block<2>(2)(0),testVectorVector1_(2));
  ASSERT_EQ(testVectorState1_.block<2>(2)(1),testVectorVector1_(3));
}

// Test getId
TEST_F(VectorStateTest, getId) {
  for(int i=0;i<D_;i++){
    ASSERT_TRUE(testVectorState1_.getId(testVectorState1_[i]) == i);
  }
  ASSERT_TRUE(testVectorState1_.getId(testVectorState1_.block<2>(0)) == 0);
  ASSERT_TRUE(testVectorState1_.getId(testVectorState1_.block<2>(1)) == 1);
  ASSERT_TRUE(testVectorState1_.getId(testVectorState1_.block<2>(2)) == 2);
  ASSERT_TRUE(testVectorState1_.getId(testVectorState1_.block<3>(0)) == 0);
  ASSERT_TRUE(testVectorState1_.getId(testVectorState1_.block<3>(1)) == 1);
  ASSERT_TRUE(testVectorState1_.getId(testVectorState1_.block<4>(0)) == 0);
}

// The fixture for testing class StateSVQ
class StateSVQTest : public virtual ::testing::Test {
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
  }
  virtual ~StateSVQTest() {
  }
  static const unsigned int S_ = 4;
  static const unsigned int V_ = 3;
  static const unsigned int Q_ = 2;
  LWF::StateSVQ<S_,V_,Q_> testState1_;
  LWF::StateSVQ<S_,V_,Q_> testState2_;
  LWF::StateSVQ<S_,V_,Q_>::mtDifVec difVec_;
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
  LWF::StateSVQ<S_,V_,Q_> testState2(testState2_);
  testState2.boxMinus(testState2_,difVec_);
  ASSERT_NEAR(difVec_.norm(),0.0,1e-6);
}

// Test setIdentity and Identity
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
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testState1_.Identity().scalarList[i],0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testState1_.Identity().vectorList[i](0),0.0);
    ASSERT_EQ(testState1_.Identity().vectorList[i](1),0.0);
    ASSERT_EQ(testState1_.Identity().vectorList[i](2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testState1_.Identity().quaternionList[i].w(),1.0);
    ASSERT_EQ(testState1_.Identity().quaternionList[i].x(),0.0);
    ASSERT_EQ(testState1_.Identity().quaternionList[i].y(),0.0);
    ASSERT_EQ(testState1_.Identity().quaternionList[i].z(),0.0);
  }
}

// Test plus and minus
TEST_F(StateSVQTest, plusAndMinus) {
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

// Test accessors and naming
TEST_F(StateSVQTest, accessors) {
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testState1_.s(i) == testScalar1_[i]);
    ASSERT_TRUE(testState1_.s("s"+std::to_string(i)) == testScalar1_[i]);
    testState1_.sName(i) = "scalar"+std::to_string(i);
    ASSERT_TRUE(testState1_.s("scalar"+std::to_string(i)) == testScalar1_[i]);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testState1_.v(i)(0) == testVector1_[i](0));
    ASSERT_TRUE(testState1_.v(i)(1) == testVector1_[i](1));
    ASSERT_TRUE(testState1_.v(i)(2) == testVector1_[i](2));
    ASSERT_TRUE(testState1_.v("v"+std::to_string(i))(0) == testVector1_[i](0));
    ASSERT_TRUE(testState1_.v("v"+std::to_string(i))(1) == testVector1_[i](1));
    ASSERT_TRUE(testState1_.v("v"+std::to_string(i))(2) == testVector1_[i](2));
    testState1_.vName(i) = "vector"+std::to_string(i);
    ASSERT_TRUE(testState1_.v("vector"+std::to_string(i))(0) == testVector1_[i](0));
    ASSERT_TRUE(testState1_.v("vector"+std::to_string(i))(1) == testVector1_[i](1));
    ASSERT_TRUE(testState1_.v("vector"+std::to_string(i))(2) == testVector1_[i](2));
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState1_.q(i).isNear(testQuat1_[i],1e-6));
    ASSERT_TRUE(testState1_.q("q"+std::to_string(i)).isNear(testQuat1_[i],1e-6));
    testState1_.qName(i) = "quat"+std::to_string(i);
    ASSERT_TRUE(testState1_.q("quat"+std::to_string(i)).isNear(testQuat1_[i],1e-6));
  }
}

// Test operator=
TEST_F(StateSVQTest, operatorEQ) {
  testState2_ = testState1_;
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testState2_.s(i) == testScalar1_[i]);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testState2_.v(i)(0) == testVector1_[i](0));
    ASSERT_TRUE(testState2_.v(i)(1) == testVector1_[i](1));
    ASSERT_TRUE(testState2_.v(i)(2) == testVector1_[i](2));
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState2_.q(i).isNear(testQuat1_[i],1e-6));
  }
}

// Test getId
TEST_F(StateSVQTest, getId) {
  testState1_.setIdentity();
  testState1_ = testState2_;
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testState1_.getId(testState1_.s(i)) == i);
    ASSERT_TRUE(testState1_.getId("s"+std::to_string(i)) == i);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testState1_.getId(testState1_.v(i)) == S_+3*i);
    ASSERT_TRUE(testState1_.getId("v"+std::to_string(i)) == S_+3*i);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState1_.getId(testState1_.q(i)) == S_+3*(V_+i));
    ASSERT_TRUE(testState1_.getId("q"+std::to_string(i)) == S_+3*(V_+i));
  }
}

class Auxillary{
 public:
  Auxillary(){
    x_ = 1.0;
  };
  ~Auxillary(){};
  double x_;
};

// The fixture for testing class StateSVQ
class AugmentedStateTest : public ::testing::Test {
 protected:
  AugmentedStateTest() {
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
    for(int i=0;i<S_;i++){
      testState1_.scalarList[i] = testScalar1_[i];
    }
    for(int i=0;i<V_;i++){
      testState1_.vectorList[i] = testVector1_[i];
    }
    for(int i=0;i<Q_;i++){
      testState1_.quaternionList[i] = testQuat1_[i];
    }
    testState1_.aux().x_ = 2.3;
    for(int i=0;i<S_;i++){
      testState2_.scalarList[i] = testScalar2_[i];
    }
    for(int i=0;i<V_;i++){
      testState2_.vectorList[i] = testVector2_[i];
    }
    for(int i=0;i<Q_;i++){
      testState2_.quaternionList[i] = testQuat2_[i];
    }
    testState2_.aux().x_ = 3.2;
  }
  virtual ~AugmentedStateTest() {
  }
  static const unsigned int S_ = 4;
  static const unsigned int V_ = 3;
  static const unsigned int Q_ = 2;
  LWF::AugmentedState<LWF::StateSVQ<S_,V_,Q_>,Auxillary> testState1_;
  LWF::AugmentedState<LWF::StateSVQ<S_,V_,Q_>,Auxillary> testState2_;
  LWF::AugmentedState<LWF::StateSVQ<S_,V_,Q_>,Auxillary>::mtDifVec difVec_;
  double testScalar1_[S_];
  double testScalar2_[S_];
  Eigen::Vector3d testVector1_[V_];
  Eigen::Vector3d testVector2_[V_];
  rot::RotationQuaternionPD testQuat1_[Q_];
  rot::RotationQuaternionPD testQuat2_[Q_];
};

// Test constructors
TEST_F(AugmentedStateTest, constructors) {
  LWF::AugmentedState<LWF::StateSVQ<S_,V_,Q_>,Auxillary> testState1;
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
  ASSERT_EQ(testState1.aux().x_,1.0);
  LWF::AugmentedState<LWF::StateSVQ<S_,V_,Q_>,Auxillary> testState2(testState2_);
  testState2.boxMinus(testState2_,difVec_);
  ASSERT_NEAR(difVec_.norm(),0.0,1e-6);
  ASSERT_EQ(testState2.aux().x_,3.2);
}

// Test setIdentity and Identity
TEST_F(AugmentedStateTest, setIdentity) {
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
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testState1_.Identity().scalarList[i],0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testState1_.Identity().vectorList[i](0),0.0);
    ASSERT_EQ(testState1_.Identity().vectorList[i](1),0.0);
    ASSERT_EQ(testState1_.Identity().vectorList[i](2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testState1_.Identity().quaternionList[i].w(),1.0);
    ASSERT_EQ(testState1_.Identity().quaternionList[i].x(),0.0);
    ASSERT_EQ(testState1_.Identity().quaternionList[i].y(),0.0);
    ASSERT_EQ(testState1_.Identity().quaternionList[i].z(),0.0);
  }
}

// Test plus and minus
TEST_F(AugmentedStateTest, plusAndMinus) {
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
  ASSERT_EQ(testState2_.aux().x_,testState1_.aux().x_);
}

// Test accessors
TEST_F(AugmentedStateTest, accessors) {
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testState1_.s(i) == testScalar1_[i]);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testState1_.v(i)(0) == testVector1_[i](0));
    ASSERT_TRUE(testState1_.v(i)(1) == testVector1_[i](1));
    ASSERT_TRUE(testState1_.v(i)(2) == testVector1_[i](2));
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState1_.q(i).isNear(testQuat1_[i],1e-6));
  }
  ASSERT_EQ(testState1_.aux().x_,2.3);
}

// Test operator=
TEST_F(AugmentedStateTest, operatorEQ) {
  testState2_ = testState1_;
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testState2_.s(i) == testScalar1_[i]);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testState2_.v(i)(0) == testVector1_[i](0));
    ASSERT_TRUE(testState2_.v(i)(1) == testVector1_[i](1));
    ASSERT_TRUE(testState2_.v(i)(2) == testVector1_[i](2));
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState2_.q(i).isNear(testQuat1_[i],1e-6));
  }
}

// Test getId
TEST_F(AugmentedStateTest, getId) {
  testState1_.setIdentity();
  testState1_ = testState2_;
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

// The fixture for testing class StateSVQ
class PairStateTest : public virtual ::testing::Test, public StateSVQTest, public VectorStateTest{
 protected:
  PairStateTest() {
    testPairState1_.first() = testState1_;
    testPairState1_.second() = testVectorState1_;
    testPairState2_.first() = testState2_;
    testPairState2_.second() = testVectorState2_;
  }
  virtual ~PairStateTest() {
  }

  LWF::PairState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>> testPairState1_;
  LWF::PairState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>> testPairState2_;
  LWF::PairState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>>::mtDifVec difVecPairState_;
};

// Test constructors
TEST_F(PairStateTest, constructors) {
  LWF::PairState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>> testState1;
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testState1.first().scalarList[i],0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testState1.first().vectorList[i](0),0.0);
    ASSERT_EQ(testState1.first().vectorList[i](1),0.0);
    ASSERT_EQ(testState1.first().vectorList[i](2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testState1.first().quaternionList[i].w(),1.0);
    ASSERT_EQ(testState1.first().quaternionList[i].x(),0.0);
    ASSERT_EQ(testState1.first().quaternionList[i].y(),0.0);
    ASSERT_EQ(testState1.first().quaternionList[i].z(),0.0);
  }
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testState1.second()[i],0.0);
  }
  LWF::PairState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>> testState2(testPairState2_);
  testState2.boxMinus(testPairState2_,difVecPairState_);
  ASSERT_NEAR(difVecPairState_.norm(),0.0,1e-6);
}

// Test setIdentity and Identity
TEST_F(PairStateTest, setIdentity) {
  testPairState1_.setIdentity();
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testPairState1_.first().scalarList[i],0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testPairState1_.first().vectorList[i](0),0.0);
    ASSERT_EQ(testPairState1_.first().vectorList[i](1),0.0);
    ASSERT_EQ(testPairState1_.first().vectorList[i](2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testPairState1_.first().quaternionList[i].w(),1.0);
    ASSERT_EQ(testPairState1_.first().quaternionList[i].x(),0.0);
    ASSERT_EQ(testPairState1_.first().quaternionList[i].y(),0.0);
    ASSERT_EQ(testPairState1_.first().quaternionList[i].z(),0.0);
  }
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testPairState1_.second()[i],0.0);
  }
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testPairState1_.Identity().first().scalarList[i],0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testPairState1_.Identity().first().vectorList[i](0),0.0);
    ASSERT_EQ(testPairState1_.Identity().first().vectorList[i](1),0.0);
    ASSERT_EQ(testPairState1_.Identity().first().vectorList[i](2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testPairState1_.Identity().first().quaternionList[i].w(),1.0);
    ASSERT_EQ(testPairState1_.Identity().first().quaternionList[i].x(),0.0);
    ASSERT_EQ(testPairState1_.Identity().first().quaternionList[i].y(),0.0);
    ASSERT_EQ(testPairState1_.Identity().first().quaternionList[i].z(),0.0);
  }
  for(int i=0;i<D_;i++){
    ASSERT_EQ(testPairState1_.Identity().second()[i],0.0);
  }
}

// Test plus and minus
TEST_F(PairStateTest, plusAndMinus) {
  testPairState2_.boxMinus(testPairState1_,difVecPairState_);
  unsigned int index=0;
  for(int i=0;i<S_;i++){
    ASSERT_EQ(difVecPairState_(index),testScalar2_[i]-testScalar1_[i]);
    index ++;
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(difVecPairState_(index),testVector2_[i](0)-testVector1_[i](0));
    index ++;
    ASSERT_EQ(difVecPairState_(index),testVector2_[i](1)-testVector1_[i](1));
    index ++;
    ASSERT_EQ(difVecPairState_(index),testVector2_[i](2)-testVector1_[i](2));
    index ++;
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(difVecPairState_(index),testQuat2_[i].boxMinus(testQuat1_[i])(0));
    index ++;
    ASSERT_EQ(difVecPairState_(index),testQuat2_[i].boxMinus(testQuat1_[i])(1));
    index ++;
    ASSERT_EQ(difVecPairState_(index),testQuat2_[i].boxMinus(testQuat1_[i])(2));
    index ++;
  }
  for(int i=0;i<D_;i++){
    ASSERT_NEAR((difVecPairState_[LWF::StateSVQ<S_,V_,Q_>::D_+i]),(testVectorVector2_[i]-testVectorVector1_[i]),1e-6);
  }
  testPairState1_.boxPlus(difVecPairState_,testPairState2_);
  for(int i=0;i<S_;i++){
    ASSERT_NEAR(testPairState2_.first().scalarList[i],testScalar2_[i],1e-6);
  }
  for(int i=0;i<V_;i++){
    ASSERT_NEAR(testPairState2_.first().vectorList[i](0),testVector2_[i](0),1e-6);
    ASSERT_NEAR(testPairState2_.first().vectorList[i](1),testVector2_[i](1),1e-6);
    ASSERT_NEAR(testPairState2_.first().vectorList[i](2),testVector2_[i](2),1e-6);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testPairState2_.first().quaternionList[i].isNear(testQuat2_[i],1e-6));
  }
  for(int i=0;i<D_;i++){
    ASSERT_NEAR((testPairState2_.second()[i]),testVectorVector2_[i],1e-6);
  }
}

// Test accessors
TEST_F(PairStateTest, accessors) {
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testPairState1_.first().s(i) == testScalar1_[i]);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testPairState1_.first().v(i)(0) == testVector1_[i](0));
    ASSERT_TRUE(testPairState1_.first().v(i)(1) == testVector1_[i](1));
    ASSERT_TRUE(testPairState1_.first().v(i)(2) == testVector1_[i](2));
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testPairState1_.first().q(i).isNear(testQuat1_[i],1e-6));
  }
  for(int i=0;i<D_;i++){
    ASSERT_NEAR((testPairState1_.second()[i]),testVectorVector1_[i],1e-6);
  }
}

// Test operator=
TEST_F(PairStateTest, operatorEQ) {
  testPairState2_ = testPairState1_;
  for(int i=0;i<S_;i++){
    ASSERT_TRUE(testPairState2_.first().s(i) == testScalar1_[i]);
  }
  for(int i=0;i<V_;i++){
    ASSERT_TRUE(testPairState2_.first().v(i)(0) == testVector1_[i](0));
    ASSERT_TRUE(testPairState2_.first().v(i)(1) == testVector1_[i](1));
    ASSERT_TRUE(testPairState2_.first().v(i)(2) == testVector1_[i](2));
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testPairState2_.first().q(i).isNear(testQuat1_[i],1e-6));
  }
  for(int i=0;i<D_;i++){
    ASSERT_NEAR((testPairState2_.second()[i]),testVectorVector1_[i],1e-6);
  }
}

// The fixture for testing class ComposedState
class ComposedStateTest : public virtual ::testing::Test, public StateSVQTest, public VectorStateTest{
 protected:
  ComposedStateTest() {

  }
  virtual ~ComposedStateTest() {
  }
  LWF::ComposedState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>> testState_;
  LWF::ComposedState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>>::mtDifVec difVecPairState_;
};

// Test constructors
TEST_F(ComposedStateTest, constructors) {
  LWF::ComposedState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>> testState1;
  LWF::StateSVQ<S_,V_,Q_> stateSVQ;
  LWF::VectorState<D_> vectorState;
  stateSVQ = testState1.state_;
  stateSVQ = testState1.get<LWF::StateSVQ<S_,V_,Q_>,0>();
  vectorState = testState1.subComposedState_;
  vectorState = testState1.get<LWF::VectorState<D_>,1>();

  LWF::StateSVQNew<S_,V_,Q_> testState2;
  testState2.print();
  testState2.s(0) = 2.3;
  testState2.print();
//  for(int i=0;i<S_;i++){
//    ASSERT_EQ(testState1.first().scalarList[i],0.0);
//  }
//  for(int i=0;i<V_;i++){
//    ASSERT_EQ(testState1.first().vectorList[i](0),0.0);
//    ASSERT_EQ(testState1.first().vectorList[i](1),0.0);
//    ASSERT_EQ(testState1.first().vectorList[i](2),0.0);
//  }
//  for(int i=0;i<Q_;i++){
//    ASSERT_EQ(testState1.first().quaternionList[i].w(),1.0);
//    ASSERT_EQ(testState1.first().quaternionList[i].x(),0.0);
//    ASSERT_EQ(testState1.first().quaternionList[i].y(),0.0);
//    ASSERT_EQ(testState1.first().quaternionList[i].z(),0.0);
//  }
//  for(int i=0;i<D_;i++){
//    ASSERT_EQ(testState1.second()[i],0.0);
//  }
//  LWF::PairState<LWF::StateSVQ<S_,V_,Q_>,LWF::VectorState<D_>> testState2(testPairState2_);
//  testState2.boxMinus(testPairState2_,difVecPairState_);
//  ASSERT_NEAR(difVecPairState_.norm(),0.0,1e-6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
