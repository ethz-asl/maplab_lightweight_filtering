#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

// The fixture for testing class ScalarState
class ScalarStateTest : public virtual ::testing::Test {
 protected:
  ScalarStateTest() {
    testState1_.setRandom(1);
    testState2_.setRandom(2);
  }
  virtual ~ScalarStateTest() {
  }
  LWF::ScalarState testState1_;
  LWF::ScalarState testState2_;
  LWF::ScalarState::mtDifVec difVec_;
};

// Test constructors
TEST_F(ScalarStateTest, constructor) {
  LWF::ScalarState testState1;
}

// Test setIdentity and Identity
TEST_F(ScalarStateTest, setIdentity) {
  testState1_.setIdentity();
  ASSERT_EQ(testState1_.s_,0.0);
  ASSERT_EQ(LWF::ScalarState::Identity().s_,0.0);
}

// Test plus and minus
TEST_F(ScalarStateTest, plusAndMinus) {
  testState2_.boxMinus(testState1_,difVec_);
  ASSERT_EQ(difVec_(0),testState2_.s_-testState1_.s_);
  LWF::ScalarState testState3;
  testState1_.boxPlus(difVec_,testState3);
  ASSERT_NEAR(testState2_.s_,testState3.s_,1e-6);
}

// Test getValue, getId
TEST_F(ScalarStateTest, accessors) {
  ASSERT_TRUE(testState1_.getValue<0>() == testState1_.s_);
  ASSERT_TRUE(testState1_.getId<0>() == 0);
}

// Test operator=
TEST_F(ScalarStateTest, operatorEQ) {
  testState2_ = testState1_;
  ASSERT_NEAR(testState2_.s_,testState1_.s_,1e-6);
}

// Test createDefaultNames
TEST_F(ScalarStateTest, naming) {
  testState1_.createDefaultNames("test");
  ASSERT_TRUE(testState1_.name_ == "test");
}

// The fixture for testing class VectorState
class VectorStateTest : public virtual ::testing::Test {
 protected:
  VectorStateTest() {
    testState1_.setRandom(1);
    testState2_.setRandom(2);
  }
  virtual ~VectorStateTest() {
  }
  static const unsigned int N_ = 4;
  LWF::VectorState<N_> testState1_;
  LWF::VectorState<N_> testState2_;
  LWF::VectorState<N_>::mtDifVec difVec_;
};

// Test constructors
TEST_F(VectorStateTest, constructor) {
  LWF::VectorState<N_> testState1;
}

// Test setIdentity and Identity
TEST_F(VectorStateTest, setIdentity) {
  testState1_.setIdentity();
  ASSERT_EQ(testState1_.v_.norm(),0.0);
  ASSERT_EQ(LWF::VectorState<N_>::Identity().v_.norm(),0.0);
}

// Test plus and minus
TEST_F(VectorStateTest, plusAndMinus) {
  testState2_.boxMinus(testState1_,difVec_);
  ASSERT_NEAR((difVec_-(testState2_.v_-testState1_.v_)).norm(),0.0,1e-6);
  LWF::VectorState<N_> testState3;
  testState1_.boxPlus(difVec_,testState3);
  ASSERT_NEAR((testState2_.v_-testState3.v_).norm(),0.0,1e-6);
}

// Test getValue, getId
TEST_F(VectorStateTest, accessors) {
  ASSERT_TRUE(testState1_.getValue<0>() == testState1_.v_);
  ASSERT_TRUE(testState1_.getId<0>() == 0);
}

// Test operator=
TEST_F(VectorStateTest, operatorEQ) {
  testState2_ = testState1_;
  ASSERT_NEAR((testState2_.v_-testState1_.v_).norm(),0.0,1e-6);
}

// Test createDefaultNames
TEST_F(VectorStateTest, naming) {
  testState1_.createDefaultNames("test");
  ASSERT_TRUE(testState1_.name_ == "test");
}

// The fixture for testing class VectorState
class QuaternionStateTest : public virtual ::testing::Test {
 protected:
  QuaternionStateTest() {
    testState1_.setRandom(1);
    testState2_.setRandom(2);
  }
  virtual ~QuaternionStateTest() {
  }
  LWF::QuaternionState testState1_;
  LWF::QuaternionState testState2_;
  LWF::QuaternionState::mtDifVec difVec_;
};

// Test constructors
TEST_F(QuaternionStateTest, constructor) {
  LWF::QuaternionState testState1;
}

// Test setIdentity and Identity
TEST_F(QuaternionStateTest, setIdentity) {
  testState1_.setIdentity();
  ASSERT_TRUE(testState1_.q_.isNear(rot::RotationQuaternionPD(),1e-6));
  ASSERT_TRUE(LWF::QuaternionState::Identity().q_.isNear(rot::RotationQuaternionPD(),1e-6));
}

// Test plus and minus
TEST_F(QuaternionStateTest, plusAndMinus) {
  testState2_.boxMinus(testState1_,difVec_);
  ASSERT_NEAR((difVec_-testState2_.q_.boxMinus(testState1_.q_)).norm(),0.0,1e-6);
  LWF::QuaternionState testState3;
  testState1_.boxPlus(difVec_,testState3);
  ASSERT_TRUE(testState2_.q_.isNear(testState3.q_,1e-6));
}

// Test getValue, getId
TEST_F(QuaternionStateTest, accessors) {
  ASSERT_TRUE(testState1_.getValue<0>().isNear(testState1_.q_,1e-6));
  ASSERT_TRUE(testState1_.getId<0>() == 0);
}

// Test operator=
TEST_F(QuaternionStateTest, operatorEQ) {
  testState2_ = testState1_;
  ASSERT_TRUE(testState2_.q_.isNear(testState1_.q_,1e-6));
}

// Test createDefaultNames
TEST_F(QuaternionStateTest, naming) {
  testState1_.createDefaultNames("test");
  ASSERT_TRUE(testState1_.name_ == "test");
}

// The fixture for testing class VectorState
class NormalVectorStateTest : public virtual ::testing::Test {
 protected:
  NormalVectorStateTest() {
    testState1_.setRandom(1);
    testState2_.setRandom(2);
  }
  virtual ~NormalVectorStateTest() {
  }
  LWF::NormalVectorState testState1_;
  LWF::NormalVectorState testState2_;
  LWF::NormalVectorState::mtDifVec difVec_;
};

// Test constructors
TEST_F(NormalVectorStateTest, constructor) {
  LWF::NormalVectorState testState1;
}

// Test setIdentity and Identity
TEST_F(NormalVectorStateTest, setIdentity) {
  testState1_.setIdentity();
  ASSERT_TRUE(testState1_.n_ == Eigen::Vector3d(1,0,0));
  ASSERT_TRUE(LWF::NormalVectorState::Identity().n_ == Eigen::Vector3d(1,0,0));
}

// Test plus and minus
TEST_F(NormalVectorStateTest, plusAndMinus) {
  testState2_.boxMinus(testState1_,difVec_);
  LWF::NormalVectorState testState3;
  testState1_.boxPlus(difVec_,testState3);
  ASSERT_NEAR((testState2_.n_-testState3.n_).norm(),0.0,1e-6);
}

// Test getValue, getId
TEST_F(NormalVectorStateTest, accessors) {
  ASSERT_TRUE(testState1_.getValue<0>() == testState1_.n_);
  ASSERT_TRUE(testState1_.getId<0>() == 0);
}

// Test operator=
TEST_F(NormalVectorStateTest, operatorEQ) {
  testState2_ = testState1_;
  ASSERT_TRUE(testState2_.n_ == testState1_.n_);
}

// Test createDefaultNames
TEST_F(NormalVectorStateTest, naming) {
  testState1_.createDefaultNames("test");
  ASSERT_TRUE(testState1_.name_ == "test");
}

// Test getTwoNormals
TEST_F(NormalVectorStateTest, getTwoNormals) {
  Eigen::Vector3d m0;
  Eigen::Vector3d m1;
  testState1_.getTwoNormals(m0,m1);
  ASSERT_NEAR(m0.dot(testState1_.n_),0.0,1e-6);
  ASSERT_NEAR(m1.dot(testState1_.n_),0.0,1e-6);
}

// Test derivative of boxplus
TEST_F(NormalVectorStateTest, derivative) {
  const double d  = 1e-6;
  Eigen::Vector3d m0;
  Eigen::Vector3d m1;
  testState1_.getTwoNormals(m0,m1);
  difVec_.setZero();
  difVec_(0) = d;
  testState1_.boxPlus(difVec_,testState2_);
  ASSERT_NEAR(((testState2_.n_-testState1_.n_)/d-(-m1)).norm(),0.0,1e-6);
  difVec_.setZero();
  difVec_(1) = d;
  testState1_.boxPlus(difVec_,testState2_);
  ASSERT_NEAR(((testState2_.n_-testState1_.n_)/d-m0).norm(),0.0,1e-6);
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
class StateTesting : public virtual ::testing::Test {
 protected:
  enum StateNames{
    s0, s1, s2, s3,
    v0, v1, v2,
    q0, q1
  };
  StateTesting() {
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
      testState1_.getState<0>().array_[i].s_ = testScalar1_[i];
    }
    for(int i=0;i<V_;i++){
      testState1_.getState<1>().array_[i].v_ = testVector1_[i];
    }
    for(int i=0;i<Q_;i++){
      testState1_.getState<2>().array_[i].q_ = testQuat1_[i];
    }
    for(int i=0;i<S_;i++){
      testState2_.getState<0>().array_[i].s_ = testScalar2_[i];
    }
    for(int i=0;i<V_;i++){
      testState2_.getState<1>().array_[i].v_ = testVector2_[i];
    }
    for(int i=0;i<Q_;i++){
      testState2_.getState<2>().array_[i].q_ = testQuat2_[i];
    }
    testState1_.aux().x_ = 2.3;
    testState2_.aux().x_ = 3.2;
  }
  virtual ~StateTesting() {
  }
  static const unsigned int S_ = 4;
  static const unsigned int V_ = 3;
  static const unsigned int Q_ = 2;
  typedef LWF::AugmentedState<LWF::ComposedState<LWF::StateArray<LWF::ScalarState,S_>,LWF::StateArray<LWF::VectorState<3>,V_>,LWF::StateArray<LWF::QuaternionState,Q_>>,Auxillary> StateSVQAugmented;
  StateSVQAugmented testState1_;
  StateSVQAugmented testState2_;
  StateSVQAugmented::mtDifVec difVec_;
  double testScalar1_[S_];
  double testScalar2_[S_];
  Eigen::Vector3d testVector1_[V_];
  Eigen::Vector3d testVector2_[V_];
  rot::RotationQuaternionPD testQuat1_[Q_];
  rot::RotationQuaternionPD testQuat2_[Q_];
};

// Test constructors
TEST_F(StateTesting, constructors) {
  StateSVQAugmented testState1;
  ASSERT_EQ(testState1.aux().x_,1.0);
  StateSVQAugmented testState2(testState2_);
  testState2.boxMinus(testState2_,difVec_);
  ASSERT_NEAR(difVec_.norm(),0.0,1e-6);
  ASSERT_EQ(testState2.aux().x_,3.2);
}

// Test setIdentity and Identity
TEST_F(StateTesting, setIdentity) {
  testState1_.setIdentity();
  for(int i=0;i<S_;i++){
    ASSERT_EQ(testState1_.getState<0>().array_[i].s_,0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(testState1_.getState<1>().array_[i].v_(0),0.0);
    ASSERT_EQ(testState1_.getState<1>().array_[i].v_(1),0.0);
    ASSERT_EQ(testState1_.getState<1>().array_[i].v_(2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(testState1_.getState<2>().array_[i].q_.w(),1.0);
    ASSERT_EQ(testState1_.getState<2>().array_[i].q_.x(),0.0);
    ASSERT_EQ(testState1_.getState<2>().array_[i].q_.y(),0.0);
    ASSERT_EQ(testState1_.getState<2>().array_[i].q_.z(),0.0);
  }
  for(int i=0;i<S_;i++){
    ASSERT_EQ(StateSVQAugmented::Identity().getState<0>().array_[i].s_,0.0);
  }
  for(int i=0;i<V_;i++){
    ASSERT_EQ(StateSVQAugmented::Identity().getState<1>().array_[i].v_(0),0.0);
    ASSERT_EQ(StateSVQAugmented::Identity().getState<1>().array_[i].v_(1),0.0);
    ASSERT_EQ(StateSVQAugmented::Identity().getState<1>().array_[i].v_(2),0.0);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_EQ(StateSVQAugmented::Identity().getState<2>().array_[i].q_.w(),1.0);
    ASSERT_EQ(StateSVQAugmented::Identity().getState<2>().array_[i].q_.x(),0.0);
    ASSERT_EQ(StateSVQAugmented::Identity().getState<2>().array_[i].q_.y(),0.0);
    ASSERT_EQ(StateSVQAugmented::Identity().getState<2>().array_[i].q_.z(),0.0);
  }
}

// Test plus and minus
TEST_F(StateTesting, plusAndMinus) {
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
    ASSERT_NEAR(testState2_.getState<0>().array_[i].s_,testScalar2_[i],1e-6);
  }
  for(int i=0;i<V_;i++){
    ASSERT_NEAR((testState2_.getState<1>().array_[i].v_-testVector2_[i]).norm(),0.0,1e-6);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState2_.getState<2>().array_[i].q_.isNear(testQuat2_[i],1e-6));
  }
  ASSERT_EQ(testState2_.aux().x_,testState1_.aux().x_);
}

// Test getValue, getId, and auxillary accessor
TEST_F(StateTesting, accessors) {
  ASSERT_TRUE(testState1_.getValue<s0>() == testScalar1_[0]);
  ASSERT_TRUE(testState1_.getValue<s1>() == testScalar1_[1]);
  ASSERT_TRUE(testState1_.getValue<s2>() == testScalar1_[2]);
  ASSERT_TRUE(testState1_.getValue<s3>() == testScalar1_[3]);
  ASSERT_NEAR((testState1_.getValue<v0>()-testVector1_[0]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState1_.getValue<v1>()-testVector1_[1]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState1_.getValue<v2>()-testVector1_[2]).norm(),0.0,1e-6);
  ASSERT_TRUE(testState1_.getValue<q0>().isNear(testQuat1_[0],1e-6));
  ASSERT_TRUE(testState1_.getValue<q1>().isNear(testQuat1_[1],1e-6));

  ASSERT_TRUE(testState1_.getId<s0>() == 0);
  ASSERT_TRUE(testState1_.getId<s1>() == 1);
  ASSERT_TRUE(testState1_.getId<s2>() == 2);
  ASSERT_TRUE(testState1_.getId<s3>() == 3);
  ASSERT_TRUE(testState1_.getId<v0>() == 4);
  ASSERT_TRUE(testState1_.getId<v1>() == 7);
  ASSERT_TRUE(testState1_.getId<v2>() == 10);
  ASSERT_TRUE(testState1_.getId<q0>() == 13);
  ASSERT_TRUE(testState1_.getId<q1>() == 16);

  ASSERT_EQ(testState1_.aux().x_,2.3);
}

// Test operator=
TEST_F(StateTesting, operatorEQ) {
  testState2_ = testState1_;
  for(int i=0;i<S_;i++){
    ASSERT_NEAR(testState2_.getState<0>().array_[i].s_,testScalar1_[i],1e-6);
  }
  for(int i=0;i<V_;i++){
    ASSERT_NEAR((testState2_.getState<1>().array_[i].v_-testVector1_[i]).norm(),0.0,1e-6);
  }
  for(int i=0;i<Q_;i++){
    ASSERT_TRUE(testState2_.getState<2>().array_[i].q_.isNear(testQuat1_[i],1e-6));
  }
  ASSERT_EQ(testState2_.aux().x_,testState1_.aux().x_);
}

// Test createDefaultNames
TEST_F(StateTesting, naming) {
  testState1_.createDefaultNames("test");
  ASSERT_TRUE(testState1_.name_ == "test");
  ASSERT_TRUE(testState1_.getState<0>().name_ == "test_0");
  ASSERT_TRUE(testState1_.getState<1>().name_ == "test_1");
  ASSERT_TRUE(testState1_.getState<2>().name_ == "test_2");
  ASSERT_TRUE(testState1_.getState<0>().getState<0>().name_ == "test_0_0");
  ASSERT_TRUE(testState1_.getState<0>().getState<1>().name_ == "test_0_1");
  ASSERT_TRUE(testState1_.getState<0>().getState<2>().name_ == "test_0_2");
  ASSERT_TRUE(testState1_.getState<0>().getState<3>().name_ == "test_0_3");
  ASSERT_TRUE(testState1_.getState<1>().getState<0>().name_ == "test_1_0");
  ASSERT_TRUE(testState1_.getState<1>().getState<1>().name_ == "test_1_1");
  ASSERT_TRUE(testState1_.getState<1>().getState<2>().name_ == "test_1_2");
  ASSERT_TRUE(testState1_.getState<2>().getState<0>().name_ == "test_2_0");
  ASSERT_TRUE(testState1_.getState<2>().getState<1>().name_ == "test_2_1");
}

// Test ZeroArray
TEST_F(StateTesting, ZeroArray) {
  LWF::StateArray<LWF::ScalarState,0> testState1;
}

// Test Constness
TEST_F(StateTesting, Constness) {
  const StateSVQAugmented testState1(testState1_);
  std::cout << testState1.getValue<q1>() << std::endl;
}

// The fixture for testing class StateSVQ
class StateTesting2 : public virtual ::testing::Test {
 protected:
  enum StateNames{
    s0, s1, s2, s3,
    v0, v1, v2,
    q0, q1
  };
  StateTesting2() {
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
    testState1_.s<0>() = testScalar1_[0];
    testState1_.s<1>() = testScalar1_[1];
    testState1_.s<2>() = testScalar1_[2];
    testState1_.s<3>() = testScalar1_[3];
    testState1_.v<0>() = testVector1_[0];
    testState1_.v<1>() = testVector1_[1];
    testState1_.v<2>() = testVector1_[2];
    testState1_.q<0>() = testQuat1_[0];
    testState1_.q<1>() = testQuat1_[1];
    testState2_.s<0>() = testScalar2_[0];
    testState2_.s<1>() = testScalar2_[1];
    testState2_.s<2>() = testScalar2_[2];
    testState2_.s<3>() = testScalar2_[3];
    testState2_.v<0>() = testVector2_[0];
    testState2_.v<1>() = testVector2_[1];
    testState2_.v<2>() = testVector2_[2];
    testState2_.q<0>() = testQuat2_[0];
    testState2_.q<1>() = testQuat2_[1];
    testState1_.aux().x_ = 2.3;
    testState2_.aux().x_ = 3.2;
  }
  virtual ~StateTesting2() {
  }
  static const unsigned int S_ = 4;
  static const unsigned int V_ = 3;
  static const unsigned int Q_ = 2;
  typedef LWF::AugmentedState<LWF::StateSVQ2<S_,V_,Q_>,Auxillary> StateSVQAugmented;
  StateSVQAugmented testState1_;
  StateSVQAugmented testState2_;
  StateSVQAugmented::mtDifVec difVec_;
  double testScalar1_[S_];
  double testScalar2_[S_];
  Eigen::Vector3d testVector1_[V_];
  Eigen::Vector3d testVector2_[V_];
  rot::RotationQuaternionPD testQuat1_[Q_];
  rot::RotationQuaternionPD testQuat2_[Q_];
};

// Test constructors
TEST_F(StateTesting2, constructors) {
  StateSVQAugmented testState1;
  ASSERT_EQ(testState1.aux().x_,1.0);
  StateSVQAugmented testState2(testState2_);
  testState2.boxMinus(testState2_,difVec_);
  ASSERT_NEAR(difVec_.norm(),0.0,1e-6);
  ASSERT_EQ(testState2.aux().x_,3.2);
}

// Test setIdentity and Identity
TEST_F(StateTesting2, setIdentity) {
  testState1_.setIdentity();
  ASSERT_EQ(testState1_.s<0>(),0.0);
  ASSERT_EQ(testState1_.s<1>(),0.0);
  ASSERT_EQ(testState1_.s<2>(),0.0);
  ASSERT_EQ(testState1_.s<3>(),0.0);
  ASSERT_EQ(testState1_.v<0>().norm(),0.0);
  ASSERT_EQ(testState1_.v<1>().norm(),0.0);
  ASSERT_EQ(testState1_.v<2>().norm(),0.0);
  ASSERT_EQ(testState1_.q<0>().boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(testState1_.q<1>().boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().s<0>(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().s<1>(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().s<2>(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().s<3>(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().v<0>().norm(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().v<1>().norm(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().v<2>().norm(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().q<0>().boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(StateSVQAugmented::Identity().q<1>().boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
}

// Test plus and minus
TEST_F(StateTesting2, plusAndMinus) {
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
  ASSERT_NEAR(testState2_.s<0>(),testScalar2_[0],1e-6);
  ASSERT_NEAR(testState2_.s<1>(),testScalar2_[1],1e-6);
  ASSERT_NEAR(testState2_.s<2>(),testScalar2_[2],1e-6);
  ASSERT_NEAR(testState2_.s<3>(),testScalar2_[3],1e-6);
  ASSERT_NEAR((testState2_.v<0>()-testVector2_[0]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState2_.v<1>()-testVector2_[1]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState2_.v<2>()-testVector2_[2]).norm(),0.0,1e-6);
  ASSERT_TRUE(testState2_.q<0>().isNear(testQuat2_[0],1e-6));
  ASSERT_TRUE(testState2_.q<1>().isNear(testQuat2_[1],1e-6));
  ASSERT_EQ(testState2_.aux().x_,testState1_.aux().x_);
}

// Test getValue, getId, and auxillary accessor
TEST_F(StateTesting2, accessors) {
  ASSERT_TRUE(testState1_.getValue<s0>() == testScalar1_[0]);
  ASSERT_TRUE(testState1_.getValue<s1>() == testScalar1_[1]);
  ASSERT_TRUE(testState1_.getValue<s2>() == testScalar1_[2]);
  ASSERT_TRUE(testState1_.getValue<s3>() == testScalar1_[3]);
  ASSERT_NEAR((testState1_.getValue<v0>()-testVector1_[0]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState1_.getValue<v1>()-testVector1_[1]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState1_.getValue<v2>()-testVector1_[2]).norm(),0.0,1e-6);
  ASSERT_TRUE(testState1_.getValue<q0>().isNear(testQuat1_[0],1e-6));
  ASSERT_TRUE(testState1_.getValue<q1>().isNear(testQuat1_[1],1e-6));

  ASSERT_TRUE(testState1_.getId<s0>() == 0);
  ASSERT_TRUE(testState1_.getId<s1>() == 1);
  ASSERT_TRUE(testState1_.getId<s2>() == 2);
  ASSERT_TRUE(testState1_.getId<s3>() == 3);
  ASSERT_TRUE(testState1_.getId<v0>() == 4);
  ASSERT_TRUE(testState1_.getId<v1>() == 7);
  ASSERT_TRUE(testState1_.getId<v2>() == 10);
  ASSERT_TRUE(testState1_.getId<q0>() == 13);
  ASSERT_TRUE(testState1_.getId<q1>() == 16);

  ASSERT_EQ(testState1_.aux().x_,2.3);
}

// Test operator=
TEST_F(StateTesting2, operatorEQ) {
  testState2_ = testState1_;
  ASSERT_TRUE(testState2_.getValue<s0>() == testScalar1_[0]);
  ASSERT_TRUE(testState2_.getValue<s1>() == testScalar1_[1]);
  ASSERT_TRUE(testState2_.getValue<s2>() == testScalar1_[2]);
  ASSERT_TRUE(testState2_.getValue<s3>() == testScalar1_[3]);
  ASSERT_NEAR((testState2_.getValue<v0>()-testVector1_[0]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState2_.getValue<v1>()-testVector1_[1]).norm(),0.0,1e-6);
  ASSERT_NEAR((testState2_.getValue<v2>()-testVector1_[2]).norm(),0.0,1e-6);
  ASSERT_TRUE(testState2_.getValue<q0>().isNear(testQuat1_[0],1e-6));
  ASSERT_TRUE(testState2_.getValue<q1>().isNear(testQuat1_[1],1e-6));
  ASSERT_EQ(testState2_.aux().x_,testState1_.aux().x_);
}

// Test createDefaultNames
TEST_F(StateTesting2, naming) {
  testState1_.createDefaultNames("test");
  ASSERT_TRUE(testState1_.name_ == "test");
  ASSERT_TRUE(std::get<s0>(testState1_.mStates_).name_ == "test_0");
  ASSERT_TRUE(std::get<s1>(testState1_.mStates_).name_ == "test_1");
  ASSERT_TRUE(std::get<s2>(testState1_.mStates_).name_ == "test_2");
  ASSERT_TRUE(std::get<s3>(testState1_.mStates_).name_ == "test_3");
  ASSERT_TRUE(std::get<v0>(testState1_.mStates_).name_ == "test_4");
  ASSERT_TRUE(std::get<v1>(testState1_.mStates_).name_ == "test_5");
  ASSERT_TRUE(std::get<v2>(testState1_.mStates_).name_ == "test_6");
  ASSERT_TRUE(std::get<q0>(testState1_.mStates_).name_ == "test_7");
  ASSERT_TRUE(std::get<q1>(testState1_.mStates_).name_ == "test_8");

//  TODO
  LWF::ArrayElement<LWF::ScalarElement,4> test;
  test.get(0) = 4.3;
  test.get(1) = 1.3;
  test.get(2) = 2.3;
  test.get(3) = 3.3;
  test.print();
}

// Test ZeroArray
TEST_F(StateTesting2, ZeroArray) {
  LWF::StateArray<LWF::ScalarState,0> testState1;
}

// Test Constness
TEST_F(StateTesting2, Constness) {
  const StateSVQAugmented testState1(testState1_);
  std::cout << testState1.getValue<q1>() << std::endl;
}

// Test LMat
TEST_F(StateTesting, LMat) {
  double d = 0.00001;
  LWF::StateSVQ<0,0,1> att;
  LWF::StateSVQ<0,1,0> vec;
  vec.v(0) = Eigen::Vector3d(0.4,-0.2,1.7);
  Eigen::Matrix3d J;
  LWF::StateSVQ<0,0,1> attDisturbed;
  LWF::StateSVQ<0,1,0> vecDisturbed;
  Eigen::Matrix3d I;
  I.setIdentity();
  Eigen::Vector3d dif;
  I = d*I;
  att.q(0) = att.q(0).exponentialMap(vec.v(0));
  for(unsigned int i=0;i<3;i++){
    vec.boxPlus(I.col(i),vecDisturbed);
    attDisturbed.q(0) = attDisturbed.q(0).exponentialMap(vecDisturbed.v(0));
    attDisturbed.boxMinus(att,dif);
    J.col(i) = dif*1/d;
  }
  ASSERT_NEAR((J-LWF::Lmat(vec.v(0))).norm(),0.0,1e-5);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
