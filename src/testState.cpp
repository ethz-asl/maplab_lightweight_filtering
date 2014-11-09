#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

// The fixture for testing class StateSVQ
class ScalarStateTest : public virtual ::testing::Test {
 protected:
  ScalarStateTest() {
    testScalar1_ = 0.1;
    testScalar2_ = -3.4;
    testState1_.s_ = testScalar1_;
    testState2_.s_ = testScalar2_;
  }
  virtual ~ScalarStateTest() {
  }
  LWF::ScalarState testState1_;
  LWF::ScalarState testState2_;
  double testScalar1_;
  double testScalar2_;
};

// Test constructors
TEST_F(ScalarStateTest, constructor) {
  LWF::ScalarState testState1;
  std::default_random_engine generator (-1);
  std::normal_distribution<double> distribution (0.0,1.0);
  std::cout << distribution(generator) << std::endl;
  std::cout << distribution(generator) << std::endl;
  std::cout << distribution(generator) << std::endl;
  std::srand(1);
  std::cout << std::rand() << std::endl;
  std::cout << std::rand() << std::endl;
  std::cout << std::rand() << std::endl;
  std::srand(1);
  std::cout << std::rand() << std::endl;
  std::cout << std::rand() << std::endl;
  std::cout << std::rand() << std::endl;
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
