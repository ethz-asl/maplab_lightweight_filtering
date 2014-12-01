#include "State.hpp"
#include "gtest/gtest.h"
#include <assert.h>

// The fixture for testing class ScalarState
class ScalarElementTest : public virtual ::testing::Test {
 protected:
  ScalarElementTest() {
    unsigned int s = 1;
    testElement1_.setRandom(s);
    testElement2_.setRandom(s);
  }
  virtual ~ScalarElementTest() {
  }
  LWF::ScalarElement testElement1_;
  LWF::ScalarElement testElement2_;
  LWF::ScalarElement::mtDifVec difVec_;
};

// Test constructors
TEST_F(ScalarElementTest, constructor) {
  LWF::ScalarElement testElement1;
}

// Test setIdentity and Identity
TEST_F(ScalarElementTest, setIdentity) {
  testElement1_.setIdentity();
  ASSERT_EQ(testElement1_.s_,0.0);
  ASSERT_EQ(LWF::ScalarElement::Identity().s_,0.0);
}

// Test plus and minus
TEST_F(ScalarElementTest, plusAndMinus) {
  testElement2_.boxMinus(testElement1_,difVec_);
  ASSERT_EQ(difVec_(0),testElement2_.s_-testElement1_.s_);
  LWF::ScalarElement testElement3;
  testElement1_.boxPlus(difVec_,testElement3);
  ASSERT_NEAR(testElement2_.s_,testElement3.s_,1e-6);
}

// Test getValue
TEST_F(ScalarElementTest, accessors) {
  ASSERT_TRUE(testElement1_.get() == testElement1_.s_);
}

// Test operator=
TEST_F(ScalarElementTest, operatorEQ) {
  testElement2_ = testElement1_;
  ASSERT_NEAR(testElement2_.s_,testElement1_.s_,1e-6);
}

// The fixture for testing class VectorState
class VectorElementTest : public virtual ::testing::Test {
 protected:
  VectorElementTest() {
    unsigned int s = 1;
    testElement1_.setRandom(s);
    testElement2_.setRandom(s);
  }
  virtual ~VectorElementTest() {
  }
  static const unsigned int N_ = 4;
  LWF::VectorElement<N_> testElement1_;
  LWF::VectorElement<N_> testElement2_;
  LWF::VectorElement<N_>::mtDifVec difVec_;
};

// Test constructors
TEST_F(VectorElementTest, constructor) {
  LWF::VectorElement<N_> testElement1;
}

// Test setIdentity and Identity
TEST_F(VectorElementTest, setIdentity) {
  testElement1_.setIdentity();
  ASSERT_EQ(testElement1_.v_.norm(),0.0);
  ASSERT_EQ(LWF::VectorElement<N_>::Identity().v_.norm(),0.0);
}

// Test plus and minus
TEST_F(VectorElementTest, plusAndMinus) {
  testElement2_.boxMinus(testElement1_,difVec_);
  ASSERT_NEAR((difVec_-(testElement2_.v_-testElement1_.v_)).norm(),0.0,1e-6);
  LWF::VectorElement<N_> testElement3;
  testElement1_.boxPlus(difVec_,testElement3);
  ASSERT_NEAR((testElement2_.v_-testElement3.v_).norm(),0.0,1e-6);
}

// Test getValue, getId
TEST_F(VectorElementTest, accessors) {
  ASSERT_TRUE(testElement1_.get() == testElement1_.v_);
}

// Test operator=
TEST_F(VectorElementTest, operatorEQ) {
  testElement2_ = testElement1_;
  ASSERT_NEAR((testElement2_.v_-testElement1_.v_).norm(),0.0,1e-6);
}

// The fixture for testing class QuaternionElementTest
class QuaternionElementTest : public virtual ::testing::Test {
 protected:
  QuaternionElementTest() {
    unsigned int s = 1;
    testElement1_.setRandom(s);
    testElement2_.setRandom(s);
  }
  virtual ~QuaternionElementTest() {
  }
  LWF::QuaternionElement testElement1_;
  LWF::QuaternionElement testElement2_;
  LWF::QuaternionElement::mtDifVec difVec_;
};

// Test constructors
TEST_F(QuaternionElementTest, constructor) {
  LWF::QuaternionElement testElement1;
}

// Test setIdentity and Identity
TEST_F(QuaternionElementTest, setIdentity) {
  testElement1_.setIdentity();
  ASSERT_TRUE(testElement1_.q_.isNear(rot::RotationQuaternionPD(),1e-6));
  ASSERT_TRUE(LWF::QuaternionElement::Identity().q_.isNear(rot::RotationQuaternionPD(),1e-6));
}

// Test plus and minus
TEST_F(QuaternionElementTest, plusAndMinus) {
  testElement2_.boxMinus(testElement1_,difVec_);
  ASSERT_NEAR((difVec_-testElement2_.q_.boxMinus(testElement1_.q_)).norm(),0.0,1e-6);
  LWF::QuaternionElement testElement3;
  testElement1_.boxPlus(difVec_,testElement3);
  ASSERT_TRUE(testElement2_.q_.isNear(testElement3.q_,1e-6));
}

// Test getValue, getId
TEST_F(QuaternionElementTest, accessors) {
  ASSERT_TRUE(testElement1_.get().isNear(testElement1_.q_,1e-6));
}

// Test operator=
TEST_F(QuaternionElementTest, operatorEQ) {
  testElement2_ = testElement1_;
  ASSERT_TRUE(testElement2_.q_.isNear(testElement1_.q_,1e-6));
}

// Test LMat
TEST_F(QuaternionElementTest, LMat) {
  double d = 0.00001;
  LWF::QuaternionElement att;
  LWF::VectorElement<3> vec;
  vec.v_ = Eigen::Vector3d(0.4,-0.2,1.7);
  Eigen::Matrix3d J;
  LWF::QuaternionElement attDisturbed;
  LWF::VectorElement<3> vecDisturbed;
  Eigen::Matrix3d I;
  I.setIdentity();
  Eigen::Vector3d dif;
  I = d*I;
  att.q_ = att.q_.exponentialMap(vec.v_);
  for(unsigned int i=0;i<3;i++){
    vec.boxPlus(I.col(i),vecDisturbed);
    attDisturbed.q_ = attDisturbed.q_.exponentialMap(vecDisturbed.v_);
    attDisturbed.boxMinus(att,dif);
    J.col(i) = dif*1/d;
  }
  ASSERT_NEAR((J-LWF::Lmat(vec.v_)).norm(),0.0,1e-5);
}

// The fixture for testing class NormalVectorElementTest
class NormalVectorElementTest : public virtual ::testing::Test {
 protected:
  NormalVectorElementTest() {
    unsigned int s = 1;
    testElement1_.setRandom(s);
    testElement2_.setRandom(s);
  }
  virtual ~NormalVectorElementTest() {
  }
  LWF::NormalVectorElement testElement1_;
  LWF::NormalVectorElement testElement2_;
  LWF::NormalVectorElement::mtDifVec difVec_;
};

// Test constructors
TEST_F(NormalVectorElementTest, constructor) {
  LWF::NormalVectorElement testElement1;
}

// Test setIdentity and Identity
TEST_F(NormalVectorElementTest, setIdentity) {
  testElement1_.setIdentity();
  ASSERT_TRUE(testElement1_.n_ == Eigen::Vector3d(1,0,0));
  ASSERT_TRUE(LWF::NormalVectorElement::Identity().n_ == Eigen::Vector3d(1,0,0));
}

// Test plus and minus
TEST_F(NormalVectorElementTest, plusAndMinus) {
  testElement2_.boxMinus(testElement1_,difVec_);
  LWF::NormalVectorElement testElement3;
  testElement1_.boxPlus(difVec_,testElement3);
  ASSERT_NEAR((testElement2_.n_-testElement3.n_).norm(),0.0,1e-6);
}

// Test getValue, getId
TEST_F(NormalVectorElementTest, accessors) {
  ASSERT_TRUE(testElement1_.get() == testElement1_.n_);
}

// Test operator=
TEST_F(NormalVectorElementTest, operatorEQ) {
  testElement2_ = testElement1_;
  ASSERT_TRUE(testElement2_.n_ == testElement1_.n_);
}

// Test getTwoNormals
TEST_F(NormalVectorElementTest, getTwoNormals) {
  Eigen::Vector3d m0;
  Eigen::Vector3d m1;
  testElement1_.getTwoNormals(m0,m1);
  ASSERT_NEAR(m0.dot(testElement1_.n_),0.0,1e-6);
  ASSERT_NEAR(m1.dot(testElement1_.n_),0.0,1e-6);
}

// Test derivative of boxplus
TEST_F(NormalVectorElementTest, derivative) {
  const double d  = 1e-6;
  Eigen::Vector3d m0;
  Eigen::Vector3d m1;
  testElement1_.getTwoNormals(m0,m1);
  difVec_.setZero();
  difVec_(0) = d;
  testElement1_.boxPlus(difVec_,testElement2_);
  ASSERT_NEAR(((testElement2_.n_-testElement1_.n_)/d-(-m1)).norm(),0.0,1e-6);
  difVec_.setZero();
  difVec_(1) = d;
  testElement1_.boxPlus(difVec_,testElement2_);
  ASSERT_NEAR(((testElement2_.n_-testElement1_.n_)/d-m0).norm(),0.0,1e-6);
}

// The fixture for testing class ArrayElementTest
class ArrayElementTest : public virtual ::testing::Test {
 protected:
  ArrayElementTest() {
    unsigned int s = 1;
    testElement1_.setRandom(s);
    testElement2_.setRandom(s);
  }
  virtual ~ArrayElementTest() {
  }
  static const unsigned int N_ = 5;
  LWF::ArrayElement<LWF::QuaternionElement,N_> testElement1_;
  LWF::ArrayElement<LWF::QuaternionElement,N_> testElement2_;
  LWF::ArrayElement<LWF::QuaternionElement,N_>::mtDifVec difVec_;
};

// Test constructors
TEST_F(ArrayElementTest, constructor) {
  LWF::ArrayElement<LWF::QuaternionElement,N_> testElement1;
}

// Test setIdentity and Identity
TEST_F(ArrayElementTest, setIdentity) {
  testElement1_.setIdentity();
  for(unsigned int i=0;i<N_;i++){
    ASSERT_TRUE(testElement1_.array_[i].q_.isNear(rot::RotationQuaternionPD(),1e-6));
    ASSERT_TRUE((LWF::ArrayElement<LWF::QuaternionElement,N_>::Identity().array_[i].q_.isNear(rot::RotationQuaternionPD(),1e-6)));
  }
}

// Test plus and minus
TEST_F(ArrayElementTest, plusAndMinus) {
  testElement2_.boxMinus(testElement1_,difVec_);
  for(unsigned int i=0;i<N_;i++){
    ASSERT_NEAR((difVec_.block<3,1>(i*3,0)-testElement2_.array_[i].q_.boxMinus(testElement1_.array_[i].q_)).norm(),0.0,1e-6);
  }
  LWF::ArrayElement<LWF::QuaternionElement,N_> testElement3;
  testElement1_.boxPlus(difVec_,testElement3);
  for(unsigned int i=0;i<N_;i++){
    ASSERT_TRUE(testElement2_.array_[i].q_.isNear(testElement3.array_[i].q_,1e-6));
  }
}

// Test getValue, getId
TEST_F(ArrayElementTest, accessors) {
  for(unsigned int i=0;i<N_;i++){
    ASSERT_TRUE(testElement1_.get(i).isNear(testElement1_.array_[i].q_,1e-6));
  }
}

// Test operator=
TEST_F(ArrayElementTest, operatorEQ) {
  testElement2_ = testElement1_;
  for(unsigned int i=0;i<N_;i++){
    ASSERT_TRUE(testElement2_.array_[i].q_.isNear(testElement1_.array_[i].q_,1e-6));
  }
}

class AuxillaryElement: public LWF::AuxiliaryBase<AuxillaryElement>{
 public:
  AuxillaryElement(){
    x_ = 1.0;
  };
  ~AuxillaryElement(){};
  double x_;
};

// The fixture for testing class StateSVQ
class StateTesting : public virtual ::testing::Test {
 protected:
  static const unsigned int _sca = 0;
  static const unsigned int _vec0 = _sca+1;
  static const unsigned int _vec1 = _vec0+1;
  static const unsigned int _vec2 = _vec1+1;
  static const unsigned int _vec3 = _vec2+1;
  static const unsigned int _qua0 = _vec3+1;
  static const unsigned int _qua1 = _qua0+1;
  static const unsigned int _aux = _qua1+1;
  StateTesting() {
    testScalar1_ = 4.5;
    testScalar2_ = -17.34;
    testVector1_[0] << 2.1, -0.2, -1.9;
    testVector2_[0] << -10.6, 0.2, -105.2;
    for(int i=1;i<4;i++){
      testVector1_[i] = testVector1_[i-1] + Eigen::Vector3d(0.3,10.9,2.3);
      testVector2_[i] = testVector2_[i-1] + Eigen::Vector3d(-1.5,12,1785.23);
    }
    testQuat1_[0] = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testQuat2_[0] = rot::RotationQuaternionPD(0.0,0.36,0.48,0.8);
    for(int i=1;i<4;i++){
      testQuat1_[i] = testQuat1_[i-1].boxPlus(testVector1_[i-1]);
      testQuat2_[i] = testQuat2_[i-1].boxPlus(testVector2_[i-1]);
    }
    testState1_.get<_sca>() = testScalar1_;
    testState1_.get<_vec0>() = testVector1_[0];
    testState1_.get<_vec1>() = testVector1_[1];
    testState1_.get<_vec2>() = testVector1_[2];
    testState1_.get<_vec3>() = testVector1_[3];
    testState1_.get<_qua0>(0) = testQuat1_[0];
    testState1_.get<_qua0>(1) = testQuat1_[1];
    testState1_.get<_qua1>(0) = testQuat1_[2];
    testState1_.get<_qua1>(1) = testQuat1_[3];
    testState1_.get<_aux>().x_ = 2.3;
    testState2_.get<_sca>() = testScalar2_;
    testState2_.get<_vec0>() = testVector2_[0];
    testState2_.get<_vec1>() = testVector2_[1];
    testState2_.get<_vec2>() = testVector2_[2];
    testState2_.get<_vec3>() = testVector2_[3];
    testState2_.get<_qua0>(0) = testQuat2_[0];
    testState2_.get<_qua0>(1) = testQuat2_[1];
    testState2_.get<_qua1>(0) = testQuat2_[2];
    testState2_.get<_qua1>(1) = testQuat2_[3];
    testState2_.get<_aux>().x_ = 3.2;
  }
  virtual ~StateTesting() {
  }
  typedef LWF::State<
      LWF::ScalarElement,
      LWF::TH_multiple_elements<LWF::VectorElement<3>,4>,
      LWF::TH_multiple_elements<LWF::ArrayElement<LWF::QuaternionElement,2>,2>,
      AuxillaryElement> mtState;
  mtState testState1_;
  mtState testState2_;
  mtState::mtDifVec difVec_;
  double testScalar1_;
  double testScalar2_;
  Eigen::Vector3d testVector1_[4];
  Eigen::Vector3d testVector2_[4];
  rot::RotationQuaternionPD testQuat1_[4];
  rot::RotationQuaternionPD testQuat2_[4];
};

// Test constructors
TEST_F(StateTesting, constructors) {
  mtState testState1;
  ASSERT_EQ(testState1.get<_aux>().x_,1.0);
  mtState testState2(testState2_);
  testState2.boxMinus(testState2_,difVec_);
  ASSERT_NEAR(difVec_.norm(),0.0,1e-6);
  ASSERT_EQ(testState2.get<_aux>().x_,3.2);
}

// Test setIdentity and Identity
TEST_F(StateTesting, setIdentity) {
  testState1_.setIdentity();
  ASSERT_EQ(testState1_.get<_sca>(),0);
  ASSERT_EQ(testState1_.get<_vec0>().norm(),0);
  ASSERT_EQ(testState1_.get<_vec1>().norm(),0);
  ASSERT_EQ(testState1_.get<_vec2>().norm(),0);
  ASSERT_EQ(testState1_.get<_vec3>().norm(),0);
  ASSERT_EQ(testState1_.get<_qua0>(0).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(testState1_.get<_qua0>(1).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(testState1_.get<_qua1>(0).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(testState1_.get<_qua1>(1).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(testState1_.get<_aux>().x_,2.3);
  ASSERT_EQ(mtState::Identity().get<_sca>(),0);
  ASSERT_EQ(mtState::Identity().get<_vec0>().norm(),0);
  ASSERT_EQ(mtState::Identity().get<_vec1>().norm(),0);
  ASSERT_EQ(mtState::Identity().get<_vec2>().norm(),0);
  ASSERT_EQ(mtState::Identity().get<_vec3>().norm(),0);
  ASSERT_EQ(mtState::Identity().get<_qua0>(0).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(mtState::Identity().get<_qua0>(1).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(mtState::Identity().get<_qua1>(0).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(mtState::Identity().get<_qua1>(1).boxMinus(rot::RotationQuaternionPD()).norm(),0.0);
  ASSERT_EQ(mtState::Identity().get<_aux>().x_,1.0);
}

// Test plus and minus
TEST_F(StateTesting, plusAndMinus) {
  testState2_.boxMinus(testState1_,difVec_);
  unsigned int index=0;
  ASSERT_EQ(difVec_(index),testScalar2_-testScalar1_);
  index ++;
  for(int i=0;i<4;i++){
    ASSERT_EQ((difVec_.block<3,1>(index,0)-(testVector2_[i]-testVector1_[i])).norm(),0);
    index = index + 3;
  }
  for(int i=0;i<4;i++){
    ASSERT_EQ((difVec_.block<3,1>(index,0)-testQuat2_[i].boxMinus(testQuat1_[i])).norm(),0);
    index = index + 3;
  }
  testState1_.boxPlus(difVec_,testState2_);
  ASSERT_NEAR(testState2_.get<_sca>(),testScalar2_,1e-10);
  ASSERT_NEAR((testState2_.get<_vec0>()-testVector2_[0]).norm(),0,1e-10);
  ASSERT_NEAR((testState2_.get<_vec1>()-testVector2_[1]).norm(),0,1e-10);
  ASSERT_NEAR((testState2_.get<_vec2>()-testVector2_[2]).norm(),0,1e-10);
  ASSERT_NEAR((testState2_.get<_vec3>()-testVector2_[3]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua0>(0).boxMinus(testQuat2_[0]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua0>(1).boxMinus(testQuat2_[1]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua1>(0).boxMinus(testQuat2_[2]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua1>(1).boxMinus(testQuat2_[3]).norm(),0,1e-10);
  ASSERT_EQ(testState2_.get<_aux>().x_,testState1_.get<_aux>().x_);
}

// Test getValue, getId
TEST_F(StateTesting, accessors) {
  ASSERT_NEAR(testState1_.get<_sca>(),testScalar1_,1e-10);
  ASSERT_NEAR((testState1_.get<_vec0>()-testVector1_[0]).norm(),0,1e-10);
  ASSERT_NEAR((testState1_.get<_vec1>()-testVector1_[1]).norm(),0,1e-10);
  ASSERT_NEAR((testState1_.get<_vec2>()-testVector1_[2]).norm(),0,1e-10);
  ASSERT_NEAR((testState1_.get<_vec3>()-testVector1_[3]).norm(),0,1e-10);
  ASSERT_NEAR(testState1_.get<_qua0>(0).boxMinus(testQuat1_[0]).norm(),0,1e-10);
  ASSERT_NEAR(testState1_.get<_qua0>(1).boxMinus(testQuat1_[1]).norm(),0,1e-10);
  ASSERT_NEAR(testState1_.get<_qua1>(0).boxMinus(testQuat1_[2]).norm(),0,1e-10);
  ASSERT_NEAR(testState1_.get<_qua1>(1).boxMinus(testQuat1_[3]).norm(),0,1e-10);
  ASSERT_EQ(testState1_.get<_aux>().x_,2.3);

  ASSERT_TRUE(testState1_.getId<_sca>() == 0);
  ASSERT_TRUE(testState1_.getId<_vec0>() == 1);
  ASSERT_TRUE(testState1_.getId<_vec1>() == 4);
  ASSERT_TRUE(testState1_.getId<_vec2>() == 7);
  ASSERT_TRUE(testState1_.getId<_vec3>() == 10);
  ASSERT_TRUE(testState1_.getId<_qua0>() == 13);
  ASSERT_TRUE(testState1_.getId<_qua1>() == 19);
  ASSERT_TRUE(testState1_.getId<_aux>() == 25);
}

// Test operator=
TEST_F(StateTesting, operatorEQ) {
  testState2_ = testState1_;
  ASSERT_NEAR(testState2_.get<_sca>(),testScalar1_,1e-10);
  ASSERT_NEAR((testState2_.get<_vec0>()-testVector1_[0]).norm(),0,1e-10);
  ASSERT_NEAR((testState2_.get<_vec1>()-testVector1_[1]).norm(),0,1e-10);
  ASSERT_NEAR((testState2_.get<_vec2>()-testVector1_[2]).norm(),0,1e-10);
  ASSERT_NEAR((testState2_.get<_vec3>()-testVector1_[3]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua0>(0).boxMinus(testQuat1_[0]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua0>(1).boxMinus(testQuat1_[1]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua1>(0).boxMinus(testQuat1_[2]).norm(),0,1e-10);
  ASSERT_NEAR(testState2_.get<_qua1>(1).boxMinus(testQuat1_[3]).norm(),0,1e-10);
  ASSERT_EQ(testState2_.get<_aux>().x_,2.3);
}

// Test createDefaultNames
TEST_F(StateTesting, naming) {
  testState1_.createDefaultNames("test");
  ASSERT_TRUE(testState1_.getName<_sca>() == "test_0");
  ASSERT_TRUE(testState1_.getName<_vec0>() == "test_1");
  ASSERT_TRUE(testState1_.getName<_vec1>() == "test_2");
  ASSERT_TRUE(testState1_.getName<_vec2>() == "test_3");
  ASSERT_TRUE(testState1_.getName<_vec3>() == "test_4");
  ASSERT_TRUE(testState1_.getName<_qua0>() == "test_5");
  ASSERT_TRUE(testState1_.getName<_qua1>() == "test_6");
  ASSERT_TRUE(testState1_.getName<_aux>() == "test_7");
}

// Test ZeroArray
TEST_F(StateTesting, ZeroArray) {
  LWF::State<LWF::TH_multiple_elements<LWF::ScalarElement,0>,LWF::QuaternionElement> testState1;
  ASSERT_EQ(std::tuple_size<decltype(testState1.mElements_)>::value,1);
}

// Test Constness
TEST_F(StateTesting, Constness) {
  const mtState testState1(testState1_);
  ASSERT_NEAR(testState1.get<_qua0>(0).boxMinus(testQuat1_[0]).norm(),0,1e-10);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
