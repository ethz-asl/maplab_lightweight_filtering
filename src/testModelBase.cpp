#include "State.hpp"
#include "ModelBase.hpp"
#include "gtest/gtest.h"
#include <assert.h>

class Input: public LWF::StateSVQ<0,2,2>{
 public:
  Input(){};
  ~Input(){};
};
class Output: public LWF::StateSVQ<0,1,1>{
 public:
  Output(){};
  ~Output(){};
};
class Meas: public LWF::StateSVQ<0,1,1>{
 public:
  Meas(){};
  ~Meas(){};
};
class Noise: public LWF::VectorState<6>{
 public:
  Noise(){};
  ~Noise(){};
};
class ModelExample: public LWF::ModelBase<Input,Output,Meas,Noise>{
 public:
  ModelExample(){};
  ~ModelExample(){};
  Output eval(const Input& input, const Meas& meas, const Noise noise, const double dt) const{
    Output output;
    output.v(0) = (input.q(0).inverted()*input.q(1)).rotate(input.v(1))-input.v(0)+noise.template block<3>(0)-meas.v(0);
    rot::RotationQuaternionPD dQ = dQ.exponentialMap(noise.template block<3>(3));
    output.q(0) = meas.q(0).inverted()*dQ*input.q(1).inverted()*input.q(0);
    return output;
  }
  mtJacInput jacInput(const Input& input, const Meas& meas, const double dt) const{
    Output output;
    mtJacInput J;
    J.setZero();
    J.block<3,3>(output.getId(output.v(0)),input.getId(input.v(0))) = -Eigen::Matrix3d::Identity();
    J.block<3,3>(output.getId(output.v(0)),input.getId(input.v(1))) = rot::RotationMatrixPD(input.q(0).inverted()*input.q(1)).matrix();
    J.block<3,3>(output.getId(output.v(0)),input.getId(input.q(0))) = -kindr::linear_algebra::getSkewMatrixFromVector((input.q(0).inverted()*input.q(1)).rotate(input.v(1)))*rot::RotationMatrixPD(input.q(0).inverted()).matrix();
    J.block<3,3>(output.getId(output.v(0)),input.getId(input.q(1))) = kindr::linear_algebra::getSkewMatrixFromVector((input.q(0).inverted()*input.q(1)).rotate(input.v(1)))*rot::RotationMatrixPD(input.q(0).inverted()).matrix();
    J.block<3,3>(output.getId(output.q(0)),input.getId(input.q(0))) = rot::RotationMatrixPD(meas.q(0).inverted()*input.q(1).inverted()).matrix();
    J.block<3,3>(output.getId(output.q(0)),input.getId(input.q(1))) = -rot::RotationMatrixPD(meas.q(0).inverted()*input.q(1).inverted()).matrix();
    return J;
  }
  mtJacNoise jacNoise(const Input& input, const Meas& meas, const double dt) const{
    Output output;
    mtJacNoise J;
    J.setZero();
    J.block<3,3>(output.getId(output.v(0)),0) = Eigen::Matrix3d::Identity();
    J.block<3,3>(output.getId(output.q(0)),3) = rot::RotationMatrixPD(meas.q(0).inverted()).matrix();
    return J;
  }
};

// The fixture for testing class PredictionModel
class ModelBaseTest : public ::testing::Test {
 protected:
  ModelBaseTest() {
    testInput_.v(0) = Eigen::Vector3d(2.1,-0.2,-1.9);
    testInput_.v(1) = Eigen::Vector3d(0.3,10.9,2.3);
    testInput_.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testInput_.q(1) = rot::RotationQuaternionPD(0.0,0.36,0.48,0.8);
    testMeas_.v(0) = Eigen::Vector3d(-1.5,12,1785.23);
    testMeas_.q(0) = rot::RotationQuaternionPD(-3.0/sqrt(15.0),1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
  }
  virtual ~ModelBaseTest(){}
  ModelExample model_;
  Input testInput_;
  Meas testMeas_;
};

// Test finite difference Jacobians
TEST_F(ModelBaseTest, FDjacobians) {
  ModelExample::mtJacInput F;
  F = model_.jacInputFD(testInput_,testMeas_,0.1,0.0000001);
  ASSERT_NEAR((F-model_.jacInput(testInput_,testMeas_,0.1)).norm(),0.0,1e-5);
  ModelExample::mtJacNoise Fn;
  Fn = model_.jacNoiseFD(testInput_,testMeas_,0.1,0.0000001);
  ASSERT_NEAR((Fn-model_.jacNoise(testInput_,testMeas_,0.1)).norm(),0.0,1e-5);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
