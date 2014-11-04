#include "State.hpp"
#include "ModelBase.hpp"
#include "gtest/gtest.h"
#include <assert.h>

class Input: public LWF::StateSVQ<0,2,2>{
 public:
  enum StateNames{
    v0, v1, q0, q1
  };
  Input(){};
  ~Input(){};
};
class Output: public LWF::StateSVQ<0,1,1>{
 public:
  enum StateNames{
    v0, q0
  };
  Output(){};
  ~Output(){};
};
class Meas: public LWF::StateSVQ<0,1,1>{
 public:
  enum StateNames{
    v0, q0
  };
  Meas(){};
  ~Meas(){};
};
class Noise: public LWF::StateSVQ<0,2,0>{
 public:
  enum StateNames{
    v0, v1
  };
  Noise(){};
  ~Noise(){};
};
class ModelExample: public LWF::ModelBase<Input,Output,Meas,Noise>{
 public:
  ModelExample(){};
  ~ModelExample(){};
  Output eval(const Input& input, const Meas& meas, const Noise noise, double dt) const{
    Output output;
    output.getValue<Output::v0>() = (input.getValue<Input::q0>().inverted()*input.getValue<Input::q1>()).rotate(input.getValue<Input::v1>())-input.getValue<Input::v0>()+noise.getValue<Noise::v0>()-meas.getValue<Meas::v0>();
    rot::RotationQuaternionPD dQ = dQ.exponentialMap(noise.getValue<Noise::v1>());
    output.getValue<Output::q0>() = meas.getValue<Meas::q0>().inverted()*dQ*input.getValue<Input::q1>().inverted()*input.getValue<Input::q0>();
    return output;
  }
  mtJacInput jacInput(const Input& input, const Meas& meas, double dt) const{
    Output output;
    mtJacInput J;
    J.setZero();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::v0>()) = -Eigen::Matrix3d::Identity();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::v1>()) = rot::RotationMatrixPD(input.getValue<Input::q0>().inverted()*input.getValue<Input::q1>()).matrix();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::q0>()) = -kindr::linear_algebra::getSkewMatrixFromVector((input.getValue<Input::q0>().inverted()*input.getValue<Input::q1>()).rotate(input.getValue<Input::v1>()))*rot::RotationMatrixPD(input.getValue<Input::q0>().inverted()).matrix();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::q1>()) = kindr::linear_algebra::getSkewMatrixFromVector((input.getValue<Input::q0>().inverted()*input.getValue<Input::q1>()).rotate(input.getValue<Input::v1>()))*rot::RotationMatrixPD(input.getValue<Input::q0>().inverted()).matrix();
    J.block<3,3>(output.getId<Output::q0>(),input.getId<Input::q0>()) = rot::RotationMatrixPD(meas.getValue<Meas::q0>().inverted()*input.getValue<Input::q1>().inverted()).matrix();
    J.block<3,3>(output.getId<Output::q0>(),input.getId<Input::q1>()) = -rot::RotationMatrixPD(meas.getValue<Meas::q0>().inverted()*input.getValue<Input::q1>().inverted()).matrix();
    return J;
  }
  mtJacNoise jacNoise(const Input& input, const Meas& meas, double dt) const{
    Output output;
    mtJacNoise J;
    J.setZero();
    J.block<3,3>(output.getId<Output::v0>(),0) = Eigen::Matrix3d::Identity();
    J.block<3,3>(output.getId<Output::q0>(),3) = rot::RotationMatrixPD(meas.getValue<Meas::q0>().inverted()).matrix();
    return J;
  }
};

// The fixture for testing class PredictionModel
class ModelBaseTest : public ::testing::Test {
 protected:
  ModelBaseTest() {
    testInput_.getValue<Input::v0>() = Eigen::Vector3d(2.1,-0.2,-1.9);
    testInput_.getValue<Input::v1>() = Eigen::Vector3d(0.3,10.9,2.3);
    testInput_.getValue<Input::q0>() = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testInput_.getValue<Input::q1>() = rot::RotationQuaternionPD(0.0,0.36,0.48,0.8);
    testMeas_.getValue<Meas::v0>() = Eigen::Vector3d(-1.5,12,1785.23);
    testMeas_.getValue<Meas::q0>() = rot::RotationQuaternionPD(-3.0/sqrt(15.0),1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
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
