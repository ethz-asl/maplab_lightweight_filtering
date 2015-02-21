#include "State.hpp"
#include "ModelBase.hpp"
#include "gtest/gtest.h"
#include <assert.h>

class Input: public LWF::State<LWF::TH_multiple_elements<LWF::VectorElement<3>,2>,LWF::TH_multiple_elements<LWF::QuaternionElement,2>>{
 public:
  enum StateNames{
    v0, v1, q0, q1
  };
  Input(){};
  ~Input(){};
};
class Output: public LWF::State<LWF::VectorElement<3>,LWF::QuaternionElement>{
 public:
  enum StateNames{
    v0, q0
  };
  Output(){};
  ~Output(){};
};
class Meas: public LWF::State<LWF::VectorElement<3>,LWF::QuaternionElement>{
 public:
  enum StateNames{
    v0, q0
  };
  Meas(){};
  ~Meas(){};
};
class Noise: public LWF::State<LWF::TH_multiple_elements<LWF::VectorElement<3>,2>>{
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
  void eval(Output& output, const Input& input, const Meas& meas, const Noise noise, double dt) const{
    output.get<Output::v0>() = (input.get<Input::q0>().inverted()*input.get<Input::q1>()).rotate(input.get<Input::v1>())-input.get<Input::v0>()+noise.get<Noise::v0>()-meas.get<Meas::v0>();
    rot::RotationQuaternionPD dQ = dQ.exponentialMap(noise.get<Noise::v1>());
    output.get<Output::q0>() = meas.get<Meas::q0>().inverted()*dQ*input.get<Input::q1>().inverted()*input.get<Input::q0>();
  }
  mtJacInput jacInput(const Input& input, const Meas& meas, double dt) const{
    Output output;
    mtJacInput J;
    J.setZero();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::v0>()) = -Eigen::Matrix3d::Identity();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::v1>()) = rot::RotationMatrixPD(input.get<Input::q0>().inverted()*input.get<Input::q1>()).matrix();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::q0>()) = -kindr::linear_algebra::getSkewMatrixFromVector((input.get<Input::q0>().inverted()*input.get<Input::q1>()).rotate(input.get<Input::v1>()))*rot::RotationMatrixPD(input.get<Input::q0>().inverted()).matrix();
    J.block<3,3>(output.getId<Output::v0>(),input.getId<Input::q1>()) = kindr::linear_algebra::getSkewMatrixFromVector((input.get<Input::q0>().inverted()*input.get<Input::q1>()).rotate(input.get<Input::v1>()))*rot::RotationMatrixPD(input.get<Input::q0>().inverted()).matrix();
    J.block<3,3>(output.getId<Output::q0>(),input.getId<Input::q0>()) = rot::RotationMatrixPD(meas.get<Meas::q0>().inverted()*input.get<Input::q1>().inverted()).matrix();
    J.block<3,3>(output.getId<Output::q0>(),input.getId<Input::q1>()) = -rot::RotationMatrixPD(meas.get<Meas::q0>().inverted()*input.get<Input::q1>().inverted()).matrix();
    return J;
  }
  mtJacNoise jacNoise(const Input& input, const Meas& meas, double dt) const{
    Output output;
    mtJacNoise J;
    J.setZero();
    J.block<3,3>(output.getId<Output::v0>(),0) = Eigen::Matrix3d::Identity();
    J.block<3,3>(output.getId<Output::q0>(),3) = rot::RotationMatrixPD(meas.get<Meas::q0>().inverted()).matrix();
    return J;
  }
};

// The fixture for testing class PredictionModel
class ModelBaseTest : public ::testing::Test {
 protected:
  ModelBaseTest() {
    testInput_.get<Input::v0>() = Eigen::Vector3d(2.1,-0.2,-1.9);
    testInput_.get<Input::v1>() = Eigen::Vector3d(0.3,10.9,2.3);
    testInput_.get<Input::q0>() = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    testInput_.get<Input::q1>() = rot::RotationQuaternionPD(0.0,0.36,0.48,0.8);
    testMeas_.get<Meas::v0>() = Eigen::Vector3d(-1.5,12,1785.23);
    testMeas_.get<Meas::q0>() = rot::RotationQuaternionPD(-3.0/sqrt(15.0),1.0/sqrt(15.0),1.0/sqrt(15.0),2.0/sqrt(15.0));
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
