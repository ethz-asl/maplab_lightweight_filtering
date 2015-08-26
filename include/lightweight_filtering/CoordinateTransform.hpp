/*
 * CoordinateTransform.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_CoordinateTransform_HPP_
#define LWF_CoordinateTransform_HPP_

#include "lightweight_filtering/common.hpp"
#include "lightweight_filtering/ModelBase.hpp"

namespace LWF{

template<typename Input, typename Output>
class CoordinateTransform: public ModelBase<CoordinateTransform<Input,Output>,Output,Input>{
 public:
  typedef ModelBase<CoordinateTransform<Input,Output>,Output,Input> mtModelBaseNew;
  typedef typename mtModelBaseNew::mtInputTuple mtInputTuple;
  typedef Input mtInput;
  typedef Output mtOutput;
  Eigen::MatrixXd J_;
  Eigen::MatrixXd inverseProblem_C_;
  typename mtInput::mtDifVec inputDiff_;
  typename mtInput::mtDifVec correction_;
  typename mtInput::mtDifVec lastCorrection_;
  typename mtOutput::mtDifVec outputDiff_;
  mtOutput output_;
  CoordinateTransform(): J_((int)(mtOutput::D_),(int)(mtInput::D_)),inverseProblem_C_((int)(mtOutput::D_),(int)(mtOutput::D_)){
  };
  virtual ~CoordinateTransform(){};
  void eval_(mtOutput& x, const mtInputTuple& inputs, double dt) const{
    evalTransform(x,std::get<0>(inputs));
  }
  template<int i,typename std::enable_if<i==0>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    jacTransform(F,std::get<0>(inputs));
  }
  virtual void evalTransform(mtOutput& output, const mtInput& input) const = 0;
  virtual void jacTransform(Eigen::MatrixXd& F, const mtInput& input) const = 0;
  void transformState(const mtInput& input, mtOutput& output) const{
    evalTransform(output, input);
  }
  void transformCovMat(const mtInput& input,const Eigen::MatrixXd& inputCov,Eigen::MatrixXd& outputCov){
    jacInput(J_,input);
    outputCov = J_*inputCov*J_.transpose();
    postProcess(outputCov,input);
  }
  virtual void postProcess(Eigen::MatrixXd& cov,const mtInput& input){}
  bool solveInverseProblem(mtInput& input,const Eigen::MatrixXd& inputCov, const mtOutput& outputRef, const double tolerance = 1e-6, const int max_iter = 10){
    const mtInput inputRef = input;
    int count = 0;
    while(count < max_iter){
      jacInput(J_,input,input);
      inputRef.boxMinus(input,inputDiff_);
      transformState(input,output_);
      outputRef.boxMinus(output_,outputDiff_);
      inverseProblem_C_ = J_*inputCov*J_.transpose();
      correction_ = inputDiff_ + inputCov*J_.transpose()*inverseProblem_C_.inverse()*(outputDiff_-J_*inputDiff_);
      input.boxPlus(correction_,input);
      if(correction_.norm() < tolerance){
        return true;
      }
      count++;
    }
    return false;
  }
  bool solveInverseProblemRelaxed(mtInput& input,const Eigen::MatrixXd& inputCov, const mtOutput& outputRef,const Eigen::MatrixXd& outputCov, const double tolerance = 1e-6, const int max_iter = 10){
    const mtInput inputRef = input; // TODO: correct all for boxminus Jacobian
    int count = 0;
    double startError;
    lastCorrection_.setZero();
    while(count < max_iter){
      jacInput(J_,input,input);
      inputRef.boxMinus(input,inputDiff_);
      transformState(input,output_);
      outputRef.boxMinus(output_,outputDiff_);
      if(count==0) startError = (outputDiff_.transpose()*outputCov.inverse()*outputDiff_ + inputDiff_.transpose()*inputCov.inverse()*inputDiff_)(0);
      inverseProblem_C_ = J_*inputCov*J_.transpose()+outputCov;
      correction_ = inputCov*J_.transpose()*inverseProblem_C_.inverse()*(outputDiff_-J_*inputDiff_);
//      std::cout << "    Output error: " << outputDiff_.transpose()*outputCov.inverse()*outputDiff_ << ", Input error: " << inputDiff_.transpose()*inputCov.inverse()*inputDiff_ << ", Correction norm: " << correction_.norm() << std::endl;
      inputRef.boxPlus(correction_,input);
      if((lastCorrection_-correction_).norm() < tolerance){
        inputRef.boxMinus(input,inputDiff_);
        transformState(input,output_);
        outputRef.boxMinus(output_,outputDiff_);
        const double endError = (outputDiff_.transpose()*outputCov.inverse()*outputDiff_ + inputDiff_.transpose()*inputCov.inverse()*inputDiff_)(0);
        if(startError > endError){
          return true;
        } else {
          return false;
        }
      }
      lastCorrection_ = correction_;
      count++;
    }
    return false;
  }
  bool testTransformJac(double d = 1e-6,double th = 1e-6){
    mtInputTuple inputs;
    unsigned int s = 1;
    const double dt = 0.1;
    this->setRandomInputs(inputs,s);
    return this->testJacs(inputs,d,th,dt);
  }
};

}

#endif /* LWF_CoordinateTransform_HPP_ */
