/*
 * CoordinateTransform.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_CoordinateTransform_HPP_
#define LWF_CoordinateTransform_HPP_

#include <Eigen/Dense>
#include "lightweight_filtering/ModelBase.hpp"
#include "lightweight_filtering/State.hpp"
#include "lightweight_filtering/common.hpp"

namespace LWF{

template<typename Input, typename Output, bool useDynamicMatrix = false>
class CoordinateTransform: public ModelBase<Input,Output,Input,Input,useDynamicMatrix>{
 public:
  typedef ModelBase<Input,Output,Input,Input,useDynamicMatrix> Base;
  using Base::eval;
  using Base::jacInput;
  typedef Input mtInput;
  typedef LWFMatrix<mtInput::D_,mtInput::D_,useDynamicMatrix> mtInputCovMat;
  typedef Output mtOutput;
  typedef LWFMatrix<mtOutput::D_,mtOutput::D_,useDynamicMatrix> mtOutputCovMat;
  typedef typename Base::mtJacInput mtJacInput;
  mtJacInput J_;
  LWFMatrix<mtOutput::D_,mtOutput::D_,useDynamicMatrix> inverseProblem_C_;
  typename mtInput::mtDifVec inputDiff_;
  typename mtInput::mtDifVec correction_;
  typename mtInput::mtDifVec lastCorrection_;
  typename mtOutput::mtDifVec outputDiff_;
  mtOutput output_;
  CoordinateTransform(){
  };
  virtual ~CoordinateTransform(){};
  void transformState(const mtInput& input, mtOutput& output) const{
    eval(output, input, input);
  }
  void transformCovMat(const mtInput& input,const mtInputCovMat& inputCov,mtOutputCovMat& outputCov){
    jacInput(J_,input,input);
    outputCov = J_*inputCov*J_.transpose();
    postProcess(outputCov,input);
  }
  virtual void postProcess(mtOutputCovMat& cov,const mtInput& input){}
  bool solveInverseProblem(mtInput& input,const mtInputCovMat& inputCov, const mtOutput& outputRef, const double tolerance = 1e-6, const int max_iter = 10){
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
  bool solveInverseProblemRelaxed(mtInput& input,const mtInputCovMat& inputCov, const mtOutput& outputRef,const mtOutputCovMat& outputCov, const double tolerance = 1e-6, const int max_iter = 10){
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
};

}

#endif /* LWF_CoordinateTransform_HPP_ */
