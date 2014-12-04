/*
 * CoordinateTransform.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef CoordinateTransform_HPP_
#define CoordinateTransform_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
#include "ModelBase.hpp"
#include "State.hpp"

namespace LWF{

template<typename Input, typename Output>
class CoordinateTransform: public ModelBase<Input,Output,Input,Input>{
 public:
  typedef ModelBase<Input,Output,Input,Input> Base;
  using Base::eval;
  typedef Input mtInput;
  typedef typename Input::mtCovMat mtInputCovMat;
  typedef Output mtOutput;
  typedef typename Output::mtCovMat mtOutputCovMat;
  typedef typename Base::mtJacInput mtJacInput;
  mtJacInput J_;
  mtOutputCovMat outputCov_;
  mtOutput transformState(const mtInput& input) const{
    return eval(input, input);
  }
  mtOutputCovMat transformCovMat(const mtInput& input,const mtInputCovMat& inputCov){
    J_ = jacInput(input,input);
    outputCov_ = J_*inputCov*J_.transpose();
    postProcess(outputCov_,input);
    return outputCov_;
  }
  virtual void postProcess(mtOutputCovMat& cov,const mtInput& input){}
};

}

#endif /* CoordinateTransform_HPP_ */
