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
  CoordinateTransform(){};
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
};

}

#endif /* LWF_CoordinateTransform_HPP_ */
