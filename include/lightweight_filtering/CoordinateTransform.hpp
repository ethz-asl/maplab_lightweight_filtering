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
  mtOutputCovMat outputCov_;
  CoordinateTransform(){};
  virtual ~CoordinateTransform(){};
  mtOutput transformState(mtInput& input) const{
    mtOutput output;
    eval(output, input, input);
    return output;
  }
  mtOutputCovMat transformCovMat(const mtInput& input,const mtInputCovMat& inputCov){
    jacInput(J_,input,input);
    outputCov_ = J_*inputCov*J_.transpose();
    postProcess(outputCov_,input);
    return outputCov_;
  }
  virtual void postProcess(mtOutputCovMat& cov,const mtInput& input){}
};

}

#endif /* LWF_CoordinateTransform_HPP_ */
