/*
 * ModelBase.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef ModelBase_HPP_
#define ModelBase_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"

namespace LWF{

template<typename Input, typename Output, typename Meas, typename Noise>
class ModelBase{
 public:
  typedef Input mtInput;
  typedef Output mtOutput;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef Eigen::Matrix<double,mtOutput::D_,mtInput::D_> mtJacInput;
  typedef Eigen::Matrix<double,mtOutput::D_,mtNoise::D_> mtJacNoise;
  typedef Eigen::Matrix<double,mtNoise::D_,1> mtNoiseVector;
  ModelBase(){};
  virtual ~ModelBase(){};
  virtual mtInput eval(const mtInput* mpInput, const mtMeas* mpMeas, const double dt = 0.0) const{
    mtNoise n;
    n.setIdentity();
    return eval(mpInput,mpMeas,&n,dt);
  }
  virtual mtInput eval(const mtInput* mpInput, const mtMeas* mpMeas, const mtNoise* mpNoise, const double dt) const = 0;
  virtual mtJacInput jacInput(const mtInput* mpInput, const mtMeas* mpMeas, const double dt) const = 0;
  virtual mtJacNoise jacNoise(const mtInput* mpInput, const mtMeas* mpMeas, const double dt) const = 0;
  mtJacInput jacInputFD(const mtInput* mpInput, const mtMeas* mpMeas, double dt, double d){
    mtJacInput F;
    mtInput inputDisturbed;
    mtOutput outputReference = eval(mpInput,mpMeas,dt);
    typename mtInput::CovMat I;
    typename mtInput::DiffVec dif;
    I.setIdentity();
    for(unsigned int i=0;i<mtInput::D_;i++){
      mpInput->boxPlus(d*I.col(i),inputDisturbed);
      eval(&inputDisturbed,mpMeas,dt).boxMinus(outputReference,dif);
      F.col(i) = dif/d;
    }
    return F;
  }
  mtJacNoise jacNoiseFD(const mtInput* mpInput, const mtMeas* mpMeas, double dt, double d){
    mtNoise noise;
    noise.setIdentity();
    mtJacNoise H;
    mtNoise noiseDisturbed;
    mtOutput outputReference = eval(mpInput,mpMeas,noise,dt);
    typename mtNoise::CovMat I;
    typename mtNoise::DiffVec dif;
    I.setIdentity();
    for(unsigned int i=0;i<mtNoise::D_;i++){
      noise.boxPlus(d*I.col(i),noiseDisturbed);
      eval(mpInput,mpMeas,noiseDisturbed,dt).boxMinus(outputReference,dif);
      H.col(i) = dif/d;
    }
    return H;
  }
};

}

#endif /* ModelBase_HPP_ */
