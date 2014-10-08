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
  virtual mtOutput eval(const mtInput& input, const mtMeas& meas, const double dt = 0.0) const{
    mtNoise noise;
    noise.setIdentity();
    return eval(input,meas,noise,dt);
  }
  virtual mtOutput eval(const mtInput& input, const mtMeas& meas, const mtNoise noise, const double dt) const = 0;
  virtual mtJacInput jacInput(const mtInput& input, const mtMeas& meas, const double dt) const = 0;
  virtual mtJacNoise jacNoise(const mtInput& input, const mtMeas& meas, const double dt) const = 0;
  mtJacInput jacInputFD(const mtInput& input, const mtMeas& meas, double dt, double d){
    mtJacInput F;
    mtInput inputDisturbed;
    mtOutput outputReference = eval(input,meas,dt);
    typename mtInput::CovMat I;
    typename mtInput::DiffVec dif;
    I.setIdentity();
    for(unsigned int i=0;i<mtInput::D_;i++){
      input.boxPlus(d*I.col(i),inputDisturbed);
      eval(&inputDisturbed,meas,dt).boxMinus(outputReference,dif);
      F.col(i) = dif/d;
    }
    return F;
  }
  mtJacNoise jacNoiseFD(const mtInput& input, const mtMeas& meas, double dt, double d){
    mtNoise noise;
    noise.setIdentity();
    mtJacNoise H;
    mtNoise noiseDisturbed;
    mtOutput outputReference = eval(input,meas,noise,dt);
    typename mtNoise::CovMat I;
    typename mtNoise::DiffVec dif;
    I.setIdentity();
    for(unsigned int i=0;i<mtNoise::D_;i++){
      noise.boxPlus(d*I.col(i),noiseDisturbed);
      eval(input,meas,noiseDisturbed,dt).boxMinus(outputReference,dif);
      H.col(i) = dif/d;
    }
    return H;
  }
};

}

#endif /* ModelBase_HPP_ */
