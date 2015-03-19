/*
 * ModelBase.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_ModelBase_HPP_
#define LWF_ModelBase_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "lightweight_filtering/common.hpp"

namespace LWF{

template<typename Input, typename Output, typename Meas, typename Noise,bool useDynamicMatrix = false>
class ModelBase{
 public:
  typedef Input mtInput;
  typedef Output mtOutput;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef LWFMatrix<Output::D_,Input::D_,useDynamicMatrix> mtJacInput;
  typedef LWFMatrix<Output::D_,Noise::D_,useDynamicMatrix> mtJacNoise;
  ModelBase(){};
  virtual ~ModelBase(){};
  virtual void eval(mtOutput& output, const mtInput& input, const mtMeas& meas, double dt = 0.0) const{
    mtNoise noise;
    noise.setIdentity();
    eval(output,input,meas,noise,dt);
  }
  virtual void eval(mtOutput& output, const mtInput& input, const mtMeas& meas, const mtNoise noise, double dt = 0.0) const{
  }
  virtual void jacInput(mtJacInput& F, const mtInput& input, const mtMeas& meas, double dt = 0.0) const{
    F.setZero();
  }
  virtual void jacNoise(mtJacNoise& H, const mtInput& input, const mtMeas& meas, double dt = 0.0) const{
    H.setZero();
  }
  void jacInputFD(mtJacInput& F, const mtInput& input, const mtMeas& meas, double dt, double d){
    F.setZero();
    mtInput inputDisturbed;
    mtOutput outputReference;
    mtOutput outputDisturbed;
    eval(outputReference,input,meas,dt);
    LWFMatrix<mtInput::D_,mtInput::D_,useDynamicMatrix> I;
    typename mtOutput::mtDifVec dif;
    I.setIdentity();
    for(unsigned int i=0;i<mtInput::D_;i++){
      input.boxPlus(d*I.col(i),inputDisturbed);
      eval(outputDisturbed,inputDisturbed,meas,dt);
      outputDisturbed.boxMinus(outputReference,dif);
      F.col(i) = dif/d;
    }
  }
  void jacNoiseFD(mtJacNoise& H, const mtInput& input, const mtMeas& meas, double dt, double d){
    mtNoise noise;
    noise.setIdentity();
    H.setZero();
    mtNoise noiseDisturbed;
    mtOutput outputReference;
    mtOutput outputDisturbed;
    eval(outputReference,input,meas,noise,dt);
    LWFMatrix<mtNoise::D_,mtNoise::D_,useDynamicMatrix> I;
    typename mtOutput::mtDifVec dif;
    I.setIdentity();
    for(unsigned int i=0;i<mtNoise::D_;i++){
      noise.boxPlus(d*I.col(i),noiseDisturbed);
      eval(outputDisturbed,input,meas,noiseDisturbed,dt);
      outputDisturbed.boxMinus(outputReference,dif);
      H.col(i) = dif/d;
    }
  }
  void testJacInput(double d = 1e-6,double th = 1e-6,unsigned int s = 0,double dt = 0.1){
    mtInput input;
    mtMeas meas;
    input.setRandom(s);
    meas.setRandom(s);
    testJacInput(input,meas,d,th,dt);
  }
  void testJacNoise(double d = 1e-6,double th = 1e-6,unsigned int s = 0,double dt = 0.1){
    mtInput input;
    mtMeas meas;
    input.setRandom(s);
    meas.setRandom(s);
    testJacNoise(input,meas,d,th,dt);
  }
  void testJacInput(const mtInput& input, const mtMeas& meas, double d = 1e-6,double th = 1e-6,double dt = 0.1){
    mtJacInput F,F_FD;
    mtOutput output;
    typename mtJacInput::Index maxRow, maxCol = 0;
    jacInput(F,input,meas,dt);
    jacInputFD(F_FD,input,meas,dt,d);
    const double r = (F-F_FD).array().abs().maxCoeff(&maxRow, &maxCol);
    if(r>th){
      unsigned int outputId = mtOutput::getElementId(maxRow);
      unsigned int inputId = mtInput::getElementId(maxCol);
      std::cout << "==== Model jacInput Test failed: " << r << " is larger than " << th << " at row " << maxRow << "("<< output.getName(outputId) << ") and col " << maxCol << "("<< input.getName(inputId) << ") ====" << std::endl;
      std::cout << "  " << F(maxRow,maxCol) << "  " << F_FD(maxRow,maxCol) << std::endl;
    } else {
      std::cout << "==== Test successful (" << r << ") ====" << std::endl;
    }
  }
  void testJacNoise(const mtInput& input, const mtMeas& meas, double d = 1e-6,double th = 1e-6,double dt = 0.1){
    mtJacNoise H,H_FD;
    mtOutput output;
    mtNoise noise;
    typename mtJacNoise::Index maxRow, maxCol = 0;
    jacNoise(H,input,meas,dt);
    jacNoiseFD(H_FD,input,meas,dt,d);
    const double r = (H-H_FD).array().abs().maxCoeff(&maxRow, &maxCol);
    if(r>th){
      unsigned int outputId = mtOutput::getElementId(maxRow);
      unsigned int noiseId = mtNoise::getElementId(maxCol);
      std::cout << "==== Model jacNoise Test failed: " << r << " is larger than " << th << " at row " << maxRow << "("<< output.getName(outputId) << ") and col " << maxCol << "("<< noise.getName(noiseId) << ") ====" << std::endl;
      std::cout << "  " << H(maxRow,maxCol) << "  " << H_FD(maxRow,maxCol) << std::endl;
    } else {
      std::cout << "==== Test successful (" << r << ") ====" << std::endl;
    }
  }
  void testJacs(double d = 1e-6,double th = 1e-6,unsigned int s = 0,double dt = 0.1){
    testJacInput(d,th,s,dt);
    testJacNoise(d,th,s,dt);
  }
  void testJacs(const mtInput& input, const mtMeas& meas, double d = 1e-6,double th = 1e-6,double dt = 0.1){
    testJacInput(input,meas,d,th,dt);
    testJacNoise(input,meas,d,th,dt);
  }
};

}

#endif /* LWF_ModelBase_HPP_ */
