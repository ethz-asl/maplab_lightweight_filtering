/*
 * PredictionModel.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef PREDICTIONMODEL_HPP_
#define PREDICTIONMODEL_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"

namespace LWF{

template<typename State, typename PredictionMeas, unsigned int predictionNoiseDim>
class PredictionModel{
 public:
  typedef State mtState;
  typedef PredictionMeas mtPredictionMeas;
  typedef Eigen::Matrix<double,mtState::D_,mtState::D_> mtPredictionJacState;
  typedef Eigen::Matrix<double,mtState::D_,predictionNoiseDim> mtPredictionJacNoise;
  typedef Eigen::Matrix<double,predictionNoiseDim,1> mtNoiseVector;
  PredictionModel(){};
  virtual ~PredictionModel(){};
  virtual mtState evalPrediction(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const double dt) const{
    mtNoiseVector n;
    n.setZero();
    return evalPrediction(mpState,mpPredictionMeas,&n,dt);
  }
  virtual mtState evalPrediction(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtNoiseVector* mpNoiseVector, const double dt) const = 0;
  virtual mtPredictionJacState evalPredictionJacState(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const double dt) const = 0;
  virtual mtPredictionJacNoise evalPredictionJacNoise(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const double dt) const = 0;
  void testPredictionJacState(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, double dt, double d){
    mtPredictionJacState F;
    mtState stateDisturbed;
    mtState statePredicted = evalPrediction(mpState,mpPredictionMeas,dt);
    typename mtState::CovMat I;
    typename mtState::DiffVec dif;
    I.setIdentity();
    for(unsigned int i=0;i<mtState::D_;i++){
      mpState->boxPlus(d*I.col(i),stateDisturbed);
      evalPrediction(&stateDisturbed,mpPredictionMeas,dt).boxMinus(statePredicted,dif);
      F.col(i) = dif/d;
    }
    std::cout << F << std::endl;
    std::cout << evalPredictionJacState(mpState,mpPredictionMeas,dt) << std::endl;
  }
};

}

#endif /* PREDICTIONMODEL_HPP_ */
