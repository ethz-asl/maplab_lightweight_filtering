/*
 * FilterBase.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef FilterBase_HPP_
#define FilterBase_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
#include <map>
#include "ModelBase.hpp"

namespace LWF{

template<typename State>
class FilterBase{
 public:
  typedef State mtState;
  /*! dimension of state */
  static const unsigned int D_ = mtState::D_;
  /*! estimated covariance-matrix of Kalman Filter */
  Eigen::Matrix<double,D_,D_> stateP_;
  /*! initial covariance-matrix */
  Eigen::Matrix<double,D_,D_> initStateP_;
  /*! estimated state */
  mtState state_;
  /*! (estimated) initial state */
  mtState initState_;

//  /*! Storage for prediction measurements */
//  std::map<double,mtPredictionMeas> predictionMeasMap_;
//  /*! Storage for update measurements */
//  std::map<double,mtUpdateMeas> updateMeasMap_;
  FilterBase(){};
  virtual ~FilterBase(){};
  template<typename PredictionModel>
  void predict(const PredictionModel::mtMeas* mpPredictionMeas,const double dt){
    // Calculate mean and variance
    double tNext = state_.t_ + dt;
    state_ = PredictionModel::eval(&state_,mpPredictionMeas,dt);
    state_.fix();
    PredictionModel::mtJacInput F_ = jacInput(&state_,mpPredictionMeas,dt);
    PredictionModel::mtJacNoise Fn_ = jacNoise(&state_,mpPredictionMeas,dt);
//    stateP_ = F_*stateP_*F_.transpose() + Fn_*prenoiP_*Fn_.transpose(); //TODO
    state_.t_ = tNext;
  }
};

}

#endif /* FilterBase_HPP_ */
