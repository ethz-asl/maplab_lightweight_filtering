/*
 * EKF.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef EKF_HPP_
#define EKF_HPP_

#include "State.hpp"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <vector>
#include <map>

namespace LightWeightEKF{

template<typename Innovation>
class OutlierDetection{
 public:
  OutlierDetection(int startIndex,int endIndex,double mahalanobisTh){
    outlier_ = false;
    startIndex_ = startIndex;
    endIndex_ = endIndex;
    N_ = endIndex_ - startIndex_ + 1;
    mahalanobisTh_ = mahalanobisTh;
  }
  void check(const typename Innovation::DiffVec& innVector,const Eigen::Matrix<double,Innovation::D_,Innovation::D_>& Py){
    const double d = ((innVector.block(startIndex_,0,N_,1)).transpose()*Py.block(startIndex_,startIndex_,N_,N_).inverse()*innVector.block(startIndex_,0,N_,1))(0,0);
    outlier_ = d > mahalanobisTh_;

  }
  bool outlier_;
  unsigned int startIndex_;
  unsigned int endIndex_;
  unsigned int N_;
  double mahalanobisTh_;
};

template<typename State,typename Innovation,typename PredictionMeas,typename UpdateMeas, unsigned int processNoiseDim,unsigned int updateNoiseDim>
class EKF{
 public:
  typedef State mtState;
  typedef Innovation mtInnovation;
  typedef PredictionMeas mtPredictionMeas;
  typedef UpdateMeas mtUpdateMeas;
  /*! dimension of state */
  static const unsigned int D_ = mtState::D_;
  /*! dimension of innovation */
  static const unsigned int iD_ = mtInnovation::D_;
  /*! dimension of prediction (process) noise */
  static const unsigned int pD_ = processNoiseDim;
  /*! dimension of update (measurement) noise */
  static const unsigned int uD_ = updateNoiseDim;
  typedef Eigen::Matrix<double,D_,D_> mtPredictionJacState;
  typedef Eigen::Matrix<double,D_,pD_> mtPredictionJacNoise;
  typedef Eigen::Matrix<double,iD_,D_> mtInnovationJacState;
  typedef Eigen::Matrix<double,iD_,uD_> mtInnovationJacNoise;
  /*! estimated covariance-matrix of Kalman Filter */
  Eigen::Matrix<double,D_,D_> stateP_;
  /*! initial covariance-matrix */
  Eigen::Matrix<double,D_,D_> initStateP_;
  /*! covariance-matrix of process noise */
  Eigen::Matrix<double,pD_,pD_> prenoiP_;
  /*! covariance-matrix of update noise */
  Eigen::Matrix<double,uD_,uD_> updnoiP_;
  /*! correlation between prediction and update noise */
  Eigen::Matrix<double,pD_,uD_> preupdnoiP_;
  /*! estimated state */
  mtState state_;
  /*! (estimated) initial state */
  mtState initState_;
  /*! predicted measurements */
  mtInnovation y_;
  /*! difference between predicted and actual measurements */
  typename mtInnovation::DiffVec innVector_;
  /*! correction vector in update step */
  typename mtState::DiffVec updateVec_;
  typedef OutlierDetection<mtInnovation> mtOutlierDetection;
  /*! measurement outlier detection */
  std::vector<mtOutlierDetection> outlierDetection_;
  /*! identity vector (zero) */
  const mtInnovation yIdentity_;
  /*! Prediction Jacobian */
  mtPredictionJacState F_;
  /*! Prediction Jacobian for noise*/
  mtPredictionJacNoise Fn_;
  /*! covariance of predicted measurements */
  Eigen::Matrix<double,iD_,iD_> Py_;
  /*! inverse of Py_ */
  Eigen::Matrix<double,iD_,iD_> Pyinv_;
  /*! Kalman-Gain-Matrix */
  Eigen::Matrix<double,D_,iD_> K_;
  /*! Innovation Jacobian */
  mtInnovationJacState H_;
  /*! Innovation Jacobian for noise*/
  mtInnovationJacNoise Hn_;

  /*! Safe estimated state */
  mtState stateSafe_;
  /*! Safe estimated covariance-matrix of Kalman Filter */
  Eigen::Matrix<double,D_,D_> stateSafeP_;
  /*! Front estimated state */
  mtState stateFront_;
  /*! Front estimated covariance-matrix of Kalman Filter */
  Eigen::Matrix<double,D_,D_> stateFrontP_;
  /*! Flag which checks is front estimate is valid */
  bool validFront_;
  /*! Default prediction Measurement (for front, if no prediction meas is available) */
  mtPredictionMeas defaultPredictionMeas_;

  /*! Storage for prediction measurements */
  std::map<double,mtPredictionMeas> predictionMeasMap_;
  /*! Storage for update measurements */
  std::map<double,mtUpdateMeas> updateMeasMap_;

  EKF(){
    initStateP_.setIdentity();
    prenoiP_.setIdentity();
    updnoiP_.setIdentity();
    preupdnoiP_.setZero();
    reset();
  };
  virtual ~EKF(){};
  virtual mtState evalPrediction(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const double dt) const = 0;
  virtual mtPredictionJacState evalPredictionJacState(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const double dt) const = 0;
  virtual mtPredictionJacNoise evalPredictionJacNoise(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const double dt) const = 0;
  virtual mtInnovation evalInnovation(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtUpdateMeas* mpUpdateMeas, const double dt) const = 0;
  virtual mtInnovationJacState evalInnovationJacState(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtUpdateMeas* mpUpdateMeas, const double dt) const = 0;
  virtual mtInnovationJacNoise evalInnovationJacNoise(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtUpdateMeas* mpUpdateMeas, const double dt) const = 0;
  virtual void postProcess(const mtPredictionMeas* mpPredictionMeas,const mtUpdateMeas* mpUpdateMeas,const double dt){};
  virtual void preProcess(const mtPredictionMeas* mpPredictionMeas,const mtUpdateMeas* mpUpdateMeas,const double dt){};
  void reset(){
    state_ = initState_;
    stateP_ = initStateP_;
    stateSafe_ = state_;
    stateSafeP_ = stateP_;
    stateFront_ = state_;
    stateFrontP_ = stateP_;
    validFront_ = false;
  };
  void print(){
    state_.print();
    std::cout << "Covariance Matrix:" << std::endl;
    std::cout << stateP_ << std::endl;
  }
  void predictAndUpdate(const mtPredictionMeas* mpPredictionMeas,const mtUpdateMeas* mpUpdateMeas,const double dt){
    predict(mpPredictionMeas,dt);
    preProcess(mpPredictionMeas,mpUpdateMeas,dt);

    // Update
    y_ = evalInnovation(&state_,mpPredictionMeas,mpUpdateMeas,dt);
    H_ = evalInnovationJacState(&state_,mpPredictionMeas,mpUpdateMeas,dt);
    Hn_ = evalInnovationJacNoise(&state_,mpPredictionMeas,mpUpdateMeas,dt);
    Eigen::Matrix<double,D_,iD_> C = Fn_*preupdnoiP_*Hn_.transpose();
    Py_ = H_*stateP_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose() + H_*C + C.transpose()*H_.transpose();
    y_.boxminus(yIdentity_,innVector_);

    // Outlier detection
    for(typename std::vector<mtOutlierDetection>::iterator it = outlierDetection_.begin(); it != outlierDetection_.end(); it++){
      it->check(innVector_,Py_);
      if(it->outlier_){
        innVector_.block(it->startIndex_,0,it->N_,1).setZero();
        Py_.block(0,it->startIndex_,iD_,it->N_).setZero();
        Py_.block(it->startIndex_,0,it->N_,iD_).setZero();
        Py_.block(it->startIndex_,it->startIndex_,it->N_,it->N_).setIdentity();
        H_.block(it->startIndex_,0,it->N_,D_).setZero();
      }
    }
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);	// alternative way to calculate the inverse

    // Kalman Update
    K_ = (stateP_*H_.transpose()+C)*Pyinv_;
    stateP_ = stateP_ - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;
    state_.boxplus(updateVec_,state_);

    // Postprocessing
    postProcess(mpPredictionMeas,mpUpdateMeas,dt);
  }
  void predict(const mtPredictionMeas* mpPredictionMeas,const double dt){
    // Calculate mean and variance
    double tNext = state_.t_ + dt;
    state_ = evalPrediction(&state_,mpPredictionMeas,dt);
    state_.fix();
    F_ = evalPredictionJacState(&state_,mpPredictionMeas,dt);
    Fn_ = evalPredictionJacNoise(&state_,mpPredictionMeas,dt);
    stateP_ = F_*stateP_*F_.transpose() + Fn_*prenoiP_*Fn_.transpose();
    state_.t_ = tNext;
  }
  void testPredictionJac(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, double dt, double d){
    mtPredictionJacState F;
    mtState stateDisturbed;
    typename mtState::CovMat I;
    typename mtState::DiffVec dif;
    I.setIdentity();
    I = d*I;
    for(unsigned int i=0;i<D_;i++){
      mpState->boxplus(I.col(i),stateDisturbed);
      evalPrediction(&stateDisturbed,mpPredictionMeas,dt).boxminus(evalPrediction(mpState,mpPredictionMeas,dt),dif);
      F.col(i) = dif*1/d;
    }
    std::cout << F << std::endl;
    std::cout << evalPredictionJacState(mpState,mpPredictionMeas,dt) << std::endl;
  }
  void testInnovationJac(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtUpdateMeas* mpUpdateMeas, double dt, double d){
    mtInnovationJacState H;
    mtState stateDisturbed;
    typename mtState::CovMat I;
    typename mtInnovation::DiffVec dif;
    I.setIdentity();
    I = d*I;
    for(unsigned int i=0;i<D_;i++){
      mpState->boxplus(I.col(i),stateDisturbed);
      evalInnovation(&stateDisturbed,mpPredictionMeas,mpUpdateMeas,dt).boxminus(evalInnovation(mpState,mpPredictionMeas,mpUpdateMeas,dt),dif);
      H.col(i) = dif*1/d;
    }
    std::cout << H << std::endl;
    std::cout << evalInnovationJacState(mpState,mpPredictionMeas,mpUpdateMeas,dt) << std::endl;
  }
  void clean(const double& t){
    while(!predictionMeasMap_.empty() && predictionMeasMap_.begin()->first<=t){
      predictionMeasMap_.erase(predictionMeasMap_.begin());
    }
    while(!updateMeasMap_.empty() && updateMeasMap_.begin()->first<=t){
      updateMeasMap_.erase(updateMeasMap_.begin());
    }
  }
  void addPredictionMeas(const mtPredictionMeas& mPredictionMeas, const double& t){
    assert(t>stateSafe_.t_);
    predictionMeasMap_[t] = mPredictionMeas;
    if(stateFront_.t_>=t) validFront_ = false;
  }
  void addUpdateMeas(const mtUpdateMeas& mUpdateMeas, const double& t){
    assert(t>stateSafe_.t_);
    updateMeasMap_[t] = mUpdateMeas;
    if(stateFront_.t_>=t) validFront_ = false;
  }
  void updateSafe(){
    if(predictionMeasMap_.empty() || updateMeasMap_.empty()) return;
    double nextSafeTime = std::min(predictionMeasMap_.rbegin()->first,updateMeasMap_.rbegin()->first);
    if(stateFront_.t_<=nextSafeTime && validFront_ && stateFront_.t_>stateSafe_.t_){
      stateSafe_ = stateFront_;
      stateSafeP_ = stateFrontP_;
    }
    state_ = stateSafe_;
    stateP_ = stateSafeP_;
    update(nextSafeTime);
    stateSafe_ = state_;
    stateSafeP_ = stateP_;
    clean(nextSafeTime);
  }
  void updateFront(const double& t){
    updateSafe();
    if(!validFront_ || stateSafe_.t_>stateFront_.t_){
      stateFront_ = stateSafe_;
      stateFrontP_ = stateSafeP_;
    }
    state_ = stateFront_;
    stateP_ = stateFrontP_;
    validFront_ = true;
    update(t);
    stateFront_ = state_;
    stateFrontP_ = stateP_;
  }
  void update(const double& tEnd){
    typename std::map<double,mtPredictionMeas>::iterator itPredictionMeas;
    typename std::map<double,mtUpdateMeas>::iterator itUpdateMeas;
    itPredictionMeas = predictionMeasMap_.upper_bound(state_.t_);
    itUpdateMeas = updateMeasMap_.upper_bound(state_.t_);
    double tNext = state_.t_;
    mtPredictionMeas usedPredictionMeas;
    while(state_.t_<tEnd){
      bool isFullUpdate = false;
      tNext = tEnd;
      if(itPredictionMeas!=predictionMeasMap_.end()){
        if(itPredictionMeas->first<tNext){
          tNext = itPredictionMeas->first;
        }
      }
      if(itUpdateMeas!=updateMeasMap_.end()){
        if(itUpdateMeas->first<=tNext){
          tNext = itUpdateMeas->first;
          isFullUpdate = true;
        }
      }
      if(itPredictionMeas != predictionMeasMap_.end()){
        usedPredictionMeas = itPredictionMeas->second;
      } else {
        usedPredictionMeas = defaultPredictionMeas_;
      }
      if(isFullUpdate){
        predictAndUpdate(&usedPredictionMeas,&itUpdateMeas->second,tNext-state_.t_);
      } else {
        predict(&usedPredictionMeas,tNext-state_.t_);
      }
      itPredictionMeas = predictionMeasMap_.upper_bound(state_.t_);
      itUpdateMeas = updateMeasMap_.upper_bound(state_.t_);
    }
  }
  void resetTime(const double& t){
    state_.t_ = t;
    stateSafe_.t_ = t;
    stateFront_.t_ = t;
  }
};

Eigen::Matrix3d Lmat (Eigen::Vector3d a) {
  double aNorm = a.norm();
  double factor1 = 0;
  double factor2 = 0;
  Eigen::Matrix3d ak;
  Eigen::Matrix3d ak2;
  Eigen::Matrix3d G_k;

  // Get sqew matrices
  ak = kindr::linear_algebra::getSkewMatrixFromVector(a);
  ak2 = ak*ak;

  // Compute factors
  if(aNorm >= 1e-10){
    factor1 = (1 - cos(aNorm))/pow(aNorm,2);
    factor2 = (aNorm-sin(aNorm))/pow(aNorm,3);
  } else {
    factor1 = 1/2;
    factor2 = 1/6;
  }

  return Eigen::Matrix3d::Identity()-factor1*ak+factor2*ak2;
}

}

#endif /* EKF_HPP_ */
