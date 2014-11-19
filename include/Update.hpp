/*
 * Update.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef UPDATEMODEL_HPP_
#define UPDATEMODEL_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
#include "ModelBase.hpp"
#include "Prediction.hpp"
#include "State.hpp"
#include "PropertyHandler.hpp"
#include <initializer_list>

namespace LWF{

enum UpdateFilteringMode{
  UpdateEKF,
  UpdateUKF
};

template<unsigned int S, unsigned int N, unsigned int... I>
class UpdateOutlierDetectionNew{ // TODO rename, implement doOutlierDetection, support enabling and disabling
 public:
  const unsigned int S_ = S;
  const unsigned int N_ = N;
  bool outlier_;
  double mahalanobisTh_;
  unsigned int outlierCount_;
  UpdateOutlierDetectionNew<I...> sub_;
  UpdateOutlierDetectionNew(){
    mahalanobisTh_ = -0.0376136*N_*N_+1.99223*N_+2.05183; // Quadratic fit to chi square
    reset();
  }
  template<int D>
  void check(const Eigen::Matrix<double,D,1>& innVector,const Eigen::Matrix<double,D,D>& Py){
    const double d = ((innVector.block(S_,0,N_,1)).transpose()*Py.block(S_,S_,N_,N_).inverse()*innVector.block(S_,0,N_,1))(0,0);
    outlier_ = d > mahalanobisTh_;
    if(outlier_){
      outlierCount_++;
    } else {
      outlierCount_ = 0;
    }
    sub_.check(innVector,Py);
  }
  void reset(){
    outlier_ = false;
    outlierCount_ = 0;
    sub_.reset();
  }
  bool isOutlier(unsigned int i){
    if(i==0){
      return outlier_;
    } else {
      return sub_.isOutlier();
    }
  }
};

template<unsigned int S, unsigned int N>
class UpdateOutlierDetectionNew<S,N>{
 public:
  const unsigned int S_ = S;
  const unsigned int N_ = N;
  bool outlier_;
  double mahalanobisTh_;
  unsigned int outlierCount_;
  UpdateOutlierDetectionNew(){
    mahalanobisTh_ = -0.0376136*N_*N_+1.99223*N_+2.05183; // Quadratic fit to chi square
    reset();
  }
  template<int D>
  void check(const Eigen::Matrix<double,D,1>& innVector,const Eigen::Matrix<double,D,D>& Py){
    const double d = ((innVector.block(S_,0,N_,1)).transpose()*Py.block(S_,S_,N_,N_).inverse()*innVector.block(S_,0,N_,1))(0,0);
    outlier_ = d > mahalanobisTh_;
    if(outlier_){
      outlierCount_++;
    } else {
      outlierCount_ = 0;
    }
  }
  void reset(){
    outlier_ = false;
    outlierCount_ = 0;
  }
  bool isOutlier(unsigned int i){
    if(i>0){
      std::cout << "Error: wrong index in isOutlier()" << std::endl;
    }
    return outlier_;
  }
};

class UpdateOutlierDetection{
 public:
  UpdateOutlierDetection(){}
  UpdateOutlierDetection(int startIndex,int endIndex,double mahalanobisTh){
    startIndex_ = startIndex;
    endIndex_ = endIndex;
    N_ = endIndex_ - startIndex_ + 1;
    mahalanobisTh_ = mahalanobisTh;
    reset();
  }
  template<int D>
  void check(const Eigen::Matrix<double,D,1>& innVector,const Eigen::Matrix<double,D,D>& Py){
    const double d = ((innVector.block(startIndex_,0,N_,1)).transpose()*Py.block(startIndex_,startIndex_,N_,N_).inverse()*innVector.block(startIndex_,0,N_,1))(0,0);
    outlier_ = d > mahalanobisTh_;
    if(outlier_){
      outlierCount_++;
    } else {
      outlierCount_ = 0;
    }
  }
  void reset(){
    outlier_ = false;
    outlierCount_ = 0;
  }
  bool outlier_;
  unsigned int startIndex_;
  unsigned int endIndex_;
  unsigned int N_;
  double mahalanobisTh_;
  unsigned int outlierCount_;
};

template<typename Innovation, typename State, typename Meas, typename Noise, typename Prediction = DummyPrediction, bool isCoupled = false>
class Update: public ModelBase<State,Innovation,Meas,Noise>, public PropertyHandler{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef typename Prediction::mtMeas mtPredictionMeas;
  typedef typename Prediction::mtNoise mtPredictionNoise;
  typedef ComposedState<mtPredictionNoise,mtNoise> mtJointNoise;
  static const int noiseDim_ = (isCoupled)*mtPredictionNoise::D_+mtNoise::D_;
  static const bool coupledToPrediction_ = isCoupled;
  UpdateFilteringMode mode_;
  typename ModelBase<State,Innovation,Meas,Noise>::mtJacInput H_;
  typename ModelBase<State,Innovation,Meas,Noise>::mtJacNoise Hn_;
  typename mtNoise::mtCovMat updnoiP_;
  Eigen::Matrix<double,mtPredictionNoise::D_,mtNoise::D_> preupdnoiP_;
  Eigen::Matrix<double,mtNoise::D_,mtNoise::D_> noiP_;
  Eigen::Matrix<double,mtJointNoise::D_,mtJointNoise::D_> jointNoiP_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> C_;
  mtInnovation y_;
  typename mtInnovation::mtCovMat Py_;
  typename mtInnovation::mtCovMat Pyinv_;
  typename mtInnovation::mtDifVec innVector_;
  mtInnovation yIdentity_;
  typename mtState::mtDifVec updateVec_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> K_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> Pxy_;
  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+noiseDim_)+1,0> stateSigmaPoints_;
  SigmaPoints<mtState,2*(mtState::D_+mtJointNoise::D_)+1,2*(mtState::D_+mtJointNoise::D_)+1,0> coupledStateSigmaPointsPre_;
  SigmaPoints<mtNoise,2*mtNoise::D_+1,2*(mtState::D_+mtNoise::D_)+1,2*mtState::D_> stateSigmaPointsNoi_;
  SigmaPoints<mtJointNoise,2*mtJointNoise::D_+1,2*(mtState::D_+mtJointNoise::D_)+1,2*(mtState::D_)> coupledStateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+noiseDim_)+1,2*(mtState::D_+noiseDim_)+1,0> innSigmaPoints_;
  SigmaPoints<LWF::VectorState<mtState::D_>,2*mtState::D_+1,2*mtState::D_+1,0> updateVecSP_;
  SigmaPoints<mtState,2*mtState::D_+1,2*mtState::D_+1,0> posterior_;
  std::vector<UpdateOutlierDetection> outlierDetectionVector_;
  double alpha_;
  double beta_;
  double kappa_;
  Update(){
    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    mode_ = UpdateEKF;
    updnoiP_ = mtNoise::mtCovMat::Identity()*0.0001;
    preupdnoiP_ = Eigen::Matrix<double,mtPredictionNoise::D_,mtNoise::D_>::Zero();
    initUpdate();
    doubleRegister_.registerDiagonalMatrix("UpdateNoise",updnoiP_);
    if(isCoupled) doubleRegister_.registerMatrix("CorrelatedNoise",preupdnoiP_);
    doubleRegister_.registerScalar("alpha",alpha_);
    doubleRegister_.registerScalar("beta",beta_);
    doubleRegister_.registerScalar("kappa",kappa_);
  };
  void refreshNoiseSigmaPoints(){
    if(noiP_ != updnoiP_){
      noiP_ = updnoiP_;
      stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    }
  }
  void refreshJointNoiseSigmaPoints(const typename mtPredictionNoise::mtCovMat& prenoiP){
    if(jointNoiP_.template block<mtPredictionNoise::D_,mtPredictionNoise::D_>(0,0) != prenoiP ||
        jointNoiP_.template block<mtPredictionNoise::D_,mtNoise::D_>(0,mtPredictionNoise::D_) != preupdnoiP_ ||
        jointNoiP_.template block<mtNoise::D_,mtNoise::D_>(mtPredictionNoise::D_,mtPredictionNoise::D_) != updnoiP_){
      jointNoiP_.template block<mtPredictionNoise::D_,mtPredictionNoise::D_>(0,0) = prenoiP;
      jointNoiP_.template block<mtPredictionNoise::D_,mtNoise::D_>(0,mtPredictionNoise::D_) = preupdnoiP_;
      jointNoiP_.template block<mtNoise::D_,mtPredictionNoise::D_>(mtPredictionNoise::D_,0) = preupdnoiP_.transpose();
      jointNoiP_.template block<mtNoise::D_,mtNoise::D_>(mtPredictionNoise::D_,mtPredictionNoise::D_) = updnoiP_;
      coupledStateSigmaPointsNoi_.computeFromZeroMeanGaussian(jointNoiP_);
    }
  }
  void refreshUKFParameter(){
    stateSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    innSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    updateVecSP_.computeParameter(alpha_,beta_,kappa_);
    posterior_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    coupledStateSigmaPointsPre_.computeParameter(alpha_,beta_,kappa_);
    coupledStateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
  }
  void refreshProperties(){
    refreshPropertiesCustom();
    refreshUKFParameter();
  }
  virtual void refreshPropertiesCustom(){}
  virtual void preProcess(mtState& state, mtCovMat& cov, const mtMeas& meas){};
  virtual void postProcess(mtState& state, mtCovMat& cov, const mtMeas& meas){};
  virtual void preProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt){};
  virtual void postProcess(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt){};
  void initUpdate(){
    yIdentity_.setIdentity();
    updateVec_.setIdentity();
    refreshNoiseSigmaPoints();
    refreshJointNoiseSigmaPoints(mtPredictionNoise::mtCovMat::Identity());
    refreshUKFParameter();
    for(unsigned int i=0;i<outlierDetectionVector_.size();i++){
      outlierDetectionVector_[i].reset();
    }
  }
  virtual ~Update(){};
  void setMode(UpdateFilteringMode mode){
    mode_ = mode;
  }
  int updateState(mtState& state, mtCovMat& cov, const mtMeas& meas){
    switch(mode_){
      case UpdateEKF:
        return updateEKF(state,cov,meas);
      case UpdateUKF:
        return updateUKF(state,cov,meas);
      default:
        return updateEKF(state,cov,meas);
    }
  }
  int predictAndUpdate(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt){
    switch(mode_){
      case UpdateEKF:
        return predictAndUpdateEKF(state,cov,meas,prediction,predictionMeas,dt);
      case UpdateUKF:
        return predictAndUpdateUKF(state,cov,meas,prediction,predictionMeas,dt);
      default:
        return predictAndUpdateEKF(state,cov,meas,prediction,predictionMeas,dt);
    }
  }
  int updateEKF(mtState& state, mtCovMat& cov, const mtMeas& meas){
    assert(!isCoupled);
    preProcess(state,cov,meas);
    H_ = this->jacInput(state,meas);
    Hn_ = this->jacNoise(state,meas);
    y_ = this->eval(state,meas);

    // Update
    Py_ = H_*cov*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection
    for(typename std::vector<UpdateOutlierDetection>::iterator it = outlierDetectionVector_.begin(); it != outlierDetectionVector_.end(); it++){
      it->check(innVector_,Py_);
      if(it->outlier_){
        Py_.block(0,it->startIndex_,mtInnovation::D_,it->N_).setZero();
        Py_.block(it->startIndex_,0,it->N_,mtInnovation::D_).setZero();
        Py_.block(it->startIndex_,it->startIndex_,it->N_,it->N_).setIdentity();
        H_.block(it->startIndex_,0,it->N_,mtState::D_).setZero();
      }
    }
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = cov*H_.transpose()*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;
    state.boxPlus(updateVec_,state);
    postProcess(state,cov,meas);
    return 0;
  }
  int updateUKF(mtState& state, mtCovMat& cov, const mtMeas& meas){
    assert(!isCoupled);
    refreshNoiseSigmaPoints();
    preProcess(state,cov,meas);
    stateSigmaPoints_.computeFromGaussian(state,cov);

    // Update
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      innSigmaPoints_(i) = this->eval(stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i));
    }
    y_ = innSigmaPoints_.getMean();
    Py_ = innSigmaPoints_.getCovarianceMatrix(y_);
    Pxy_ = (innSigmaPoints_.getCovarianceMatrix(stateSigmaPoints_)).transpose();
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection
    for(typename std::vector<UpdateOutlierDetection>::iterator it = outlierDetectionVector_.begin(); it != outlierDetectionVector_.end(); it++){
      it->check(innVector_,Py_);
      if(it->outlier_){
        Py_.block(0,it->startIndex_,mtInnovation::D_,it->N_).setZero();
        Py_.block(it->startIndex_,0,it->N_,mtInnovation::D_).setZero();
        Py_.block(it->startIndex_,it->startIndex_,it->N_,it->N_).setIdentity();
        Pxy_.block(0,it->startIndex_,mtState::D_,it->N_).setZero();
      }
    }
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = Pxy_*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;

    // Adapt for proper linearization point
    updateVecSP_.computeFromZeroMeanGaussian(cov);
    for(unsigned int i=0;i<2*mtState::D_+1;i++){
      state.boxPlus(updateVec_+updateVecSP_(i).v_,posterior_(i));
    }
    state = posterior_.getMean();
    cov = posterior_.getCovarianceMatrix(state);
    postProcess(state,cov,meas);
    return 0;
  }
  int predictAndUpdateEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt){
    assert(isCoupled);
    preProcess(state,cov,meas,prediction,predictionMeas,dt);
    // Predict
    prediction.F_ = prediction.jacInput(state,predictionMeas,dt);
    prediction.Fn_ = prediction.jacNoise(state,predictionMeas,dt);
    state = prediction.eval(state,predictionMeas,dt);
    cov = prediction.F_*cov*prediction.F_.transpose() + prediction.Fn_*prediction.prenoiP_*prediction.Fn_.transpose();

    // Update
    H_ = this->jacInput(state,meas);
    Hn_ = this->jacNoise(state,meas);
    C_ = prediction.Fn_*preupdnoiP_*Hn_.transpose();
    y_ = this->eval(state,meas);
    Py_ = H_*cov*H_.transpose() + Hn_*updnoiP_*Hn_.transpose() + H_*C_ + C_.transpose()*H_.transpose();
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection
    for(typename std::vector<UpdateOutlierDetection>::iterator it = outlierDetectionVector_.begin(); it != outlierDetectionVector_.end(); it++){
      it->check(innVector_,Py_);
      if(it->outlier_){
        Py_.block(0,it->startIndex_,mtInnovation::D_,it->N_).setZero();
        Py_.block(it->startIndex_,0,it->N_,mtInnovation::D_).setZero();
        Py_.block(it->startIndex_,it->startIndex_,it->N_,it->N_).setIdentity();
        H_.block(it->startIndex_,0,it->N_,mtState::D_).setZero();
      }
    }
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = (cov*H_.transpose()+C_)*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;
    state.boxPlus(updateVec_,state);
    postProcess(state,cov,meas,prediction,predictionMeas,dt);
    return 0;
  }
  int predictAndUpdateUKF(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt){
    assert(isCoupled);
    refreshJointNoiseSigmaPoints(prediction.prenoiP_);
    preProcess(state,cov,meas,prediction,predictionMeas,dt);
    // Predict
    stateSigmaPoints_.computeFromGaussian(state,cov);

    // Prediction
    for(unsigned int i=0;i<coupledStateSigmaPointsPre_.L_;i++){
      coupledStateSigmaPointsPre_(i) = prediction.eval(stateSigmaPoints_(i),predictionMeas,coupledStateSigmaPointsNoi_(i).template getState<0>(),dt);
    }

    // Calculate mean and variance
    state = coupledStateSigmaPointsPre_.getMean();
    cov = coupledStateSigmaPointsPre_.getCovarianceMatrix(state);

    // Update
    for(unsigned int i=0;i<innSigmaPoints_.L_;i++){
      innSigmaPoints_(i) = this->eval(coupledStateSigmaPointsPre_(i),meas,coupledStateSigmaPointsNoi_(i).template getState<1>());
    }
    y_ = innSigmaPoints_.getMean();
    Py_ = innSigmaPoints_.getCovarianceMatrix(y_);
    Pxy_ = (innSigmaPoints_.getCovarianceMatrix(coupledStateSigmaPointsPre_)).transpose();
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection
    for(typename std::vector<UpdateOutlierDetection>::iterator it = outlierDetectionVector_.begin(); it != outlierDetectionVector_.end(); it++){
      it->check(innVector_,Py_);
      if(it->outlier_){
        Py_.block(0,it->startIndex_,mtInnovation::D_,it->N_).setZero();
        Py_.block(it->startIndex_,0,it->N_,mtInnovation::D_).setZero();
        Py_.block(it->startIndex_,it->startIndex_,it->N_,it->N_).setIdentity();
        Pxy_.block(0,it->startIndex_,mtState::D_,it->N_).setZero();
      }
    }
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = Pxy_*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;

    // Adapt for proper linearization point
    updateVecSP_.computeFromZeroMeanGaussian(cov);
    for(unsigned int i=0;i<2*mtState::D_+1;i++){
      state.boxPlus(updateVec_+updateVecSP_(i).v_,posterior_(i));
    }
    state = posterior_.getMean();
    cov = posterior_.getCovarianceMatrix(state);
    postProcess(state,cov,meas,prediction,predictionMeas,dt);
    return 0;
  }
};

}

#endif /* UPDATEMODEL_HPP_ */
