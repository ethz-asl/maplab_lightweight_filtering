/*
 * UKF.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef UKF_HPP_
#define UKF_HPP_

#include "State.hpp"
#include "SigmaPoints.hpp"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <vector>

namespace LightWeightUKF{

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

template<typename State,typename Innovation,typename PredictionMeas,typename UpdateMeas, unsigned int processNoiseDim,unsigned int updateNoiseDim,unsigned int obsvConstrDim = 0>
class UKF{
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
  /*! dimension of observability constraints */
  static const unsigned int ocD_ = obsvConstrDim+(obsvConstrDim==0);
  /*! dimension of observability constraints */
  static const bool useOC_ = obsvConstrDim>0;
  typedef Eigen::Matrix<double,pD_,1> mtProcessNoise;
  typedef Eigen::Matrix<double,uD_,1> mtUpdateNoise;
  /*! estimated covariance-matrix of Kalman Filter */
  Eigen::Matrix<double,D_,D_> stateP_;
  /*! initial covariance-matrix */
  Eigen::Matrix<double,D_,D_> initStateP_;
  /*! covariance-matrix of process noise */
  Eigen::Matrix<double,pD_,pD_> prenoiP_;
  /*! covariance-matrix of update noise */
  Eigen::Matrix<double,uD_,uD_> updnoiP_;
  /*! estimated state */
  mtState state_;
  /*! (estimated) initial state */
  mtState initState_;
  /*! predicted measurements */
  mtInnovation y_;
  /*! flags for states with unknown dynamics - those are used only in update step */
  bool infinitePredictionNoise_[D_];
  /*! difference between predicted and actual measurements */
  typename mtInnovation::DiffVec innVector_;
  /*! correction vector in update step */
  typename mtState::DiffVec updateVec_;
  typedef OutlierDetection<mtInnovation> mtOutlierDetection;
  /*! measurement outlier detection */
  std::vector<mtOutlierDetection> outlierDetection_;
  /*! identity vector (zero) */
  const mtInnovation yIdentity_;
  /*! covariance of predicted measurements */
  Eigen::Matrix<double,iD_,iD_> Py_;
  /*! inverse of Py_ */
  Eigen::Matrix<double,iD_,iD_> Pyinv_;
  /*! cross-covariance between state and measurement */
  Eigen::Matrix<double,D_,iD_> Pxy_;
  /*! Kalman-Gain-Matrix */
  Eigen::Matrix<double,D_,iD_> K_;
  /*! sigma-points before prediction step */
  SigmaPoints<mtState> stateSigmaPoints1_;
  /*! sigma-points after prediction step, reused in update step */
  SigmaPoints<mtState> stateSigmaPoints2_;
  /*! sigma-points after update (projected through update-function) */
  SigmaPoints<mtInnovation> innSigmaPoints3_;
  /*! sigma-points covering process noise statistics */
  SigmaPoints<VectorState<pD_>> processNoiseSP_;
  /*! sigma-points covering update noise statistics */
  SigmaPoints<VectorState<uD_>> updateNoiseSP_;
  /*! Nullspace of observability constraint */
  Eigen::Matrix<double,D_,ocD_> OC_;
  /*! QR of observability constraint */
  Eigen::Matrix<double,D_,D_> OCQR_;
  /*! flag for different formulations of update step */
  bool useAlternativeUpdate_;

  UKF(){
    unsigned int L = 2*(D_+pD_+uD_)+1;
    stateSigmaPoints1_.resize(2*D_+1,L,0);
    stateSigmaPoints2_.resize(2*(D_+pD_)+1,L,0);
    innSigmaPoints3_.resize(2*(D_+pD_+uD_)+1,L,0);
    processNoiseSP_.resize(2*pD_+1,L,2*D_);
    updateNoiseSP_.resize(2*uD_+1,L,2*(D_+pD_));
    initStateP_.setIdentity();
    prenoiP_.setIdentity();
    updnoiP_.setIdentity();
    OC_.setZero();
    reset();
    setUKFParameter(1e-3,2.0,0.0);
    for(unsigned int i=0;i<D_;i++) infinitePredictionNoise_[i]=false;
    useAlternativeUpdate_ = false;
  };
  virtual ~UKF(){};
  virtual mtState evalPrediction(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtProcessNoise pNoise, const double dt) const = 0;
  virtual mtInnovation evalInnovation(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtUpdateMeas* mpUpdateMeas, const mtProcessNoise pNoise,const mtUpdateNoise uNoise, const double dt) const = 0;
  virtual void postProcess(const mtPredictionMeas* mpPredictionMeas,const mtUpdateMeas* mpUpdateMeas,const double dt){};
  virtual void preProcess(const mtPredictionMeas* mpPredictionMeas,const mtUpdateMeas* mpUpdateMeas,const double dt){};
  virtual void getOC(const mtState* mpState, Eigen::Matrix<double,D_,ocD_>& OC){};
  void reset(){
    state_ = initState_;
    stateP_ = initStateP_;
  };
  void computeProcessAndUpdateNoiseSP(){
    processNoiseSP_.computeFromZeroMeanGaussian(prenoiP_);
    updateNoiseSP_.computeFromZeroMeanGaussian(updnoiP_);
  }
  void print(){
    state_.print();
    std::cout << "Covariance Matrix:" << std::endl;
    std::cout << stateP_ << std::endl;
  }
  void predictAndUpdate(const mtPredictionMeas* mpPredictionMeas,const mtUpdateMeas* mpUpdateMeas,const double dt){
    predict(mpPredictionMeas,dt);
    preProcess(mpPredictionMeas,mpUpdateMeas,dt);

    // Update
    for(unsigned int i=0;i<2*(D_+pD_+uD_)+1;i++){
      innSigmaPoints3_(i) = evalInnovation(&stateSigmaPoints2_(i),mpPredictionMeas,mpUpdateMeas,processNoiseSP_(i).vector_,updateNoiseSP_(i).vector_,dt);
    }
    y_ = innSigmaPoints3_.getMean();
    Py_ = innSigmaPoints3_.getCovarianceMatrix(y_);
    Pxy_ = (innSigmaPoints3_.getCovarianceMatrix(stateSigmaPoints2_)).transpose();
    y_.boxminus(yIdentity_,innVector_);

    // Outlier detection
    for(typename std::vector<mtOutlierDetection>::iterator it = outlierDetection_.begin(); it != outlierDetection_.end(); it++){
      it->check(innVector_,Py_);
      if(it->outlier_){
        innVector_.block(it->startIndex_,0,it->N_,1).setZero();
        Py_.block(0,it->startIndex_,iD_,it->N_).setZero();
        Py_.block(it->startIndex_,0,it->N_,iD_).setZero();
        Py_.block(it->startIndex_,it->startIndex_,it->N_,it->N_).setIdentity();
        Pxy_.block(0,it->startIndex_,D_,it->N_).setZero();
      }
    }
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);	// alternative way to calculate the inverse

    // Kalman Update
    if(!useAlternativeUpdate_){
      if(useOC_){ // OC was computed earlier
        Eigen::Matrix<double,iD_,D_> H = (stateP_.inverse()*Pxy_).transpose();
        H = H*(Eigen::Matrix<double,D_,D_>::Identity()-OC_*OC_.transpose());
        K_ = stateP_*H.transpose()*Pyinv_;
      } else {
        K_ = Pxy_*Pyinv_;
      }
      stateP_ = stateP_ - K_*Py_*K_.transpose();
    } else {
      Eigen::Matrix<double,iD_,D_> H = (stateP_.inverse()*Pxy_).transpose();
      Eigen::Matrix<double,D_,iD_> HTRI = H.transpose()*(Py_-H*Pxy_).inverse();
      Eigen::Matrix<double,D_,D_> adaptedInvStateP = stateP_.inverse(); // adapted for infinite prediction noise flag
      for(unsigned int i=0;i<D_;i++){
        if(infinitePredictionNoise_[i]){
          adaptedInvStateP.block(0,i,D_,1).setZero();
          adaptedInvStateP.block(i,0,1,D_).setZero();
        }
      }
      stateP_ = (adaptedInvStateP + HTRI*H).inverse();
      K_ = stateP_*HTRI;
    }
    updateVec_ = -K_*innVector_;

    // Adapt for proper linearization point
    SigmaPoints<VectorState<D_>> updateVecSP(2*D_+1,2*D_+1,0);
    updateVecSP.computeFromZeroMeanGaussian(stateP_);
    SigmaPoints<mtState> posterior(2*D_+1,2*D_+1,0);
    for(unsigned int i=0;i<2*D_+1;i++){
      state_.boxplus(updateVec_+updateVecSP(i).vector_,posterior(i));
    }
    state_ = posterior.getMean();
    stateP_ = posterior.getCovarianceMatrix(state_);


    // Postprocessing
    postProcess(mpPredictionMeas,mpUpdateMeas,dt);
  }
  void predict(const mtPredictionMeas* mpPredictionMeas,const double dt){
    // Calculate sigma-points
    if(useOC_){
      getOC(&state_,OC_);
      Eigen::ColPivHouseholderQR<Eigen::Matrix<double,D_,ocD_>> mQR(OC_);
      OCQR_ = mQR.householderQ();
      OC_ = OCQR_.template block<D_,ocD_>(0,0);
      stateSigmaPoints1_.computeFromGaussian(state_,stateP_,OCQR_);
    } else {
      stateSigmaPoints1_.computeFromGaussian(state_,stateP_);
    }

    // Prediction
    for(unsigned int i=0;i<2*(D_+pD_)+1;i++){
      stateSigmaPoints2_(i) = evalPrediction(&stateSigmaPoints1_(i),mpPredictionMeas,processNoiseSP_(i).vector_,dt);
    }

    // Calculate mean and variance
    state_ = stateSigmaPoints2_.getMean();
    state_.fix();
    stateP_ = stateSigmaPoints2_.getCovarianceMatrix(state_);
  }
  void setUKFParameter(double alpha,double beta,double kappa){
    unsigned int L = D_+pD_+uD_;
    double lambda = alpha*alpha*(L+kappa)-L;
    double gamma = sqrt(lambda + L);
    double wm = 1/(2*(L+lambda));
    double wc = wm;
    double wc0 = lambda/(L+lambda)+(1-alpha*alpha+beta);
    innSigmaPoints3_.setParameter(wm,wc,wc0,gamma);
    stateSigmaPoints2_.setParameter(wm,wc,wc0,gamma);
    stateSigmaPoints1_.setParameter(wm,wc,wc0,gamma);
    processNoiseSP_.setParameter(wm,wc,wc0,gamma);
    updateNoiseSP_.setParameter(wm,wc,wc0,gamma);
    computeProcessAndUpdateNoiseSP();
  }
  void setInfinitePredictionNoise(char stateType, unsigned int stateIndex, bool desiredCondition){
    int offset;
	  switch(stateType){
	  case 'S':
	    offset = 0;
		  for(int i = 0; i<ScalarElement::D_; i++) {
			  infinitePredictionNoise_[stateIndex+i] = desiredCondition;
		  }
		  break;
	  case 'V':
		  offset = mtState::Ds_;
		  for(int i = 0; i<VectorElement::D_; i++) {
			  infinitePredictionNoise_[offset+stateIndex*VectorElement::D_+i] = desiredCondition;
		  }
		  break;
	  case 'Q':
		  offset = mtState::Ds_+mtState::Dv_;
		  for(int i = 0; i<QuaternionElement::D_; i++) {
			  infinitePredictionNoise_[offset+stateIndex*QuaternionElement::D_+i] = desiredCondition;
		  }
		  break;
	  default:
		  std::cout << "Error: Type of state must be 'S', 'V' or 'Q'." << std::endl;
	  }
  }
};

}

#endif /* UKF_HPP_ */
