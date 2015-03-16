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
#include "State.hpp"
#include "PropertyHandler.hpp"
#include "SigmaPoints.hpp"
#include "Prediction.hpp"
#include <type_traits>

namespace LWF{
//template<unsigned int S, unsigned int D, unsigned int N = 1> struct OD_entry{
//  typedef typename boost::mpl::push_front<typename OD_entry<S+D,D,N-1>::mtVectorS,boost::mpl::int_<S>>::type mtVectorS;
//  typedef typename boost::mpl::push_front<typename OD_entry<S+D,D,N-1>::mtVectorD,boost::mpl::int_<D>>::type mtVectorD;
//  static const unsigned int N_ = N;
//};
//template<unsigned int S, unsigned int D> struct OD_entry<S,D,1>{
//  typedef boost::mpl::vector_c<int,S> mtVectorS;
//  typedef boost::mpl::vector_c<int,D> mtVectorD;
//  static const unsigned int N_ = 1;
//};
//
//template<typename arg, typename... args>
//class OD_concatenate{
//
//};

template<unsigned int S, unsigned int D, unsigned int N = 1> struct ODEntry{
  static const unsigned int S_ = S;
  static const unsigned int D_ = D;
};

template<unsigned int S, unsigned int D>
class OutlierDetectionBase{
 public:
  static const unsigned int S_ = S;
  static const unsigned int D_ = D;
  bool outlier_;
  bool enabled_;
  double mahalanobisTh_;
  unsigned int outlierCount_;
  OutlierDetectionBase(){
    mahalanobisTh_ = -0.0376136*D_*D_+1.99223*D_+2.05183; // Quadratic fit to chi square
    enabled_ = false;
    outlier_ = false;
    outlierCount_ = 0;
  }
  virtual ~OutlierDetectionBase(){};
  template<int E>
  void check(const Eigen::Matrix<double,E,1>& innVector,const Eigen::Matrix<double,E,E>& Py){
    const double d = ((innVector.block(S_,0,D_,1)).transpose()*Py.block(S_,S_,D_,D_).inverse()*innVector.block(S_,0,D_,1))(0,0);
    outlier_ = d > mahalanobisTh_;
    if(outlier_){
      outlierCount_++;
    } else {
      outlierCount_ = 0;
    }
  }
  virtual void registerToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str, unsigned int i = 0) = 0;
  virtual void reset() = 0;
  virtual bool isOutlier(unsigned int i) const = 0;
  virtual void setEnabled(unsigned int i,bool enabled) = 0;
  virtual void setEnabledAll(bool enabled) = 0;
  virtual unsigned int& getCount(unsigned int i) = 0;
  virtual double& getMahalTh(unsigned int i) = 0;
};

template<unsigned int S, unsigned int D,typename T>
class OutlierDetectionConcat: public OutlierDetectionBase<S,D>{
 public:
  using OutlierDetectionBase<S,D>::S_;
  using OutlierDetectionBase<S,D>::D_;
  using OutlierDetectionBase<S,D>::outlier_;
  using OutlierDetectionBase<S,D>::enabled_;
  using OutlierDetectionBase<S,D>::mahalanobisTh_;
  using OutlierDetectionBase<S,D>::outlierCount_;
  using OutlierDetectionBase<S,D>::check;
  T sub_;
  template<int dI, int dS>
  void doOutlierDetection(const Eigen::Matrix<double,dI,1>& innVector,Eigen::Matrix<double,dI,dI>& Py,Eigen::Matrix<double,dI,dS>& H){
    static_assert(dI>=S+D,"Outlier detection out of range");
    check(innVector,Py);
    sub_.doOutlierDetection(innVector,Py,H);
    if(outlier_ && enabled_){
      Py.block(0,S_,dI,D_).setZero();
      Py.block(S_,0,D_,dI).setZero();
      Py.block(S_,S_,D_,D_).setIdentity();
      H.block(S_,0,D_,dS).setZero();
    }
  }
  void registerToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str, unsigned int i = 0){
    mpPropertyHandler->doubleRegister_.registerScalar(str + std::to_string(i), mahalanobisTh_);
    sub_.registerToPropertyHandler(mpPropertyHandler,str,i+1);
  }
  void reset(){
    outlier_ = false;
    outlierCount_ = 0;
    sub_.reset();
  }
  bool isOutlier(unsigned int i) const{
    if(i==0){
      return outlier_;
    } else {
      return sub_.isOutlier(i-1);
    }
  }
  void setEnabled(unsigned int i,bool enabled){
    if(i==0){
      enabled_ = enabled;
    } else {
      sub_.setEnabled(i-1,enabled);
    }
  }
  void setEnabledAll(bool enabled){
    enabled_ = enabled;
    sub_.setEnabledAll(enabled);
  }
  unsigned int& getCount(unsigned int i){
    if(i==0){
      return outlierCount_;
    } else {
      return sub_.getCount(i-1);
    }
  }
  double& getMahalTh(unsigned int i){
    if(i==0){
      return mahalanobisTh_;
    } else {
      return sub_.getMahalTh(i-1);
    }
  }
};

class OutlierDetectionDefault: public OutlierDetectionBase<0,0>{
 public:
  using OutlierDetectionBase<0,0>::mahalanobisTh_;
  using OutlierDetectionBase<0,0>::outlierCount_;
  template<int dI, int dS>
  void doOutlierDetection(const Eigen::Matrix<double,dI,1>& innVector,Eigen::Matrix<double,dI,dI>& Py,Eigen::Matrix<double,dI,dS>& H){
  }
  void registerToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str, unsigned int i = 0){
  }
  void reset(){
  }
  bool isOutlier(unsigned int i) const{
    assert(0);
    return false;
  }
  void setEnabled(unsigned int i,bool enabled){
    assert(0);
  }
  void setEnabledAll(bool enabled){
  }
  unsigned int& getCount(unsigned int i){
    assert(0);
    return outlierCount_;
  }
  double& getMahalTh(unsigned int i){
    assert(0);
    return mahalanobisTh_;
  }
};

template<typename... ODEntries>
class OutlierDetection{};

template<unsigned int S, unsigned int D, unsigned int N, typename... ODEntries>
class OutlierDetection<ODEntry<S,D,N>,ODEntries...>: public OutlierDetectionConcat<S,D,OutlierDetection<ODEntry<S+D,D,N-1>,ODEntries...>>{};
template<unsigned int S, unsigned int D, typename... ODEntries>
class OutlierDetection<ODEntry<S,D,1>,ODEntries...>: public OutlierDetectionConcat<S,D,OutlierDetection<ODEntries...>>{};
template<unsigned int S, unsigned int D, typename... ODEntries>
class OutlierDetection<ODEntry<S,D,0>,ODEntries...>: public OutlierDetection<ODEntries...>{};
template<unsigned int S, unsigned int D, unsigned int N>
class OutlierDetection<ODEntry<S,D,N>>: public OutlierDetectionConcat<S,D,OutlierDetection<ODEntry<S+D,D,N-1>>>{};
template<unsigned int S, unsigned int D>
class OutlierDetection<ODEntry<S,D,1>>: public OutlierDetectionConcat<S,D,OutlierDetectionDefault>{};
template<unsigned int S, unsigned int D>
class OutlierDetection<ODEntry<S,D,0>>: public OutlierDetectionDefault{};

template<typename Innovation, typename FilterState, typename Meas, typename Noise, typename OutlierDetection = OutlierDetectionDefault, bool isCoupled = false>
class Update: public ModelBase<typename FilterState::mtState,Innovation,Meas,Noise>, public PropertyHandler{
 public: // TODO: remove unnecessary
  static_assert(!isCoupled || Noise::D_ == FilterState::noiseExtensionDim_,"Noise Size for coupled Update must match noise extension of prediction!");
  typedef FilterState mtFilterState;
  typedef typename mtFilterState::mtState mtState;
  typedef typename mtFilterState::mtFilterCovMat mtFilterCovMat;
  typedef typename mtFilterState::mtPredictionMeas mtPredictionMeas;
  typedef typename mtFilterState::mtPredictionNoise mtPredictionNoise;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef OutlierDetection mtOutlierDetection;
  static const bool coupledToPrediction_ = isCoupled;
  bool useSpecialLinearizationPoint_; // TODO: clean (with state)
  typedef ModelBase<mtState,mtInnovation,mtMeas,mtNoise> mtModelBase;
  typename mtModelBase::mtJacInput H_;
  typename mtModelBase::mtJacNoise Hn_;
  typename mtNoise::mtCovMat updnoiP_;
  Eigen::Matrix<double,mtPredictionNoise::D_,mtNoise::D_> preupdnoiP_;
  Eigen::Matrix<double,mtNoise::D_,mtNoise::D_> noiP_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> C_;
  mtInnovation y_;
  typename mtInnovation::mtCovMat Py_;
  typename mtInnovation::mtCovMat Pyinv_;
  typename mtInnovation::mtDifVec innVector_;
  mtInnovation yIdentity_;
  typename mtState::mtDifVec updateVec_;
  typename mtState::mtDifVec difVecLin_;
  double updateVecNorm_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> K_;
  Eigen::Matrix<double,mtInnovation::D_,mtState::D_> Pyx_;

  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+mtNoise::D_)+1,0> stateSigmaPoints_;
  SigmaPoints<mtNoise,2*mtNoise::D_+1,2*(mtState::D_+mtNoise::D_)+1,2*mtState::D_> stateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+mtNoise::D_)+1,2*(mtState::D_+mtNoise::D_)+1,0> innSigmaPoints_;
  SigmaPoints<mtNoise,2*(mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_)> coupledStateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,0> coupledInnSigmaPoints_;
  SigmaPoints<LWF::VectorElement<mtState::D_>,2*mtState::D_+1,2*mtState::D_+1,0> updateVecSP_;
  SigmaPoints<mtState,2*mtState::D_+1,2*mtState::D_+1,0> posterior_;
  double alpha_;
  double beta_;
  double kappa_;
  double updateVecNormTermination_;
  int maxNumIteration_;
  mtOutlierDetection outlierDetection_;
  Update(){
    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    updateVecNormTermination_ = 1e-6;
    maxNumIteration_  = 10;
    updnoiP_ = mtNoise::mtCovMat::Identity()*0.0001;
    noiP_.setZero();
    preupdnoiP_ = Eigen::Matrix<double,mtPredictionNoise::D_,mtNoise::D_>::Zero();
    useSpecialLinearizationPoint_ = false;
    initUpdate();
    mtNoise n;
    n.registerCovarianceToPropertyHandler_(updnoiP_,this,"UpdateNoise.");
    doubleRegister_.registerScalar("alpha",alpha_);
    doubleRegister_.registerScalar("beta",beta_);
    doubleRegister_.registerScalar("kappa",kappa_);
    doubleRegister_.registerScalar("updateVecNormTermination",updateVecNormTermination_);
    intRegister_.registerScalar("maxNumIteration",maxNumIteration_);
    outlierDetection_.setEnabledAll(false);
  };
  void refreshNoiseSigmaPoints(){
    if(noiP_ != updnoiP_){
      noiP_ = updnoiP_;
      stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    }
  }
  void refreshUKFParameter(){
    stateSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    innSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    coupledInnSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    updateVecSP_.computeParameter(alpha_,beta_,kappa_);
    posterior_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    coupledStateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
  }
  void refreshProperties(){
    refreshPropertiesCustom();
    refreshUKFParameter();
  }
  virtual void refreshPropertiesCustom(){}
  virtual void preProcess(mtFilterState& filterState, const mtMeas& meas){};
  virtual void postProcess(mtFilterState& filterState, const mtMeas& meas, mtOutlierDetection outlierDetection){};
  void initUpdate(){
    yIdentity_.setIdentity();
    updateVec_.setIdentity();
    refreshNoiseSigmaPoints();
    refreshUKFParameter();
  }
  virtual ~Update(){};
  int performUpdate(mtFilterState& filterState, const mtMeas& meas){
    switch(filterState.mode_){
      case ModeEKF:
        return performUpdateEKF(filterState,meas);
      case ModeUKF:
        return performUpdateUKF(filterState,meas);
      default:
        return performUpdateEKF(filterState,meas);
    }
  }
  int performUpdateEKF(mtFilterState& filterState, const mtMeas& meas){
    preProcess(filterState,meas);
    this->jacInput(H_,filterState.state_,meas);
    this->jacNoise(Hn_,filterState.state_,meas);
    this->eval(y_,filterState.state_,meas);

    if(isCoupled){
      C_ = filterState.G_*preupdnoiP_*Hn_.transpose();
      Py_ = H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose() + H_*C_ + C_.transpose()*H_.transpose();
    } else {
      Py_ = H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    }
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection
    outlierDetection_.doOutlierDetection(innVector_,Py_,H_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    if(isCoupled){
      K_ = (filterState.cov_*H_.transpose()+C_)*Pyinv_;
    } else {
      K_ = filterState.cov_*H_.transpose()*Pyinv_;
    }
    filterState.cov_ = filterState.cov_ - K_*Py_*K_.transpose();
    if(!useSpecialLinearizationPoint_){
      updateVec_ = -K_*innVector_;
    } else {
      updateVec_ = -K_*(innVector_-H_*filterState.state_.difVecLin_); // includes correction for offseted linearization point
    }
    filterState.state_.boxPlus(updateVec_,filterState.state_);
    postProcess(filterState,meas,outlierDetection_);
    return 0;
  }
  int performUpdateIEKF(mtFilterState& filterState, const mtMeas& meas){ // TODO: handle coupled
    preProcess(filterState,meas);
    mtState linState = filterState.state_;
    updateVecNorm_ = updateVecNormTermination_;
    for(unsigned int i=0;i<maxNumIteration_ & updateVecNorm_>=updateVecNormTermination_;i++){
      this->jacInput(H_,linState,meas);
      this->jacNoise(Hn_,linState,meas);
      this->eval(y_,linState,meas);

      // Update
      Py_ = H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
      y_.boxMinus(yIdentity_,innVector_);

      // Outlier detection
      outlierDetection_.doOutlierDetection(innVector_,Py_,H_);
      Pyinv_.setIdentity();
      Py_.llt().solveInPlace(Pyinv_);

      // Kalman Update
      K_ = filterState.cov_*H_.transpose()*Pyinv_;
      linState.boxMinus(filterState.state_,difVecLin_);
      updateVec_ = -K_*(innVector_-H_*difVecLin_); // includes correction for offseted linearization point
      filterState.state_.boxPlus(updateVec_,linState);
      updateVecNorm_ = updateVec_.norm();
    }
    filterState.state_ = linState;
    filterState.cov_ = filterState.cov_ - K_*Py_*K_.transpose();
    postProcess(filterState,meas,outlierDetection_);
    return 0;
  }
  int performUpdateUKF(mtFilterState& filterState, const mtMeas& meas){
    preProcess(filterState,meas);
    handleUpdateSigmaPoints<isCoupled>(filterState,meas);
    y_.boxMinus(yIdentity_,innVector_);

    outlierDetection_.doOutlierDetection(innVector_,Py_,Pyx_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = Pyx_.transpose()*Pyinv_;
    filterState.cov_ = filterState.cov_ - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;

    // Adapt for proper linearization point
    updateVecSP_.computeFromZeroMeanGaussian(filterState.cov_);
    for(unsigned int i=0;i<2*mtState::D_+1;i++){
      filterState.state_.boxPlus(updateVec_+updateVecSP_(i).v_,posterior_(i));
    }
    filterState.state_ = posterior_.getMean();
    filterState.cov_ = posterior_.getCovarianceMatrix(filterState.state_);
    postProcess(filterState,meas,outlierDetection_);
    return 0;
  }
  template<bool IC = isCoupled, typename std::enable_if<(IC)>::type* = nullptr>
  void handleUpdateSigmaPoints(mtFilterState& filterState, const mtMeas& meas){
    coupledStateSigmaPointsNoi_.extendZeroMeanGaussian(filterState.stateSigmaPointsNoi_,updnoiP_,preupdnoiP_);
    for(unsigned int i=0;i<coupledInnSigmaPoints_.L_;i++){
      this->eval(coupledInnSigmaPoints_(i),filterState.stateSigmaPointsPre_(i),meas,coupledStateSigmaPointsNoi_(i));
    }
    y_ = coupledInnSigmaPoints_.getMean();
    Py_ = coupledInnSigmaPoints_.getCovarianceMatrix(y_);
    Pyx_ = (coupledInnSigmaPoints_.getCovarianceMatrix(filterState.stateSigmaPointsPre_));
  }
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  void handleUpdateSigmaPoints(mtFilterState& filterState, const mtMeas& meas){
    refreshNoiseSigmaPoints();
    stateSigmaPoints_.computeFromGaussian(filterState.state_,filterState.cov_);
    for(unsigned int i=0;i<innSigmaPoints_.L_;i++){
      this->eval(innSigmaPoints_(i),stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i));
    }
    y_ = innSigmaPoints_.getMean();
    Py_ = innSigmaPoints_.getCovarianceMatrix(y_);
    Pyx_ = (innSigmaPoints_.getCovarianceMatrix(stateSigmaPoints_));
  }
};

}

#endif /* UPDATEMODEL_HPP_ */
