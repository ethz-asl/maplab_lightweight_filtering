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


enum UpdateFilteringMode{
  UpdateEKF,
  UpdateUKF
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

template<typename Innovation, typename State, typename Meas, typename Noise, typename OutlierDetection = OutlierDetectionDefault, typename Prediction = DummyPrediction, bool isCoupled = false>
class Update: public ModelBase<State,Innovation,Meas,Noise>, public PropertyHandler{
 public:
  typedef State mtState;
  typedef typename mtState::mtCovMat mtCovMat;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef OutlierDetection mtOutlierDetection;
  typedef Prediction mtPrediction;
  typedef typename Prediction::mtMeas mtPredictionMeas;
  typedef typename Prediction::mtNoise mtPredictionNoise;
  typedef LWF::ComposedState<mtPredictionNoise,mtNoise> mtJointNoise;
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
  typename mtState::mtDifVec difVecLin_;
  double updateVecNorm_;
  Eigen::Matrix<double,mtState::D_,mtInnovation::D_> K_;
  Eigen::Matrix<double,mtInnovation::D_,mtState::D_> Pyx_;
  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+noiseDim_)+1,0> stateSigmaPoints_;
  SigmaPoints<mtState,2*(mtState::D_+mtJointNoise::D_)+1,2*(mtState::D_+mtJointNoise::D_)+1,0> coupledStateSigmaPointsPre_;
  SigmaPoints<mtNoise,2*mtNoise::D_+1,2*(mtState::D_+mtNoise::D_)+1,2*mtState::D_> stateSigmaPointsNoi_;
  SigmaPoints<mtJointNoise,2*mtJointNoise::D_+1,2*(mtState::D_+mtJointNoise::D_)+1,2*(mtState::D_)> coupledStateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+noiseDim_)+1,2*(mtState::D_+noiseDim_)+1,0> innSigmaPoints_;
  SigmaPoints<LWF::VectorElement<mtState::D_>,2*mtState::D_+1,2*mtState::D_+1,0> updateVecSP_;
  SigmaPoints<mtState,2*mtState::D_+1,2*mtState::D_+1,0> posterior_;
  double alpha_;
  double beta_;
  double kappa_;
  double updateVecNormTermination_;
  int maxNumIteration_;
  Update(){
    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    updateVecNormTermination_ = 1e-6;
    maxNumIteration_  = 10;
    mode_ = UpdateEKF;
    updnoiP_ = mtNoise::mtCovMat::Identity()*0.0001;
    preupdnoiP_ = Eigen::Matrix<double,mtPredictionNoise::D_,mtNoise::D_>::Zero();
    initUpdate();
    mtNoise n;
    n.registerCovarianceToPropertyHandler_(updnoiP_,this,"UpdateNoise.");
//    if(isCoupled) doubleRegister_.registerMatrix("CorrelatedNoise",preupdnoiP_); // TODO: solve, for now has to be handled by user
    doubleRegister_.registerScalar("alpha",alpha_);
    doubleRegister_.registerScalar("beta",beta_);
    doubleRegister_.registerScalar("kappa",kappa_);
    doubleRegister_.registerScalar("updateVecNormTermination",updateVecNormTermination_);
    intRegister_.registerScalar("maxNumIteration",maxNumIteration_);
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
      coupledStateSigmaPointsNoi_.computeFromZeroMeanGaussian(jointNoiP_); // TODO: test
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
  }
  virtual ~Update(){};
  void setMode(UpdateFilteringMode mode){
    mode_ = mode;
  }
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  int performUpdate(mtState& state, mtCovMat& cov, const mtMeas& meas, mtOutlierDetection* mpOutlierDetection = nullptr){
    switch(mode_){
      case UpdateEKF:
        return performUpdateEKF(state,cov,meas,mpOutlierDetection);
      case UpdateUKF:
        return performUpdateUKF(state,cov,meas,mpOutlierDetection);
      default:
        return performUpdateEKF(state,cov,meas,mpOutlierDetection);
    }
  }
  template<bool IC = isCoupled, typename std::enable_if<(IC)>::type* = nullptr>
  int performPredictionAndUpdate(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt, mtOutlierDetection* mpOutlierDetection = nullptr){
    switch(mode_){
      case UpdateEKF:
        return performPredictionAndUpdateEKF(state,cov,meas,prediction,predictionMeas,dt,mpOutlierDetection);
      case UpdateUKF:
        return performPredictionAndUpdateUKF(state,cov,meas,prediction,predictionMeas,dt,mpOutlierDetection);
      default:
        return performPredictionAndUpdateEKF(state,cov,meas,prediction,predictionMeas,dt,mpOutlierDetection);
    }
  }
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  int performUpdateEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, mtOutlierDetection* mpOutlierDetection = nullptr){
    preProcess(state,cov,meas);
    H_ = this->jacInput(state,meas);
    Hn_ = this->jacNoise(state,meas);
    y_ = this->eval(state,meas);

    // Update
    Py_ = H_*cov*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection
    if(mpOutlierDetection != nullptr) mpOutlierDetection->doOutlierDetection(innVector_,Py_,H_);
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
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  int performUpdateLEKF(mtState& state, const mtState& linState, mtCovMat& cov, const mtMeas& meas, mtOutlierDetection* mpOutlierDetection = nullptr){
    preProcess(state,cov,meas);
    H_ = this->jacInput(linState,meas);
    Hn_ = this->jacNoise(linState,meas);
    y_ = this->eval(linState,meas);

    // Update
    Py_ = H_*cov*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection
    if(mpOutlierDetection != nullptr) mpOutlierDetection->doOutlierDetection(innVector_,Py_,H_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = cov*H_.transpose()*Pyinv_;
    cov = cov - K_*Py_*K_.transpose();
    linState.boxMinus(state,difVecLin_);
    updateVec_ = -K_*(innVector_-H_*difVecLin_); // includes correction for offseted linearization point
    state.boxPlus(updateVec_,state);
    postProcess(state,cov,meas);
    return 0;
  }
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  int performUpdateIEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, mtOutlierDetection* mpOutlierDetection = nullptr){
    preProcess(state,cov,meas);
    mtState linState = state;
    updateVecNorm_ = updateVecNormTermination_;
    for(unsigned int i=0;i<maxNumIteration_ & updateVecNorm_>=updateVecNormTermination_;i++){
      H_ = this->jacInput(linState,meas);
      Hn_ = this->jacNoise(linState,meas);
      y_ = this->eval(linState,meas);

      // Update
      Py_ = H_*cov*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
      y_.boxMinus(yIdentity_,innVector_);

      // Outlier detection
      if(mpOutlierDetection != nullptr) mpOutlierDetection->doOutlierDetection(innVector_,Py_,H_);
      Pyinv_.setIdentity();
      Py_.llt().solveInPlace(Pyinv_);

      // Kalman Update
      K_ = cov*H_.transpose()*Pyinv_;
      linState.boxMinus(state,difVecLin_);
      updateVec_ = -K_*(innVector_-H_*difVecLin_); // includes correction for offseted linearization point
      state.boxPlus(updateVec_,linState);
      updateVecNorm_ = updateVec_.norm();
    }
    state = linState;
    cov = cov - K_*Py_*K_.transpose();
    postProcess(state,cov,meas);
    return 0;
  }
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  int performUpdateUKF(mtState& state, mtCovMat& cov, const mtMeas& meas, mtOutlierDetection* mpOutlierDetection = nullptr){
    refreshNoiseSigmaPoints();
    preProcess(state,cov,meas);
    stateSigmaPoints_.computeFromGaussian(state,cov);

    // Update
    for(unsigned int i=0;i<stateSigmaPoints_.L_;i++){
      innSigmaPoints_(i) = this->eval(stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i));
    }
    y_ = innSigmaPoints_.getMean();
    Py_ = innSigmaPoints_.getCovarianceMatrix(y_);
    Pyx_ = (innSigmaPoints_.getCovarianceMatrix(stateSigmaPoints_));
    y_.boxMinus(yIdentity_,innVector_);

    if(mpOutlierDetection != nullptr) mpOutlierDetection->doOutlierDetection(innVector_,Py_,Pyx_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = Pyx_.transpose()*Pyinv_;
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
  template<bool IC = isCoupled, typename std::enable_if<(IC)>::type* = nullptr>
  int performPredictionAndUpdateEKF(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt, mtOutlierDetection* mpOutlierDetection = nullptr){
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
    if(mpOutlierDetection != nullptr) mpOutlierDetection->doOutlierDetection(innVector_,Py_,H_);
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
  template<bool IC = isCoupled, typename std::enable_if<(IC)>::type* = nullptr>
  int performPredictionAndUpdateUKF(mtState& state, mtCovMat& cov, const mtMeas& meas, Prediction& prediction, const mtPredictionMeas& predictionMeas, double dt, mtOutlierDetection* mpOutlierDetection = nullptr){
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
    Pyx_ = (innSigmaPoints_.getCovarianceMatrix(coupledStateSigmaPointsPre_));
    y_.boxMinus(yIdentity_,innVector_);

    if(mpOutlierDetection != nullptr) mpOutlierDetection->doOutlierDetection(innVector_,Py_,Pyx_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = Pyx_.transpose()*Pyinv_;
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
