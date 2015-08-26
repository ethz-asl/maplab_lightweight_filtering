/*
 * GIFPrediction.hpp
 *
 *  Created on: Aug 20, 2015
 *      Author: Bloeschm
 */

#ifndef LWF_GIFPREDICTIONMODEL_HPP_
#define LWF_GIFPREDICTIONMODEL_HPP_

#include "lightweight_filtering/common.hpp"
#include "lightweight_filtering/ModelBase.hpp"
#include "lightweight_filtering/PropertyHandler.hpp"

namespace LWF{

template<typename FilterState,typename Innovation,typename Meas,typename Noise>
class GIFPrediction: public ModelBase<GIFPrediction<FilterState,Innovation,Meas,Noise>,Innovation,typename FilterState::mtState,typename FilterState::mtState,Noise>, public PropertyHandler{
 public:
  typedef ModelBase<GIFPrediction<FilterState,Innovation,Meas,Noise>,Innovation,typename FilterState::mtState,typename FilterState::mtState,Noise> mtModelBaseNew;
  typedef FilterState mtFilterState;
  typedef Innovation mtInnovation;
  typedef typename mtModelBaseNew::mtInputTuple mtInputTuple;
  typedef typename mtFilterState::mtState mtState;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  Eigen::MatrixXd noiP_;
  Eigen::MatrixXd noiPwgt_;
  Eigen::MatrixXd noiPinv_;
  bool disablePreAndPostProcessingWarning_;
  mtState stateCurrentLin_;
  Eigen::MatrixXd jacPreviousState_;
  Eigen::MatrixXd jacCurrentState_;
  Eigen::MatrixXd jacNoise_;
  mtInnovation r_;
  typename mtInnovation::mtDifVec dr_;
  typename mtState::mtDifVec dx_;
  mtMeas meas_;
  Eigen::MatrixXd A00_;
  Eigen::MatrixXd A01_;
  Eigen::MatrixXd A11_;
  Eigen::MatrixXd S_;
  Eigen::MatrixXd Sinv_;
  GIFPrediction():  noiP_((int)(mtNoise::D_),(int)(mtNoise::D_)),
                    noiPinv_((int)(mtInnovation::D_),(int)(mtInnovation::D_)),
                    jacPreviousState_((int)(mtInnovation::D_),(int)(mtState::D_)),
                    jacCurrentState_((int)(mtInnovation::D_),(int)(mtState::D_)),
                    jacNoise_((int)(mtInnovation::D_),(int)(mtNoise::D_)),
                    A00_((int)(mtState::D_),(int)(mtState::D_)),
                    A01_((int)(mtState::D_),(int)(mtState::D_)),
                    A11_((int)(mtState::D_),(int)(mtState::D_)),
                    S_((int)(mtState::D_),(int)(mtState::D_)),
                    Sinv_((int)(mtState::D_),(int)(mtState::D_)){
    noiP_.setIdentity();
    noiP_ *= 0.0001;
    mtNoise n;
    n.registerCovarianceToPropertyHandler_(noiP_,this,"Noise.");
    disablePreAndPostProcessingWarning_ = false;
    refreshProperties();
  };
  void refreshProperties(){
    noiPinv_.setIdentity();
//    noiP_.llt().solveInPlace(noiPinv_); // TODO: fix and improve
  }
  void eval_(mtInnovation& y, const mtInputTuple& inputs, double dt) const{
    evalResidual(y,std::get<0>(inputs),std::get<1>(inputs),std::get<2>(inputs),dt);
  }
  template<int i,typename std::enable_if<i==0>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    jacPreviousState(F,std::get<0>(inputs),std::get<1>(inputs),dt);
  }
  template<int i,typename std::enable_if<i==1>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    jacCurrentState(F,std::get<0>(inputs),std::get<1>(inputs),dt);
  }
  template<int i,typename std::enable_if<i==2>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    jacNoise(F,std::get<0>(inputs),std::get<1>(inputs),dt);
  }
  virtual void evalResidual(mtInnovation& y, const mtState& previousState, const mtState& currentState, const mtNoise& noise, double dt) const = 0;
  virtual void evalResidualShort(mtInnovation& y, const mtState& previousState, const mtState& currentState, double dt) const{
    mtNoise n; // TODO get static for Identity()
    n.setIdentity();
    evalResidual(y,previousState,currentState,n,dt);
  }
  virtual void jacPreviousState(Eigen::MatrixXd& F, const mtState& previousState, const mtState& currentState, double dt) const = 0;
  virtual void jacCurrentState(Eigen::MatrixXd& F, const mtState& previousState, const mtState& currentState, double dt) const = 0;
  virtual void jacNoise(Eigen::MatrixXd& F, const mtState& previousState, const mtState& currentState, double dt) const = 0;
  virtual void noMeasCase(mtFilterState& filterState, mtMeas& meas, double dt){};
  virtual void preProcess(mtFilterState& filterState, const mtMeas& meas, double dt){
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: preProcessing is not implement!" << std::endl;
    }
  };
  virtual void postProcess(mtFilterState& filterState, const mtMeas& meas, double dt){
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: postProcessing is not implement!" << std::endl;
    }
  };
  virtual ~GIFPrediction(){};
  int performPrediction(mtFilterState& filterState, const mtMeas& meas, double dt){
    meas_ = meas;
    preProcess(filterState,meas,dt);
    getLinearizationPoint(stateCurrentLin_,filterState,meas,dt);
    jacPreviousState(jacPreviousState_,filterState.state_,stateCurrentLin_,dt);
    jacCurrentState(jacCurrentState_,filterState.state_,stateCurrentLin_,dt);
    jacNoise(jacNoise_,filterState.state_,stateCurrentLin_,dt);
    this->evalResidualShort(r_,filterState.state_,stateCurrentLin_,dt);
    r_.boxMinus(mtInnovation::Identity(),dr_);
    noiPinv_.setIdentity();
    (jacNoise_*noiP_*jacNoise_.transpose()).llt().solveInPlace(noiPinv_); // Make more efficient
    A00_ = jacPreviousState_.transpose()*noiPinv_*jacPreviousState_;
    A01_ = jacPreviousState_.transpose()*noiPinv_*jacCurrentState_;
    A11_ = jacCurrentState_.transpose()*noiPinv_*jacCurrentState_;
    S_ = filterState.cov_ + A00_;
    Sinv_.setIdentity();
    S_.llt().solveInPlace(Sinv_);
    filterState.cov_ = A11_ - A01_.transpose()*Sinv_*A01_;
    dx_ = - jacCurrentState_.transpose()*noiPinv_*dr_ + A01_.transpose()*Sinv_*jacPreviousState_.transpose()*noiPinv_*dr_;
    filterState.cov_.llt().solveInPlace(dx_);
    stateCurrentLin_.boxPlus(dx_,filterState.state_);
    filterState.state_.fix();
    enforceSymmetry(filterState.cov_);
    filterState.t_ += dt;
    postProcess(filterState,meas,dt);
    return 0;
  }
  virtual void getLinearizationPoint(mtState& state1, const mtFilterState& filterState, const mtMeas& meas, double dt){
    state1 = filterState.state_;
  };
  int predictMerged(mtFilterState& filterState, double tTarget, const std::map<double,mtMeas>& measMap){
    std::cout << "\033[31mGIF predictions cannot be merged!\033[0m" << std::endl;
    return 1;
  }
  bool testPredictionJacs(double d = 1e-6,double th = 1e-6,double dt = 0.1){
    mtState previousState;
    mtState currentState;
    mtMeas meas;
    unsigned int s = 1;
    previousState.setRandom(s);
    currentState.setRandom(s);
    meas.setRandom(s);
    return testPredictionJacs(previousState,currentState,meas,d,th,dt);
  }
  bool testPredictionJacs(const mtState& previousState,const mtState& currentState, const mtMeas& meas, double d = 1e-6,double th = 1e-6,double dt = 0.1){
    mtInputTuple inputs;
    std::get<0>(inputs) = previousState;
    std::get<1>(inputs) = currentState;
    std::get<2>(inputs).setIdentity(); // Noise is always set to zero for Jacobians
    meas_ = meas;
    return this->testJacs(inputs,d,th,dt);
  }
};

}

#endif /* LWF_GIFPREDICTIONMODEL_HPP_ */
