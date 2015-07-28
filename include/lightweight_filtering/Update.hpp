/*
 * Update.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_UPDATEMODEL_HPP_
#define LWF_UPDATEMODEL_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
#include "lightweight_filtering/ModelBase.hpp"
#include "lightweight_filtering/State.hpp"
#include "lightweight_filtering/FilterState.hpp"
#include "lightweight_filtering/PropertyHandler.hpp"
#include "lightweight_filtering/SigmaPoints.hpp"
#include "lightweight_filtering/OutlierDetection.hpp"
#include "lightweight_filtering/common.hpp"
#include <type_traits>

namespace LWF{

template<typename Innovation, typename FilterState, typename Meas, typename Noise, typename OutlierDetection = OutlierDetectionDefault, bool isCoupled = false>
class Update: public ModelBase<typename FilterState::mtState,Innovation,Meas,Noise,FilterState::useDynamicMatrix_>, public PropertyHandler{
 public:
  static_assert(!isCoupled || Noise::D_ == FilterState::noiseExtensionDim_,"Noise Size for coupled Update must match noise extension of prediction!");
  typedef FilterState mtFilterState;
  typedef typename mtFilterState::mtState mtState;
  typedef typename mtFilterState::mtFilterCovMat mtFilterCovMat;
  typedef typename mtFilterState::mtPredictionMeas mtPredictionMeas;
  typedef typename mtFilterState::mtPredictionNoise mtPredictionNoise;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef LWFMatrix<mtNoise::D_,mtNoise::D_,mtFilterState::useDynamicMatrix_> mtUpdateNoise;
  typedef OutlierDetection mtOutlierDetection;
  static const bool coupledToPrediction_ = isCoupled;
  bool useSpecialLinearizationPoint_;
  bool useImprovedJacobian_;
  typedef ModelBase<mtState,mtInnovation,mtMeas,mtNoise,mtFilterState::useDynamicMatrix_> mtModelBase;
  typename mtModelBase::mtJacInput H_;
  typename mtModelBase::mtJacInput Hlin_;
  mtFilterCovMat boxMinusJac_;
  typename mtModelBase::mtJacNoise Hn_;
  mtUpdateNoise updnoiP_;
  mtUpdateNoise noiP_;
  LWFMatrix<mtPredictionNoise::D_,mtNoise::D_,mtFilterState::useDynamicMatrix_> preupdnoiP_;
  LWFMatrix<mtState::D_,mtInnovation::D_,mtFilterState::useDynamicMatrix_> C_;
  mtInnovation y_;
  LWFMatrix<mtInnovation::D_,mtInnovation::D_,mtFilterState::useDynamicMatrix_> Py_;
  LWFMatrix<mtInnovation::D_,mtInnovation::D_,mtFilterState::useDynamicMatrix_> Pyinv_;
  typename mtInnovation::mtDifVec innVector_;
  mtInnovation yIdentity_;
  typename mtState::mtDifVec updateVec_;
  mtState linState_;
  double updateVecNorm_;
  LWFMatrix<mtState::D_,mtInnovation::D_,mtFilterState::useDynamicMatrix_> K_;
  LWFMatrix<mtInnovation::D_,mtState::D_,mtFilterState::useDynamicMatrix_> Pyx_;
  typename mtState::mtDifVec difVecLinInv_;

  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+mtNoise::D_)+1,0,mtFilterState::useDynamicMatrix_> stateSigmaPoints_;
  SigmaPoints<mtNoise,2*mtNoise::D_+1,2*(mtState::D_+mtNoise::D_)+1,2*mtState::D_,mtFilterState::useDynamicMatrix_> stateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+mtNoise::D_)+1,2*(mtState::D_+mtNoise::D_)+1,0,mtFilterState::useDynamicMatrix_> innSigmaPoints_;
  SigmaPoints<mtNoise,2*(mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_),mtFilterState::useDynamicMatrix_> coupledStateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,0,mtFilterState::useDynamicMatrix_> coupledInnSigmaPoints_;
  SigmaPoints<LWF::VectorElement<mtState::D_>,2*mtState::D_+1,2*mtState::D_+1,0,mtFilterState::useDynamicMatrix_> updateVecSP_;
  SigmaPoints<mtState,2*mtState::D_+1,2*mtState::D_+1,0,mtFilterState::useDynamicMatrix_> posterior_;
  double alpha_;
  double beta_;
  double kappa_;
  double updateVecNormTermination_;
  int maxNumIteration_;
  mtOutlierDetection outlierDetection_;
  unsigned int numSequences;
  bool disablePreAndPostProcessingWarning_;
  Update(){
    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    updateVecNormTermination_ = 1e-6;
    maxNumIteration_  = 10;
    updnoiP_.setIdentity();
    updnoiP_ *= 0.0001;
    noiP_.setZero();
    preupdnoiP_.setZero();
    useSpecialLinearizationPoint_ = false;
    useImprovedJacobian_ = false;
    yIdentity_.setIdentity();
    updateVec_.setIdentity();
    refreshNoiseSigmaPoints();
    refreshUKFParameter();
    mtNoise n;
    n.registerCovarianceToPropertyHandler_(updnoiP_,this,"UpdateNoise.");
    doubleRegister_.registerScalar("alpha",alpha_);
    doubleRegister_.registerScalar("beta",beta_);
    doubleRegister_.registerScalar("kappa",kappa_);
    doubleRegister_.registerScalar("updateVecNormTermination",updateVecNormTermination_);
    intRegister_.registerScalar("maxNumIteration",maxNumIteration_);
    outlierDetection_.setEnabledAll(false);
    numSequences = 1;
    disablePreAndPostProcessingWarning_ = false;
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
  virtual void preProcess(mtFilterState& filterState, const mtMeas& meas, bool& isFinished){
    isFinished = false;
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: update preProcessing is not implemented!" << std::endl;
    }
  }
  virtual void postProcess(mtFilterState& filterState, const mtMeas& meas, const mtOutlierDetection& outlierDetection, bool& isFinished){
    isFinished = true;
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: update postProcessing is not implemented!" << std::endl;
    }
  }
  virtual ~Update(){}
  int performUpdate(mtFilterState& filterState, const mtMeas& meas){
    bool isFinished = true;
    int r = 0;
    do {
      preProcess(filterState,meas,isFinished);
      if(!isFinished){
        switch(filterState.mode_){
          case ModeEKF:
            r = performUpdateEKF(filterState,meas);
            break;
          case ModeUKF:
            r = performUpdateUKF(filterState,meas);
            break;
          case ModeIEKF:
            r = performUpdateIEKF(filterState,meas);
            break;
          default:
            r = performUpdateEKF(filterState,meas);
            break;
        }
      }
      postProcess(filterState,meas,outlierDetection_,isFinished);
      filterState.state_.fix();
      enforceSymmetry(filterState.cov_);
    } while (!isFinished);
    return r;
  }
  int performUpdateEKF(mtFilterState& filterState, const mtMeas& meas){
    if(!useSpecialLinearizationPoint_){
      this->jacInput(H_,filterState.state_,meas);
      Hlin_ = H_;
      this->jacNoise(Hn_,filterState.state_,meas);
      this->eval(y_,filterState.state_,meas);
    } else {
      filterState.state_.boxPlus(filterState.difVecLin_,linState_);
      this->jacInput(H_,linState_,meas);
      if(useImprovedJacobian_){
        filterState.state_.boxMinusJac(linState_,boxMinusJac_);
        Hlin_ = H_*boxMinusJac_;
      } else {
        Hlin_ = H_;
      }
      this->jacNoise(Hn_,linState_,meas);
      this->eval(y_,linState_,meas);
    }

    if(isCoupled){
      C_ = filterState.G_*preupdnoiP_*Hn_.transpose();
      Py_ = Hlin_*filterState.cov_*Hlin_.transpose() + Hn_*updnoiP_*Hn_.transpose() + Hlin_*C_ + C_.transpose()*Hlin_.transpose();
    } else {
      Py_ = Hlin_*filterState.cov_*Hlin_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    }
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection // TODO: adapt for special linearization point
    outlierDetection_.doOutlierDetection(innVector_,Py_,Hlin_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    if(isCoupled){
      K_ = (filterState.cov_*Hlin_.transpose()+C_)*Pyinv_;
    } else {
      K_ = filterState.cov_*Hlin_.transpose()*Pyinv_;
    }
    filterState.cov_ = filterState.cov_ - K_*Py_*K_.transpose();
    if(!useSpecialLinearizationPoint_){
      updateVec_ = -K_*innVector_;
    } else {
      filterState.state_.boxMinus(linState_,difVecLinInv_);
      updateVec_ = -K_*(innVector_+H_*difVecLinInv_); // includes correction for offseted linearization point, dif must be recomputed (a-b != (-(b-a)))
    }
    filterState.state_.boxPlus(updateVec_,filterState.state_);
    return 0;
  }
  int performUpdateIEKF(mtFilterState& filterState, const mtMeas& meas){
    mtState linState = filterState.state_;
    updateVecNorm_ = updateVecNormTermination_;
    for(unsigned int i=0;i<maxNumIteration_ & updateVecNorm_>=updateVecNormTermination_;i++){
      this->jacInput(H_,linState,meas);
      this->jacNoise(Hn_,linState,meas);
      this->eval(y_,linState,meas);

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
      linState.boxMinus(filterState.state_,filterState.difVecLin_);
      updateVec_ = -K_*(innVector_-H_*filterState.difVecLin_); // includes correction for offseted linearization point
      filterState.state_.boxPlus(updateVec_,linState);
      updateVecNorm_ = updateVec_.norm();
    }
    filterState.state_ = linState;
    filterState.cov_ = filterState.cov_ - K_*Py_*K_.transpose();
    return 0;
  }
  int performUpdateUKF(mtFilterState& filterState, const mtMeas& meas){
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
    posterior_.getMean(filterState.state_);
    posterior_.getCovarianceMatrix(filterState.state_,filterState.cov_);
    return 0;
  }
  template<bool IC = isCoupled, typename std::enable_if<(IC)>::type* = nullptr>
  void handleUpdateSigmaPoints(mtFilterState& filterState, const mtMeas& meas){
    coupledStateSigmaPointsNoi_.extendZeroMeanGaussian(filterState.stateSigmaPointsNoi_,updnoiP_,preupdnoiP_);
    for(unsigned int i=0;i<coupledInnSigmaPoints_.L_;i++){
      this->eval(coupledInnSigmaPoints_(i),filterState.stateSigmaPointsPre_(i),meas,coupledStateSigmaPointsNoi_(i));
    }
    coupledInnSigmaPoints_.getMean(y_);
    coupledInnSigmaPoints_.getCovarianceMatrix(y_,Py_);
    coupledInnSigmaPoints_.getCovarianceMatrix(filterState.stateSigmaPointsPre_,Pyx_);
  }
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  void handleUpdateSigmaPoints(mtFilterState& filterState, const mtMeas& meas){
    refreshNoiseSigmaPoints();
    stateSigmaPoints_.computeFromGaussian(filterState.state_,filterState.cov_);
    for(unsigned int i=0;i<innSigmaPoints_.L_;i++){
      this->eval(innSigmaPoints_(i),stateSigmaPoints_(i),meas,stateSigmaPointsNoi_(i));
    }
    innSigmaPoints_.getMean(y_);
    innSigmaPoints_.getCovarianceMatrix(y_,Py_);
    innSigmaPoints_.getCovarianceMatrix(stateSigmaPoints_,Pyx_);
  }
};

}

#endif /* LWF_UPDATEMODEL_HPP_ */
