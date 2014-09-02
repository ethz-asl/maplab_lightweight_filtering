/*
 * SigmaPoints.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef SIGMAPOINTS_HPP_
#define SIGMAPOINTS_HPP_

#include "State.hpp"
#include <Eigen/Dense>
#include <vector>

namespace LightWeightUKF{

template<typename State>
class SigmaPoints{
 public:
  unsigned int N_ = 0;
  unsigned int L_ = 0;
  unsigned int offset_ = 0;
  double wm_ = 1.0;
  double wc_ = 1.0;
  double wc0_ = 1.0;
  double gamma_ = 1.0;
  std::vector<State> sigmaPoints_;
  SigmaPoints(){};
  SigmaPoints(unsigned int N, unsigned int L, unsigned int offset){
    resize(N,L,offset);
    const double alpha = 1e-3;
    const double beta = 2.0;
    const double kappa = 0.0;
    const unsigned int D = (L-1)/2;
    const double lambda = alpha*alpha*(D+kappa)-D;
    gamma_ = sqrt(lambda + D);
    wm_ = 1/(2*(D+lambda));
    wc_ = wm_;
    wc0_ = lambda/(D+lambda)+(1-alpha*alpha+beta);
  };
  void resize(unsigned int N, unsigned int L, unsigned int offset){
    assert(N+offset<=L);
    sigmaPoints_.resize(N);
    N_ = N;
    L_ = L;
    offset_ = offset;
  };
  State getMean() const{
    typename State::DiffVec vec;
    typename State::DiffVec vecTemp;
    vec.setZero();
    for(unsigned int i=1;i<N_;i++){
      sigmaPoints_[i].boxminus(sigmaPoints_[0],vecTemp);
      vec = vec + wm_*vecTemp;
    }
    State mean;
    sigmaPoints_[0].boxplus(vec,mean);
    return mean;
  };
  typename State::CovMat getCovarianceMatrix() const{
    State mean = getMean();
    return getCovarianceMatrix(mean);
  };
  typename State::CovMat getCovarianceMatrix(const State& mean) const{
    typename State::CovMat C;
    typename State::DiffVec vec;
    sigmaPoints_[0].boxminus(mean,vec);
    C = vec*vec.transpose()*(wc0_+ wc_*(L_-N_));
    for(unsigned int i=1;i<N_;i++){
      sigmaPoints_[i].boxminus(mean,vec);
      C += vec*vec.transpose()*wc_;
    }
    return C;
  };
  template<typename State2>
  Eigen::Matrix<double,State::D_,State2::D_> getCovarianceMatrix(const SigmaPoints<State2>& sigmaPoints2) const{
    State mean1 = getMean();
    State2 mean2 = sigmaPoints2.getMean();
    return getCovarianceMatrix(sigmaPoints2,mean1,mean2);
  };
  template<typename State2>
  Eigen::Matrix<double,State::D_,State2::D_> getCovarianceMatrix(const SigmaPoints<State2>& sigmaPoints2, const State& mean1, const State2& mean2) const{
    assert(L_==sigmaPoints2.L_);
    Eigen::Matrix<double,State::D_,State2::D_> C;
    typename State::DiffVec vec1;
    typename State2::DiffVec vec2;
    (*this)(0).boxminus(mean1,vec1);
    sigmaPoints2(0).boxminus(mean2,vec2);
    C = vec1*vec2.transpose()*wc0_;
    for(unsigned int i=1;i<L_||i<sigmaPoints2.N_;i++){
      (*this)(i).boxminus(mean1,vec1);
      sigmaPoints2(i).boxminus(mean2,vec2);
      C += vec1*vec2.transpose()*wc_;
    }
    return C;
  };
  void computeFromGaussian(const State mean, const typename State::CovMat &P){
    assert(N_==2*State::D_+1);
    Eigen::LLT<typename State::CovMat> lltOfP(P);
    typename State::CovMat S = lltOfP.matrixL();
    if(lltOfP.info()==Eigen::NumericalIssue) std::cout << "Numerical issues while computing Cholesky Matrix" << std::endl;

    sigmaPoints_[0] = mean;
    for(unsigned int i=0;i<State::D_;i++){
      mean.boxplus(S.col(i)*gamma_,sigmaPoints_[i+1]);
      mean.boxplus(-S.col(i)*gamma_,sigmaPoints_[i+1+State::D_]);
    }
  };
  void computeFromGaussian(const State mean, const typename State::CovMat &P, const typename State::CovMat &Q){
    assert(N_==2*State::D_+1);
    Eigen::LLT<typename State::CovMat> lltOfP(Q.transpose()*P*Q);
    typename State::CovMat S = Q*lltOfP.matrixL();
    if(lltOfP.info()==Eigen::NumericalIssue) std::cout << "Numerical issues while computing Cholesky Matrix" << std::endl;

    sigmaPoints_[0] = mean;
    for(unsigned int i=0;i<State::D_;i++){
      mean.boxplus(S.col(i)*gamma_,sigmaPoints_[i+1]);
      mean.boxplus(-S.col(i)*gamma_,sigmaPoints_[i+1+State::D_]);
    }
  };
  void computeFromZeroMeanGaussian(const typename State::CovMat &P){
    State identity;		// is initialized to 0 by default constructors
    computeFromGaussian(identity,P);
  };
  const State& operator()(unsigned int i) const{
    assert(i<L_);
    if(i<offset_){
      return sigmaPoints_[0];
    } else if(i<offset_+N_){
      return sigmaPoints_[i-offset_];
    } else {
      return sigmaPoints_[0];
    }
  };
  State& operator()(unsigned int i) {
    assert(i<L_);
    if(i<offset_){
      return sigmaPoints_[0];
    } else if(i<offset_+N_){
      return sigmaPoints_[i-offset_];
    } else {
      return sigmaPoints_[0];
    }
  };
  void setParameter(double wm,double wc,double wc0,double gamma){
    wm_ = wm;
    wc_ = wc;
    wc0_ = wc0;
    gamma_ = gamma;
  };
};

}

#endif /* SIGMAPOINTS_HPP_ */
