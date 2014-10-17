/*
 * SigmaPoints.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef SIGMAPOINTS_HPP_
#define SIGMAPOINTS_HPP_

#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace LWF{

template<typename State, unsigned int N, unsigned int L, unsigned int O>
class SigmaPoints{
 public:
  typedef State mtState;
  static const unsigned int N_ = N;
  static const unsigned int L_ = L;
  static const unsigned int O_ = O;
  double wm_ = 1.0;
  double wc_ = 1.0;
  double wc0_ = 1.0;
  double gamma_ = 1.0;
  mtState sigmaPoints_[N];
  SigmaPoints(){
    assert(N_+O_<=L_);
  };
  mtState getMean() const{
    typename mtState::mtDiffVec vec;
    typename mtState::mtDiffVec vecTemp;
    vec.setZero();
    for(unsigned int i=1;i<N_;i++){
      sigmaPoints_[i].boxMinus(sigmaPoints_[0],vecTemp);
      vec = vec + wm_*vecTemp;
    }
    mtState mean;
    sigmaPoints_[0].boxPlus(vec,mean);
    return mean;
  };
  typename mtState::mtCovMat getCovarianceMatrix() const{
    mtState mean = getMean();
    return getCovarianceMatrix(mean);
  };
  typename mtState::mtCovMat getCovarianceMatrix(const mtState& mean) const{
    typename mtState::mtCovMat C;
    typename mtState::mtDiffVec vec;
    sigmaPoints_[0].boxMinus(mean,vec);
    C = vec*vec.transpose()*(wc0_+ wc_*(L_-N_));
    for(unsigned int i=1;i<N_;i++){
      sigmaPoints_[i].boxMinus(mean,vec);
      C += vec*vec.transpose()*wc_;
    }
    return C;
  };
  template<typename State2, unsigned int N2, unsigned int O2>
  Eigen::Matrix<double,mtState::D_,State2::D_> getCovarianceMatrix(const SigmaPoints<State2,N2,L_,O2>& sigmaPoints2) const{
    mtState mean1 = getMean();
    State2 mean2 = sigmaPoints2.getMean();
    return getCovarianceMatrix(sigmaPoints2,mean1,mean2);
  };
  template<typename State2, unsigned int N2, unsigned int O2>
  Eigen::Matrix<double,mtState::D_,State2::D_> getCovarianceMatrix(const SigmaPoints<State2,N2,L_,O2>& sigmaPoints2, const mtState& mean1, const State2& mean2) const{
    Eigen::Matrix<double,mtState::D_,State2::D_> C;
    typename mtState::mtDiffVec vec1;
    typename State2::mtDiffVec vec2;
    (*this)(0).boxMinus(mean1,vec1);
    sigmaPoints2(0).boxMinus(mean2,vec2);
    C = vec1*vec2.transpose()*wc0_;
    for(unsigned int i=1;i<L_;i++){
      (*this)(i).boxMinus(mean1,vec1);
      sigmaPoints2(i).boxMinus(mean2,vec2);
      C += vec1*vec2.transpose()*wc_;
    }
    return C;
  };
  void computeFromGaussian(const mtState mean, const typename mtState::mtCovMat &P){
    assert(N_==2*mtState::D_+1);
    Eigen::LDLT<typename mtState::mtCovMat> ldltOfP(P);
//    typename mtState::mtCovMat ldltP = ldltOfP.transpositionsP();
    typename mtState::mtCovMat ldltL = ldltOfP.matrixL();
    typename mtState::mtCovMat ldltD = ldltOfP.vectorD().asDiagonal();
    for(unsigned int i=0;i<mtState::D_;i++){
      if(ldltD(i,i)>0){
        ldltD(i,i) = std::sqrt(ldltD(i,i));
      } else if(ldltD(i,i)==0) {
        ldltD(i,i) = 0.0;
        std::cout << "CAUTION: Covariance matrix is only positive SEMIdefinite" << std::endl;
      } else {
        ldltD(i,i) = 0.0;
        std::cout << "ERROR: Covariance matrix is not positive semidefinite" << std::endl;
      }
    }
    if(ldltOfP.info()==Eigen::NumericalIssue) std::cout << "Numerical issues while computing Cholesky Matrix" << std::endl;


    typename mtState::mtCovMat S = ldltOfP.transpositionsP().transpose()*ldltL*ldltD;

    sigmaPoints_[0] = mean;
    for(unsigned int i=0;i<mtState::D_;i++){
      mean.boxPlus(S.col(i)*gamma_,sigmaPoints_[i+1]);
      mean.boxPlus(-S.col(i)*gamma_,sigmaPoints_[i+1+mtState::D_]);
    }
  };
  void computeFromGaussian(const mtState mean, const typename mtState::mtCovMat &P, const typename mtState::mtCovMat &Q){
    // TODO: adapt to above
//    assert(N_==2*mtState::D_+1);
//    Eigen::LLT<typename mtState::mtCovMat> lltOfP(Q.transpose()*P*Q);
//    typename mtState::mtCovMat S = Q*lltOfP.matrixL();
//    if(lltOfP.info()==Eigen::NumericalIssue) std::cout << "Numerical issues while computing Cholesky Matrix" << std::endl;
//
//    sigmaPoints_[0] = mean;
//    for(unsigned int i=0;i<mtState::D_;i++){
//      mean.boxPlus(S.col(i)*gamma_,sigmaPoints_[i+1]);
//      mean.boxPlus(-S.col(i)*gamma_,sigmaPoints_[i+1+mtState::D_]);
//    }
  };
  void computeFromZeroMeanGaussian(const typename mtState::mtCovMat &P){
    mtState identity;		// is initialized to 0 by default constructors
    computeFromGaussian(identity,P);
  };
  const mtState& operator()(unsigned int i) const{
    assert(i<L_);
    if(i<O_){
      return sigmaPoints_[0];
    } else if(i<O_+N_){
      return sigmaPoints_[i-O_];
    } else {
      return sigmaPoints_[0];
    }
  };
  mtState& operator()(unsigned int i) {
    assert(i<L_);
    if(i<O_){
      return sigmaPoints_[0];
    } else if(i<O_+N_){
      return sigmaPoints_[i-O_];
    } else {
      return sigmaPoints_[0];
    }
  };
  void computeParameter(double alpha,double beta,double kappa){
    const unsigned int D = (L_-1)/2;
    const double lambda = alpha*alpha*(D+kappa)-D;
    gamma_ = sqrt(lambda + D);
    wm_ = 1/(2*(D+lambda));
    wc_ = wm_;
    wc0_ = lambda/(D+lambda)+(1-alpha*alpha+beta);
  };
};

}

#endif /* SIGMAPOINTS_HPP_ */
