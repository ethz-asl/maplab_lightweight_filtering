/*
 * State.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef STATE_HPP_
#define STATE_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/rotations/RotationEigen.hpp"
namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<unsigned int S, unsigned int V, unsigned int Q>
class StateSVQ{
 public:
  static const unsigned int S_ = S;
  static const unsigned int V_ = V;
  static const unsigned int Q_ = Q;
  static const unsigned int D_ = S_+(V_+Q_)*3;
  typedef Eigen::Matrix<double,D_,1> DiffVec;
  typedef Eigen::Matrix<double,D_,D_> CovMat;
  StateSVQ(){
    t_ = 0.0;
    setIdentity();
  };
  double t_;
  double scalarList[S_];
  Eigen::Vector3d vectorList[V_];
  rot::RotationQuaternionPD quaternionList[Q_];
  void boxPlus(const DiffVec& vecIn, StateSVQ<S_,V_,Q_>& stateOut) const{
    unsigned int index = 0;
    for(unsigned int i=0;i<S_;i++){
      stateOut.scalarList[i] = scalarList[i]+vecIn(index);
      index += 1;
    }
    for(unsigned int i=0;i<V_;i++){
      stateOut.vectorList[i] = vectorList[i]+vecIn.block(index,0,3,1);
      index += 3;
    }
    for(unsigned int i=0;i<Q_;i++){
      stateOut.quaternionList[i] = quaternionList[i].boxPlus(vecIn.block(index,0,3,1));
      index += 3;
    }
    stateOut.t_ = t_;
  }
  void boxMinus(const StateSVQ<S_,V_,Q_>& stateIn, DiffVec& vecOut) const{
    unsigned int index = 0;
    for(unsigned int i=0;i<S_;i++){
      vecOut(index) = scalarList[i]-stateIn.scalarList[i];
      index += 1;
    }
    for(unsigned int i=0;i<V_;i++){
      vecOut.block(index,0,3,1) = vectorList[i]-stateIn.vectorList[i];
      index += 3;
    }
    for(unsigned int i=0;i<Q_;i++){
      vecOut.block(index,0,3,1) = quaternionList[i].boxMinus(stateIn.quaternionList[i]);
      index += 3;
    }
  }
  void print() const{
    std::cout << "Scalars:" << std::endl;
    for(unsigned int i=0;i<S_;i++){
      std::cout << s(i) << std::endl;
    }
    std::cout << "Vectors:" << std::endl;
    for(unsigned int i=0;i<V_;i++){
      std::cout << v(i).transpose() << std::endl;
    }
    std::cout << "Quaternions:" << std::endl;
    for(unsigned int i=0;i<Q_;i++){
      std::cout << q(i) << std::endl;
    }
  }
  void setIdentity(){
    for(unsigned int i=0;i<S_;i++){
      scalarList[i] = 0.0;
    }
    for(unsigned int i=0;i<V_;i++){
      vectorList[i].setZero();
    }
    for(unsigned int i=0;i<Q_;i++){
      quaternionList[i].setIdentity();
    }
  }
  void fix(){
    for(unsigned int i=0;i<Q_;i++){
      q(i).fix();
    }
  }
  const double& s(unsigned int i) const{
    assert(i<S_);
    return scalarList[i];
  };
  double& s(unsigned int i) {
    assert(i<S_);
    return scalarList[i];
  };
  const Eigen::Matrix<double,3,1>& v(unsigned int i) const{
    assert(i<V_);
    return vectorList[i];
  };
  Eigen::Matrix<double,3,1>& v(unsigned int i) {
    assert(i<V_);
    return vectorList[i];
  };
  const rot::RotationQuaternionPD& q(unsigned int i) const{
    assert(i<Q_);
    return quaternionList[i];
  };
  rot::RotationQuaternionPD& q(unsigned int i) {
    assert(i<Q_);
    return quaternionList[i];
  };
};

template<unsigned int N>
class VectorState{
 public:
  static const unsigned int D_ = N;
  typedef Eigen::Matrix<double,D_,1> DiffVec;
  typedef Eigen::Matrix<double,D_,D_> CovMat;
  VectorState(){
    vector_.setZero();
    t_ = 0.0;
  };
  Eigen::Matrix<double,D_,1> vector_;
  double t_;
  void boxPlus(const DiffVec& vecIn, VectorState<N>& stateOut) const{
    stateOut.vector_ = vector_+vecIn;
    stateOut.t_ = t_;
  }
  void boxMinus(const VectorState<N>& stateIn, DiffVec& vecOut) const{
    vecOut = vector_-stateIn.vector_;
  }
  void print() const{
    std::cout << "Vector:" << vector_.transpose() << std::endl;
  }
  void setIdentity(){
    vector_.setZero();
  }
  const double& operator[](unsigned int i) const{
    assert(i<D_);
    return vector_(i);
  };
  double& operator[](unsigned int i){
    assert(i<D_);
    return vector_(i);
  };
};

}

#endif /* STATE_HPP_ */
