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
#include <unordered_map>
#include "kindr/rotations/RotationEigen.hpp"
#include "PropertyHandler.hpp"

namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<typename State,unsigned int N>
class StateArray{
 public:
  static const unsigned int D_ = State::D_*N;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  State array_[N];
  StateArray(){
    setIdentity();
  }
  void boxPlus(const mtDifVec& vecIn, StateArray<State,N>& stateOut) const{
    for(unsigned int i=0;i<N;i++){
      array_[i].boxPlus(vecIn.template block<State::D_,1>(State::D_*i,0),stateOut[i]);
    }
  }
  void boxMinus(const StateArray<State,N>& stateIn, mtDifVec& vecOut) const{
    typename State::mtDifVec difVec;
    for(unsigned int i=0;i<N;i++){
      array_[i].boxMinus(stateIn[i],difVec);
      vecOut.template block<State::D_,1>(State::D_*i,0) = difVec;
    }
  }
  void print() const{
    for(unsigned int i=0;i<N;i++){
      array_[i].print();
    }
  }
  void setIdentity(){
    for(unsigned int i=0;i<N;i++){
      array_[i].setIdentity();
    }
  }
  const double& operator[](unsigned int i) const{
    assert(i<N);
    return array_[i];
  };
  double& operator[](unsigned int i){
    assert(i<N);
    return array_[i];
  };
};

template<typename State, typename... Arguments>
class ComposedState{
 public:
  static const unsigned int D_ = State::D_+ComposedState<Arguments...>::D_;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  State state_;
  ComposedState<Arguments...> subComposedState_;
  void boxPlus(const mtDifVec& vecIn, ComposedState<State,Arguments...>& stateOut){
    state_.boxPlus(vecIn.template block<State::D_,1>(0,0),stateOut.state_);
    subComposedState_.boxPlus(vecIn.template block<ComposedState<Arguments...>::D_,1>(State::D_,0),stateOut.subComposedState_);
  };
};

template<typename State>
class ComposedState<State>: public State{
};

template<unsigned int S, unsigned int V, unsigned int Q>
class StateSVQ{
 public:
  static const unsigned int S_ = S;
  static const unsigned int V_ = V;
  static const unsigned int Q_ = Q;
  static const unsigned int D_ = S_+(V_+Q_)*3;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  std::unordered_map<const void*,unsigned int> IdMap_;
  std::array<std::string,S_+V_+Q_> names_;
  StateSVQ(){
    setIdentity();
    createVarLookup();
    createDefaultNames();
  };
  StateSVQ(const StateSVQ<S,V,Q>& other){
    for(unsigned int i=0;i<S_;i++){
      s(i) = other.s(i);
    }
    for(unsigned int i=0;i<V_;i++){
      v(i) = other.v(i);
    }
    for(unsigned int i=0;i<Q_;i++){
      q(i) = other.q(i);
    };
    createVarLookup();
    createDefaultNames();
  }
  double scalarList[S_];
  Eigen::Vector3d vectorList[V_];
  rot::RotationQuaternionPD quaternionList[Q_];
  void boxPlus(const mtDifVec& vecIn, StateSVQ<S_,V_,Q_>& stateOut) const{
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
  }
  void boxMinus(const StateSVQ<S_,V_,Q_>& stateIn, mtDifVec& vecOut) const{
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
  const double& s(const std::string& str) const{
    for(unsigned int i=0;i<S_;i++){
      if(names_[i]==str){
        return s(i);
      }
    }
    assert(0);
    return s(0);
  };
  double& s(const std::string& str) {
    for(unsigned int i=0;i<S_;i++){
      if(names_[i]==str){
        return s(i);
      }
    }
    assert(0);
    return s(0);
  };
  const Eigen::Matrix<double,3,1>& v(unsigned int i) const{
    assert(i<V_);
    return vectorList[i];
  };
  Eigen::Matrix<double,3,1>& v(unsigned int i) {
    assert(i<V_);
    return vectorList[i];
  };
  const Eigen::Matrix<double,3,1>& v(const std::string& str) const{
    for(unsigned int i=0;i<V_;i++){
      if(names_[i+S_]==str){
        return v(i);
      }
    }
    assert(0);
    return v(0);
  };
  Eigen::Matrix<double,3,1>& v(const std::string& str) {
    for(unsigned int i=0;i<V_;i++){
      if(names_[i+S_]==str){
        return v(i);
      }
    }
    assert(0);
    return v(0);
  };
  const rot::RotationQuaternionPD& q(unsigned int i) const{
    assert(i<Q_);
    return quaternionList[i];
  };
  rot::RotationQuaternionPD& q(unsigned int i) {
    assert(i<Q_);
    return quaternionList[i];
  };
  const rot::RotationQuaternionPD& q(const std::string& str) const{
    for(unsigned int i=0;i<Q_;i++){
      if(names_[i+S_+V_]==str){
        return q(i);
      }
    }
    assert(0);
    return q(0);
  };
  rot::RotationQuaternionPD& q(const std::string& str) {
    for(unsigned int i=0;i<Q_;i++){
      if(names_[i+S_+V_]==str){
        return q(i);
      }
    }
    assert(0);
    return q(0);
  };
  unsigned int getId(const double& var) const{
    return IdMap_.at(static_cast<const void*>(&var));
  };
  unsigned int getId(const Eigen::Matrix<double,3,1>& var) const{
    return IdMap_.at(static_cast<const void*>(&var));
  };
  unsigned int getId(const rot::RotationQuaternionPD& var) const{
    return IdMap_.at(static_cast<const void*>(&var));
  };
  unsigned int getId(const std::string& str) const{
    for(unsigned int i=0;i<S_;i++){
      if(names_[i]==str){
        return i;
      }
    }
    for(unsigned int i=S_;i<S_+V_;i++){
      if(names_[i]==str){
        return S_+(i-S_)*3;
      }
    }
    for(unsigned int i=S_+V_;i<S_+V_+Q_;i++){
      if(names_[i]==str){
        return S_+3*V_+(i-S_-V_)*3;
      }
    }
    assert(0);
    return 0;
  };
  void createVarLookup(){
    for(unsigned int i=0;i<S_;i++){
      IdMap_[static_cast<const void*>(&s(i))] = i;
    }
    for(unsigned int i=0;i<V_;i++){
      IdMap_[static_cast<const void*>(&v(i))] = S_+3*i;
    }
    for(unsigned int i=0;i<Q_;i++){
      IdMap_[static_cast<const void*>(&q(i))] = S_+3*V_+3*i;
    }
  };
  void createDefaultNames(){
    for(unsigned int i=0;i<S_;i++){
      sName(i) = "s" + std::to_string(i);
    }
    for(unsigned int i=0;i<V_;i++){
      vName(i) = "v" + std::to_string(i);
    }
    for(unsigned int i=0;i<Q_;i++){
      qName(i) = "q" + std::to_string(i);
    }
  };
  const std::string& sName(unsigned int i) const{
    assert(i<S_);
    return names_[i];
  };
  std::string& sName(unsigned int i) {
    assert(i<S_);
    return names_[i];
  };
  const std::string& vName(unsigned int i) const{
    assert(i<V_);
    return names_[i+S_];
  };
  std::string& vName(unsigned int i) {
    assert(i<V_);
    return names_[i+S_];
  };
  const std::string& qName(unsigned int i) const{
    assert(i<Q_);
    return names_[i+S_+V_];
  };
  std::string& qName(unsigned int i) {
    assert(i<Q_);
    return names_[i+S_+V_];
  };
  StateSVQ<S,V,Q>& operator=(const StateSVQ<S,V,Q>& state){
    for(unsigned int i=0;i<S_;i++){
      s(i) = state.s(i);
    }
    for(unsigned int i=0;i<V_;i++){
      v(i) = state.v(i);
    }
    for(unsigned int i=0;i<Q_;i++){
      q(i) = state.q(i);
    }
    return *this;
  }
  static StateSVQ<S,V,Q> Identity(){
    StateSVQ<S,V,Q> identity;
    return identity;
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    for(unsigned int i=0;i<S_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + sName(i), s(i));
    }
    for(unsigned int i=0;i<V_;i++){
      mtPropertyHandler->doubleRegister_.registerVector(str + vName(i), v(i));
    }
    for(unsigned int i=0;i<Q_;i++){
      mtPropertyHandler->doubleRegister_.registerQuaternion(str + qName(i), q(i));
    };
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, mtCovMat& cov){
    for(unsigned int i=0;i<S_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + sName(i), cov(i,i));
    }
    for(unsigned int i=0;i<V_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + vName(i) + "x", cov(S_+3*i+0,S_+3*i+0));
      mtPropertyHandler->doubleRegister_.registerScalar(str + vName(i) + "y", cov(S_+3*i+1,S_+3*i+1));
      mtPropertyHandler->doubleRegister_.registerScalar(str + vName(i) + "z", cov(S_+3*i+2,S_+3*i+2));
    }
    for(unsigned int i=0;i<Q_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + qName(i) + "x", cov(S_+3*(V_+i)+0,S_+3*(V_+i)+0));
      mtPropertyHandler->doubleRegister_.registerScalar(str + qName(i) + "y", cov(S_+3*(V_+i)+1,S_+3*(V_+i)+1));
      mtPropertyHandler->doubleRegister_.registerScalar(str + qName(i) + "z", cov(S_+3*(V_+i)+2,S_+3*(V_+i)+2));
    };
  }
};

template<unsigned int N>
class VectorState{
 public:
  static const unsigned int D_ = N;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  VectorState(){
    vector_.setZero();
    createVarLookup();
  };
  VectorState(const VectorState<N>& other){
    vector_ = other.vector_;
    createVarLookup();
  };
  Eigen::Matrix<double,D_,1> vector_;
  std::unordered_map<const void*,unsigned int> IdMap_;
  void boxPlus(const mtDifVec& vecIn, VectorState<N>& stateOut) const{
    stateOut.vector_ = vector_+vecIn;
  }
  void boxMinus(const VectorState<N>& stateIn, mtDifVec& vecOut) const{
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
  VectorState<N>& operator=(const Eigen::Matrix<double,D_,1>& vec){
    vector_ = vec;
    return *this;
  };
  VectorState<N>& operator=(const VectorState<N>& state){
    vector_ = state.vector_;
    return *this;
  };
  template<unsigned int M>
  Eigen::Block<Eigen::Matrix<double,D_,1>,M,1> block(unsigned int i){
    assert(M+i<=N);
    return vector_.block<M,1>(i,0);
  }
  template<unsigned int M>
  const Eigen::Block<const Eigen::Matrix<double,D_,1>,M,1> block(unsigned int i) const{
    assert(M+i<=N);
    return vector_.block<M,1>(i,0);
  }
  unsigned int getId(const double& var) const{
    return IdMap_.at(static_cast<const void*>(&var));
  };
  template<int M>
  unsigned int getId(const Eigen::Block<const Eigen::Matrix<double,D_,1>,M,1> var) const{
    return IdMap_.at(static_cast<const void*>(&var(0)));
  };
  template<int M>
  unsigned int getId(Eigen::Block<Eigen::Matrix<double,D_,1>,M,1> var) const{
    return IdMap_.at(static_cast<const void*>(&var(0)));
  };
  void createVarLookup(){
    for(unsigned int i=0;i<D_;i++){
      IdMap_[static_cast<const void*>(&vector_(i))] = i;
    }
  };
  static VectorState<N> Identity(){
    VectorState<N> identity;
    return identity;
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    for(unsigned int i=0;i<D_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + "v" + std::to_string(i), vector_(i));
    }
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, mtCovMat& cov){
    for(unsigned int i=0;i<D_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + "v" + std::to_string(i), cov(i,i));
    }
  }
};

template<typename State1, typename State2>
class PairState{
 public:
  static const unsigned int D_ = State1::D_+State2::D_;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  PairState(){};
  PairState(const PairState<State1,State2>& other){
    state1_ = other.state1_;
    state2_ = other.state2_;
  };
  State1 state1_;
  State2 state2_;
  void boxPlus(const mtDifVec& vecIn, PairState<State1,State2>& stateOut) const{
    state1_.boxPlus(vecIn.template block<State1::D_,1>(0,0),stateOut.state1_);
    state2_.boxPlus(vecIn.template block<State2::D_,1>(State1::D_,0),stateOut.state2_);
  }
  void boxMinus(const PairState<State1,State2>& stateIn, mtDifVec& vecOut) const{
    typename State1::mtDifVec difVec1;
    state1_.boxMinus(stateIn.state1_,difVec1);
    vecOut.template block<State1::D_,1>(0,0) = difVec1;
    typename State2::mtDifVec difVec2;
    state2_.boxMinus(stateIn.state2_,difVec2);
    vecOut.template block<State2::D_,1>(State1::D_,0) = difVec2;
  }
  void print() const{
    state1_.print();
    state2_.print();
  }
  void setIdentity(){
    state1_.setIdentity();
    state2_.setIdentity();
  }
  const State1& first() const{
    return state1_;
  };
  State1& first(){
    return state1_;
  };
  const State1& second() const{
    return state2_;
  };
  State2& second(){
    return state2_;
  };
  PairState<State1,State2>& operator=(const PairState<State1,State2>& state){
    state1_ = state.state1_;
    state2_ = state.state2_;
    return *this;
  };
  static PairState<State1,State2> Identity(){
    PairState<State1,State2> identity;
    return identity;
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    first().registerToPropertyHandler(mtPropertyHandler,str + "first.");
    second().registerToPropertyHandler(mtPropertyHandler,str + "second.");
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, mtCovMat& cov){
    first().registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str + "first.",cov);
    second().registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str + "second.",cov);
  }
};

template<typename State, typename Aux> // Takes care of passing auxilliary data through boxplus, copy constructor and assignement
class AugmentedState: public State{
 public:
  typedef typename State::mtDifVec mtDifVec;
  AugmentedState(){};
  AugmentedState(const AugmentedState<State,Aux>& other): State(other){
    aux_ = other.aux_;
  };
  Aux aux_;
  void boxPlus(const mtDifVec& vecIn, AugmentedState<State,Aux>& stateOut) const{
    State::boxPlus(vecIn,stateOut);
    stateOut.aux_ = aux_;
  }
  const Aux& aux() const{
    return aux_;
  };
  Aux& aux(){
    return aux_;
  };
  AugmentedState<State,Aux>& operator=(const AugmentedState<State,Aux>& other){
    State::operator=(other);
    aux_ = other.aux_;
    return *this;
  };
  static AugmentedState<State,Aux> Identity(){
    AugmentedState<State,Aux> identity;
    return identity;
  }
};

static Eigen::Matrix3d Lmat (Eigen::Vector3d a) { // TODO
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

#endif /* STATE_HPP_ */
