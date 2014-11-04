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
#include <boost/any.hpp>
#include <type_traits>

namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<typename DERIVED, unsigned int D, unsigned int E = 1>
class StateBase{ // TODO: normal vector
 public:
  StateBase(){};
  virtual ~StateBase(){};
  static const unsigned int D_ = D;
  static const unsigned int E_ = E;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  std::string name_;
  virtual void boxPlus(const mtDifVec& vecIn, DERIVED& stateOut) const = 0;
  virtual void boxMinus(const DERIVED& stateIn, mtDifVec& vecOut) const = 0;
  virtual void print() const = 0;
  virtual void setIdentity() = 0;
  virtual void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str) = 0;
  virtual void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov) = 0; // TODO, fix, cov is chosen as const in order to handle block matrices
  virtual void createDefaultNames(const std::string& str) = 0;
  static DERIVED Identity(){
    DERIVED identity;
    identity.setIdentity();
    return identity;
  }
  DERIVED& operator=(const DERIVED& other){
    other.swap(*this);
    return *this;
  }
};

class ScalarState: public StateBase<ScalarState,1>{
 public:
  typedef StateBase<ScalarState,1> Base;
  using Base::D_;
  using Base::E_;
  using typename  Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  ScalarState(){}
  ScalarState(const ScalarState& other){
    s_ = other.s_;
  }
  double s_;
  void boxPlus(const mtDifVec& vecIn, ScalarState& stateOut) const{
    stateOut.s_ = s_ + vecIn(0);
  }
  void boxMinus(const ScalarState& stateIn, mtDifVec& vecOut) const{
    vecOut(0) = s_ - stateIn.s_;
  }
  void print() const{
    std::cout << s_ << std::endl;
  }
  void setIdentity(){
    s_ = 0.0;
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    mtPropertyHandler->doubleRegister_.registerScalar(str + name_, s_);
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){
    mtPropertyHandler->doubleRegister_.registerDiagonalMatrix(str + name_, const_cast<mtCovMat&>(cov));
  }
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  ScalarState& getState(){
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const ScalarState& getState() const{
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  double& getValue(){
    return s_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const double& getValue() const{
    return s_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  static unsigned int getId(){
    return 0;
  };
  void createDefaultNames(const std::string& str){
    name_ = str;
  };
};

template<unsigned int N>
class VectorState: public StateBase<VectorState<N>,N>{ // TODO: rename
 public:
  typedef StateBase<VectorState<N>,N> Base;
  using Base::D_;
  using Base::E_;
  using typename  Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  static const unsigned int N_ = N;
  VectorState(){}
  VectorState(const VectorState<N>& other){
    v_ = other.v_;
  }
  Eigen::Matrix<double,N_,1> v_;
  void boxPlus(const mtDifVec& vecIn, VectorState<N>& stateOut) const{
    stateOut.v_ = v_ + vecIn;
  }
  void boxMinus(const VectorState<N>& stateIn, mtDifVec& vecOut) const{
    vecOut = v_ - stateIn.v_;
  }
  void print() const{
    std::cout << v_.transpose() << std::endl;
  }
  void setIdentity(){
    v_.setZero();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    for(unsigned int i = 0;i<N_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + name_ + "_" + std::to_string(i), v_(i));
    }
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){
    mtPropertyHandler->doubleRegister_.registerDiagonalMatrix(str + name_, const_cast<mtCovMat&>(cov));
  }
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  VectorState<N>& getState(){
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const VectorState<N>& getState() const{
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  Eigen::Matrix<double,N_,1>& getValue(){
    return v_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const Eigen::Matrix<double,N_,1>& getValue() const{
    return v_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  static unsigned int getId(){
    return 0;
  };
  void createDefaultNames(const std::string& str){
    name_ = str;
  };
};

class QuaternionState: public StateBase<QuaternionState,3>{
 public:
  typedef StateBase<QuaternionState,3> Base;
  using Base::D_;
  using Base::E_;
  using typename  Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  QuaternionState(){}
  QuaternionState(const QuaternionState& other){
    q_ = other.q_;
  }
  rot::RotationQuaternionPD q_;
  void boxPlus(const mtDifVec& vecIn, QuaternionState& stateOut) const{
    stateOut.q_ = q_.boxPlus(vecIn);
  }
  void boxMinus(const QuaternionState& stateIn, mtDifVec& vecOut) const{
    vecOut = q_.boxMinus(stateIn.q_);
  }
  void print() const{
    std::cout << q_ << std::endl;
  }
  void setIdentity(){
    q_.setIdentity();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    mtPropertyHandler->doubleRegister_.registerQuaternion(str + name_, q_);
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){
    mtPropertyHandler->doubleRegister_.registerDiagonalMatrix(str + name_, const_cast<mtCovMat&>(cov));
  }
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  QuaternionState& getState(){
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const QuaternionState& getState() const{
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  rot::RotationQuaternionPD& getValue(){
    return q_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const rot::RotationQuaternionPD& getValue() const{
    return q_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  static unsigned int getId(){
    return 0;
  };
  void createDefaultNames(const std::string& str){
    name_ = str;
  };
};

template<typename State,unsigned int N>
class StateArray: public StateBase<StateArray<State,N>,State::D_*N,State::E_*N>{
 public:
  typedef StateBase<StateArray<State,N>,State::D_*N,State::E_*N> Base;
  using Base::D_;
  using Base::E_;
  using typename  Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  State array_[N];
  StateArray(){}
  StateArray(const StateArray<State,N>& other){
    for(unsigned int i=0;i<N;i++){
      array_[i] = other.array_[i];
    }
  }
  void boxPlus(const mtDifVec& vecIn, StateArray<State,N>& stateOut) const{
//    for(unsigned int i=0;i<N;i++){
//      array_[i].boxPlus(vecIn.template block<State::D_,1>(State::D_*i,0),stateOut.array_[i]);
//    }
    boxPlusRecursive<0>(vecIn,stateOut);
  }
  template<unsigned int i, typename std::enable_if<(i<N)>::type* = nullptr>
  void boxPlusRecursive(const mtDifVec& vecIn, StateArray<State,N>& stateOut) const{
    if(State::D_>0){
      array_[i].boxPlus(vecIn.template block<State::D_,1>(State::D_*i,0),stateOut.array_[i]);
    }
    boxPlusRecursive<i+1>(vecIn,stateOut);
  }
  template<unsigned int i, typename std::enable_if<(i>=N)>::type* = nullptr>
  void boxPlusRecursive(const mtDifVec& vecIn, StateArray<State,N>& stateOut) const{}
  void boxMinus(const StateArray<State,N>& stateIn, mtDifVec& vecOut) const{
//    typename State::mtDifVec difVec;
//    for(unsigned int i=0;i<N;i++){
//      array_[i].boxMinus(stateIn.array_[i],difVec);
//      vecOut.template block<State::D_,1>(State::D_*i,0) = difVec; // TODO: fix for N = 0
//    }
    boxMinusRecursive<0>(stateIn,vecOut);
  }
  template<unsigned int i, typename std::enable_if<(i<N)>::type* = nullptr>
  void boxMinusRecursive(const StateArray<State,N>& stateIn, mtDifVec& vecOut) const{
    typename State::mtDifVec difVec;
    array_[i].boxMinus(stateIn.array_[i],difVec);
    if(State::D_>0){
      vecOut.template block<State::D_,1>(State::D_*i,0) = difVec;
    }
    boxMinusRecursive<i+1>(stateIn,vecOut);
  }
  template<unsigned int i, typename std::enable_if<(i>=N)>::type* = nullptr>
  void boxMinusRecursive(const StateArray<State,N>& stateIn, mtDifVec& vecOut) const{}
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
  void createDefaultNames(const std::string& str){
    name_ = str;
    for(unsigned int i=0;i<N;i++){
      array_[i].createDefaultNames(str + "_" + std::to_string(i));
    }
  };
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    for(unsigned int i=0;i<N;i++){
      array_[i].registerToPropertyHandler(mtPropertyHandler,str);
    }
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){
//    for(unsigned int i=0;i<N;i++){
//      array_[i].registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str,cov.template block<State::D_,State::D_>(i*State::D_,i*State::D_));
//    }
    registerDiagonalMatrixToPropertyHandlerRecursive<0>(mtPropertyHandler,str,cov);
  }
  template<unsigned int i, typename std::enable_if<(i<N)>::type* = nullptr>
  void registerDiagonalMatrixToPropertyHandlerRecursive(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){
    if(State::D_>0){
      array_[i].registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str,cov.template block<State::D_,State::D_>(i*State::D_,i*State::D_));
    }
    registerDiagonalMatrixToPropertyHandlerRecursive<i+1>(mtPropertyHandler,str,cov);
  }
  template<unsigned int i, typename std::enable_if<(i>=N)>::type* = nullptr>
  void registerDiagonalMatrixToPropertyHandlerRecursive(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){}
  template<unsigned int i, typename std::enable_if<(i<N)>::type* = nullptr>
  State& getState(){
    return array_[i];
  };
  template<unsigned int i, typename std::enable_if<(i<N)>::type* = nullptr>
  const State& getState() const{
    return array_[i];
  };
  template<unsigned int i, typename std::enable_if<(i<E_)>::type* = nullptr>
  auto getValue() -> decltype (array_[i/State::E_].template getValue<i%State::E_>())& {
    return array_[i/State::E_].getValue<i%State::E_>();
  };
  template<unsigned int i, typename std::enable_if<(i<E_)>::type* = nullptr>
  const auto getValue() const -> decltype (array_[i/State::E_].template getValue<i%State::E_>())& {
    return array_[i/State::E_].getValue<i%State::E_>();
  };
  template<unsigned int i, typename std::enable_if<(i<E_)>::type* = nullptr>
  static unsigned int getId(){
    return (i/State::E_)*State::D_ + State::template getId<i%State::E_>();
  };
};

template<typename State, typename... Arguments>
class ComposedState: public StateBase<ComposedState<State,Arguments...>,State::D_+ComposedState<Arguments...>::D_,State::E_+ComposedState<Arguments...>::E_>{
 public:
  typedef StateBase<ComposedState<State,Arguments...>,State::D_+ComposedState<Arguments...>::D_,State::E_+ComposedState<Arguments...>::E_> Base;
  using Base::D_;
  using Base::E_;
  using typename Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  State state_;
  ComposedState<Arguments...> subComposedState_;
  ComposedState(){}
  ComposedState(const ComposedState<State,Arguments...>& other): state_(other.state_), subComposedState_(other.subComposedState_){}
  void boxPlus(const mtDifVec& vecIn, ComposedState<State,Arguments...>& stateOut) const{
    if(State::D_>0){
      state_.boxPlus(vecIn.template block<State::D_,1>(0,0),stateOut.state_);
    }
    if(ComposedState<Arguments...>::D_>0){
      subComposedState_.boxPlus(vecIn.template block<ComposedState<Arguments...>::D_,1>(State::D_,0),stateOut.subComposedState_);
    }
  };
  void boxMinus(const ComposedState<State,Arguments...>& stateIn, mtDifVec& vecOut) const{
    if(State::D_>0){
      typename State::mtDifVec difVec1;
      state_.boxMinus(stateIn.state_,difVec1);
      vecOut.template block<State::D_,1>(0,0) = difVec1;
    }
    if(ComposedState<Arguments...>::D_>0){
      typename ComposedState<Arguments...>::mtDifVec difVec2;
      subComposedState_.boxMinus(stateIn.subComposedState_,difVec2);
      vecOut.template block<ComposedState<Arguments...>::D_,1>(State::D_,0) = difVec2;
    }
  };
  void print() const{
    state_.print();
    subComposedState_.print();
  }
  void setIdentity(){
    state_.setIdentity();
    subComposedState_.setIdentity();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    state_.registerToPropertyHandler(mtPropertyHandler,str);
    subComposedState_.registerToPropertyHandler(mtPropertyHandler,str);
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){
    if(State::D_>0){
      state_.registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str,cov.template block<State::D_,State::D_>(0,0));
    }
    if(ComposedState<Arguments...>::D_>0){
      subComposedState_.registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str,cov.template block<ComposedState<Arguments...>::D_,ComposedState<Arguments...>::D_>(State::D_,State::D_));
    }
  }
  void createDefaultNames(const std::string& str){
    createDefaultNamesWithIndex(str);
  };
  void createDefaultNamesWithIndex(const std::string& str, unsigned int i = 0){
    name_ = str;
    state_.createDefaultNames(str + "_" + std::to_string(i));
    subComposedState_.createDefaultNamesWithIndex(str,i+1);
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  auto getState() -> decltype (state_)& {
    return state_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const auto getState() const -> decltype (state_)& {
    return state_;
  };
  template<unsigned int i, typename std::enable_if<(i>0)>::type* = nullptr>
  auto getState() -> decltype (subComposedState_.getState<i-1>())& {
    return subComposedState_.getState<i-1>();
  };
  template<unsigned int i, typename std::enable_if<(i>0)>::type* = nullptr>
  const auto getState() const -> decltype (subComposedState_.getState<i-1>())& {
    return subComposedState_.getState<i-1>();
  };
  template<unsigned int i, typename std::enable_if<(i<State::E_)>::type* = nullptr>
  auto getValue() -> decltype (state_.template getValue<i>())& {
    return state_.getValue<i>();
  };
  template<unsigned int i, typename std::enable_if<(i<State::E_)>::type* = nullptr>
  const auto getValue() const -> decltype (state_.template getValue<i>())& {
    return state_.getValue<i>();
  };
  template<unsigned int i, typename std::enable_if<(i>=State::E_)>::type* = nullptr>
  auto getValue() -> decltype (subComposedState_.getValue<i-State::E_>())& {
    return subComposedState_.getValue<i-State::E_>();
  };
  template<unsigned int i, typename std::enable_if<(i>=State::E_)>::type* = nullptr>
  const auto getValue() const -> decltype (subComposedState_.getValue<i-State::E_>())& {
    return subComposedState_.getValue<i-State::E_>();
  };
  template<unsigned int i, typename std::enable_if<(i<State::E_)>::type* = nullptr>
  static unsigned int getId(){
    return State::template getId<i>();
  };
  template<unsigned int i, typename std::enable_if<(i>=State::E_ & i<E_)>::type* = nullptr>
  static unsigned int getId(){
    return State::D_ + ComposedState<Arguments...>::template getId<i-State::E_>();
  };
};

template<typename State>
class ComposedState<State>: public StateBase<ComposedState<State>,State::D_,State::E_>{
 public:
  typedef StateBase<ComposedState<State>,State::D_,State::E_> Base;
  using Base::D_;
  using Base::E_;
  using typename Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  State state_;
  ComposedState(){}
  ComposedState(const ComposedState<State>& other): state_(other.state_){}
  void boxPlus(const mtDifVec& vecIn, ComposedState<State>& stateOut) const{
    if(State::D_>0){
      state_.boxPlus(vecIn.template block<State::D_,1>(0,0),stateOut.state_);
    }
  };
  void boxMinus(const ComposedState<State>& stateIn, mtDifVec& vecOut) const{
    state_.boxMinus(stateIn.state_,vecOut);
  };
  void print() const{
    state_.print();
  }
  void setIdentity(){
    state_.setIdentity();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    state_.registerToPropertyHandler(mtPropertyHandler,str);
  }
  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, const mtCovMat& cov){
    state_.registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str,cov);
  }
  void createDefaultNames(const std::string& str){
    createDefaultNamesWithIndex(str);
  };
  void createDefaultNamesWithIndex(const std::string& str, unsigned int i = 0){
    name_ = str;
    state_.createDefaultNames(str + "_" + std::to_string(i));
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  auto getState() -> decltype (state_)& {
    return state_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const auto getState() const -> decltype (state_)& {
    return state_;
  };
  template<unsigned int i, typename std::enable_if<(i<State::E_)>::type* = nullptr>
  auto getValue() -> decltype (state_.template getValue<i>())& {
    return state_.getValue<i>();
  };
  template<unsigned int i, typename std::enable_if<(i<State::E_)>::type* = nullptr>
  const auto getValue() const -> decltype (state_.template getValue<i>())& {
    return state_.getValue<i>();
  };
  template<unsigned int i, typename std::enable_if<(i<State::E_)>::type* = nullptr>
  static unsigned int getId(){
    return State::template getId<i>();
  };
};

//template<unsigned int S, unsigned int V, unsigned int Q>
//class StateSVQNew: public ComposedState<StateArray<ScalarState,S>,StateArray<Vector3dState,V>,StateArray<Vector3dState,Q>>{
// public:
//  using ComposedState<StateArray<ScalarState,S>,StateArray<Vector3dState,V>,StateArray<Vector3dState,Q>>::state_;
//  const double& s(unsigned int i) const{
//    return state_[i].s_;
//  };
//  double& s(unsigned int i) {
//    return state_[i].s_;
//  };
//  const double& s(const std::string& str) const{
//    return state_.get(str).s_;
//  };
//  double& s(const std::string& str) {
//    return state_.get(str).s_;
//  }
//  const std::string& sName(unsigned int i) const{
//    return state_.getName(i);
//  };
//  std::string& sName(unsigned int i) {
//    return state_.getName(i);
//  };
//};

template<unsigned int S, unsigned int V, unsigned int Q>
class StateSVQ: public ComposedState<StateArray<ScalarState,S>,StateArray<VectorState<3>,V>,StateArray<QuaternionState,Q>>{
 public:
  static const unsigned int S_ = S;
  static const unsigned int V_ = V;
  static const unsigned int Q_ = Q;
  typedef ComposedState<StateArray<ScalarState,S>,StateArray<VectorState<3>,V>,StateArray<QuaternionState,Q>> Base;
  using Base::D_;
  using Base::E_;
  using typename Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  using Base::getState;
  StateSVQ(){}
  StateSVQ(const StateSVQ& other): Base(other){}
  const double& s(unsigned int i) const{
    return this->template getState<0>().array_[i].s_;
  };
  double& s(unsigned int i) {
    return this->template getState<0>().array_[i].s_;
  };
  const Eigen::Matrix<double,3,1>& v(unsigned int i) const{
    return this->template getState<1>().array_[i].v_;
  };
  Eigen::Matrix<double,3,1>& v(unsigned int i) {
    return this->template getState<1>().array_[i].v_;
  };
  const rot::RotationQuaternionPD& q(unsigned int i) const{
    return this->template getState<2>().array_[i].q_;
  };
  rot::RotationQuaternionPD& q(unsigned int i) {
    return this->template getState<2>().array_[i].q_;
  };
};

//template<unsigned int N>
//class VectorState{
// public:
//  static const unsigned int D_ = N;
//  typedef Eigen::Matrix<double,D_,1> mtDifVec;
//  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
//  std::string name_;
//  VectorState(){
//    vector_.setZero();
//    createVarLookup();
//  };
//  VectorState(const VectorState<N>& other){
//    vector_ = other.vector_;
//    createVarLookup();
//  };
//  Eigen::Matrix<double,D_,1> vector_;
//  std::unordered_map<const void*,unsigned int> IdMap_;
//  void boxPlus(const mtDifVec& vecIn, VectorState<N>& stateOut) const{
//    stateOut.vector_ = vector_+vecIn;
//  }
//  void boxMinus(const VectorState<N>& stateIn, mtDifVec& vecOut) const{
//    vecOut = vector_-stateIn.vector_;
//  }
//  void print() const{
//    std::cout << "Vector:" << vector_.transpose() << std::endl;
//  }
//  void setIdentity(){
//    vector_.setZero();
//  }
//  const double& operator[](unsigned int i) const{
//    assert(i<D_);
//    return vector_(i);
//  };
//  double& operator[](unsigned int i){
//    assert(i<D_);
//    return vector_(i);
//  };
//  VectorState<N>& operator=(const Eigen::Matrix<double,D_,1>& vec){
//    vector_ = vec;
//    return *this;
//  };
//  VectorState<N>& operator=(const VectorState<N>& state){
//    vector_ = state.vector_;
//    return *this;
//  };
//  template<unsigned int M>
//  Eigen::Block<Eigen::Matrix<double,D_,1>,M,1> block(unsigned int i){
//    assert(M+i<=N);
//    return vector_.block<M,1>(i,0);
//  }
//  template<unsigned int M>
//  const Eigen::Block<const Eigen::Matrix<double,D_,1>,M,1> block(unsigned int i) const{
//    assert(M+i<=N);
//    return vector_.block<M,1>(i,0);
//  }
//  unsigned int getId(const double& var) const{
//    return IdMap_.at(static_cast<const void*>(&var));
//  };
//  template<int M>
//  unsigned int getId(const Eigen::Block<const Eigen::Matrix<double,D_,1>,M,1> var) const{
//    return IdMap_.at(static_cast<const void*>(&var(0)));
//  };
//  template<int M>
//  unsigned int getId(Eigen::Block<Eigen::Matrix<double,D_,1>,M,1> var) const{
//    return IdMap_.at(static_cast<const void*>(&var(0)));
//  };
//  void createVarLookup(){
//    for(unsigned int i=0;i<D_;i++){
//      IdMap_[static_cast<const void*>(&vector_(i))] = i;
//    }
//  };
//  static VectorState<N> Identity(){
//    VectorState<N> identity;
//    return identity;
//  }
//  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
//    for(unsigned int i=0;i<D_;i++){
//      mtPropertyHandler->doubleRegister_.registerScalar(str + "v" + std::to_string(i), vector_(i));
//    }
//  }
//  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, mtCovMat& cov){
//    for(unsigned int i=0;i<D_;i++){
//      mtPropertyHandler->doubleRegister_.registerScalar(str + "v" + std::to_string(i), cov(i,i));
//    }
//  }
//};
//
//template<typename State1, typename State2>
//class PairState{
// public:
//  static const unsigned int D_ = State1::D_+State2::D_;
//  typedef Eigen::Matrix<double,D_,1> mtDifVec;
//  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
//  PairState(){};
//  PairState(const PairState<State1,State2>& other){
//    state1_ = other.state1_;
//    state2_ = other.state2_;
//  };
//  State1 state1_;
//  State2 state2_;
//  void boxPlus(const mtDifVec& vecIn, PairState<State1,State2>& stateOut) const{
//    state1_.boxPlus(vecIn.template block<State1::D_,1>(0,0),stateOut.state1_);
//    state2_.boxPlus(vecIn.template block<State2::D_,1>(State1::D_,0),stateOut.state2_);
//  }
//  void boxMinus(const PairState<State1,State2>& stateIn, mtDifVec& vecOut) const{
//    typename State1::mtDifVec difVec1;
//    state1_.boxMinus(stateIn.state1_,difVec1);
//    vecOut.template block<State1::D_,1>(0,0) = difVec1;
//    typename State2::mtDifVec difVec2;
//    state2_.boxMinus(stateIn.state2_,difVec2);
//    vecOut.template block<State2::D_,1>(State1::D_,0) = difVec2;
//  }
//  void print() const{
//    state1_.print();
//    state2_.print();
//  }
//  void setIdentity(){
//    state1_.setIdentity();
//    state2_.setIdentity();
//  }
//  const State1& first() const{
//    return state1_;
//  };
//  State1& first(){
//    return state1_;
//  };
//  const State1& second() const{
//    return state2_;
//  };
//  State2& second(){
//    return state2_;
//  };
//  PairState<State1,State2>& operator=(const PairState<State1,State2>& state){
//    state1_ = state.state1_;
//    state2_ = state.state2_;
//    return *this;
//  };
//  static PairState<State1,State2> Identity(){
//    PairState<State1,State2> identity;
//    return identity;
//  }
//  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
//    first().registerToPropertyHandler(mtPropertyHandler,str + "first.");
//    second().registerToPropertyHandler(mtPropertyHandler,str + "second.");
//  }
//  void registerDiagonalMatrixToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str, mtCovMat& cov){
//    first().registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str + "first.",cov);
//    second().registerDiagonalMatrixToPropertyHandler(mtPropertyHandler,str + "second.",cov);
//  }
//};

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
    identity.setIdentity();
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
