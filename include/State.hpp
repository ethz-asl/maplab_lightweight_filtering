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
#include <type_traits>
#include <tuple>

namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<typename DERIVED, unsigned int D, unsigned int E = 1>
class StateBase{
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
  virtual void setRandom(unsigned int s) = 0;
  virtual void fix() = 0;
  virtual void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str) = 0;
  virtual void createDefaultNames(const std::string& str = "") = 0;
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
  void setRandom(unsigned int s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    s_ = distribution(generator);
  }
  void fix(){
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    mtPropertyHandler->doubleRegister_.registerScalar(str + name_, s_);
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
  static constexpr unsigned int getId(){
    return 0;
  };
  void createDefaultNames(const std::string& str = ""){
    name_ = str;
  };
};

template<unsigned int N>
class VectorState: public StateBase<VectorState<N>,N>{
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
  void setRandom(unsigned int s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    for(unsigned int i=0;i<N_;i++){
      v_(i) = distribution(generator);
    }
  }
  void fix(){
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    for(unsigned int i = 0;i<N_;i++){
      mtPropertyHandler->doubleRegister_.registerScalar(str + name_ + "_" + std::to_string(i), v_(i));
    }
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
  static constexpr unsigned int getId(){
    return 0;
  };
  void createDefaultNames(const std::string& str = ""){
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
  void setRandom(unsigned int s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    q_.toImplementation().w() = distribution(generator);
    q_.toImplementation().x() = distribution(generator);
    q_.toImplementation().y() = distribution(generator);
    q_.toImplementation().z() = distribution(generator);
    fix();
  }
  void fix(){
    q_.fix();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    mtPropertyHandler->doubleRegister_.registerQuaternion(str + name_, q_);
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
  static constexpr unsigned int getId(){
    return 0;
  };
  void createDefaultNames(const std::string& str = ""){
    name_ = str;
  };
};

class NormalVectorState: public StateBase<NormalVectorState,2>{
 public:
  typedef StateBase<NormalVectorState,2> Base;
  using Base::D_;
  using Base::E_;
  using typename  Base::mtDifVec;
  using typename Base::mtCovMat;
  using Base::name_;
  NormalVectorState(){}
  NormalVectorState(const NormalVectorState& other){
    n_ = other.n_;
  }
  Eigen::Vector3d n_;
  void boxPlus(const mtDifVec& vecIn, NormalVectorState& stateOut) const{
    Eigen::Vector3d m0;
    Eigen::Vector3d m1;
    getTwoNormals(m0,m1);
    rot::RotationQuaternionPD q = q.exponentialMap(vecIn(0)*m0+vecIn(1)*m1);
    stateOut.n_ = q.rotate(n_);
  }
  void boxMinus(const NormalVectorState& stateIn, mtDifVec& vecOut) const{
    rot::RotationQuaternionPD q;
    q.setFromVectors(stateIn.n_,n_);
    Eigen::Vector3d vec = -q.logarithmicMap(); // Minus required (active/passiv messes things up probably)
    Eigen::Vector3d m0;
    Eigen::Vector3d m1;
    stateIn.getTwoNormals(m0,m1);
    vecOut(0) = m0.dot(vec);
    vecOut(1) = m1.dot(vec);
  }
  void print() const{
    std::cout << n_.transpose() << std::endl;
  }
  void setIdentity(){
    n_ = Eigen::Vector3d(1.0,0.0,0.0);
  }
  void setRandom(unsigned int s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    n_(0) = distribution(generator);
    n_(1) = distribution(generator);
    n_(2) = distribution(generator);
    fix();
  }
  void fix(){
    n_.normalize();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    mtPropertyHandler->doubleRegister_.registerVector(str + name_, n_);
  }
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  NormalVectorState& getState(){
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const NormalVectorState& getState() const{
    return *this;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  Eigen::Vector3d& getValue(){
    return n_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  const Eigen::Vector3d& getValue() const{
    return n_;
  };
  template<unsigned int i, typename std::enable_if<(i==0)>::type* = nullptr>
  static constexpr unsigned int getId(){
    return 0;
  };
  void createDefaultNames(const std::string& str = ""){
    name_ = str;
  };
  void getTwoNormals(Eigen::Vector3d& m0,Eigen::Vector3d& m1) const {
    Eigen::Vector3d vec = Eigen::Vector3d(1.0,0.0,0.0);
    double min = n_(0);
    if(n_(1)<min){
      Eigen::Vector3d(0.0,1.0,0.0);
      min = n_(1);
    }
    if(n_(2)<min){
      Eigen::Vector3d(0.0,0.0,1.0);
    }
    m0 = vec.cross(n_);
    m0.normalize();
    m1 = m0.cross(n_);
    m1.normalize();
  }
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
  void setRandom(unsigned int s){
    for(unsigned int i=0;i<N;i++){
      array_[i].setRandom(s+i);
    }
  }
  void fix(){
    for(unsigned int i=0;i<N;i++){
      array_[i].fix();
    }
  }
  void createDefaultNames(const std::string& str = ""){
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
  static constexpr unsigned int getId(){
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
  void setRandom(unsigned int s){
    state_.setRandom(s);
    subComposedState_.setRandom(s+1);
  }
  void fix(){
    state_.fix();
    subComposedState_.fix();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    state_.registerToPropertyHandler(mtPropertyHandler,str);
    subComposedState_.registerToPropertyHandler(mtPropertyHandler,str);
  }
  void createDefaultNames(const std::string& str = ""){
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
  static constexpr unsigned int getId(){
    return State::template getId<i>();
  };
  template<unsigned int i, typename std::enable_if<(i>=State::E_ & i<E_)>::type* = nullptr>
  static constexpr unsigned int getId(){
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
  void setRandom(unsigned int s){
    state_.setRandom(s);
  }
  void fix(){
    state_.fix();
  }
  void registerToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    state_.registerToPropertyHandler(mtPropertyHandler,str);
  }
  void createDefaultNames(const std::string& str = ""){
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
  static constexpr unsigned int getId(){
    return State::template getId<i>();
  };
};

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
