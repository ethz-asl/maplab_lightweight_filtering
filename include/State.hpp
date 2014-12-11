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

template<typename DERIVED, typename GET, unsigned int D, unsigned int E = D>
class ElementBase{
 public:
  ElementBase(){};
  virtual ~ElementBase(){};
  static const unsigned int D_ = D;
  static const unsigned int E_ = E;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  typedef GET mtGet;
  std::string name_;
  virtual void boxPlus(const mtDifVec& vecIn, DERIVED& stateOut) const = 0;
  virtual void boxMinus(const DERIVED& stateIn, mtDifVec& vecOut) const = 0;
  virtual void print() const = 0;
  virtual void setIdentity() = 0;
  virtual void setRandom(unsigned int& s) = 0;
  virtual void fix() = 0;
  static DERIVED Identity(){
    DERIVED identity;
    identity.setIdentity();
    return identity;
  }
  DERIVED& operator=(const DERIVED& other){
    other.swap(*this);
    return *this;
  }
  virtual mtGet& get(unsigned int i) = 0;
  virtual const mtGet& get(unsigned int i) const = 0;
  virtual void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str) = 0;
  template<int N,int j>
  void registerCovarianceToPropertyHandler(Eigen::Matrix<double,N,N>& cov, PropertyHandler* mpPropertyHandler, const std::string& str){
    assert(j+D_<=N);
    for(unsigned int i=0;i<DERIVED::D_;i++){
      mpPropertyHandler->doubleRegister_.registerScalar(str + name_ + "_" + std::to_string(i), cov(j+i,j+i));
    }
  }
};

template<typename DERIVED>
class AuxiliaryBase: public ElementBase<AuxiliaryBase<DERIVED>,DERIVED,0>{
 public:
  typedef ElementBase<AuxiliaryBase<DERIVED>,DERIVED,0> Base;
  using typename  Base::mtDifVec;
  using typename  Base::mtGet;
  AuxiliaryBase(){}
  AuxiliaryBase(const AuxiliaryBase& other){}
  virtual ~AuxiliaryBase(){}
  virtual void boxPlus(const mtDifVec& vecIn, AuxiliaryBase& stateOut) const{
    static_cast<DERIVED&>(stateOut) = static_cast<const DERIVED&>(*this);
  }
  virtual void boxMinus(const AuxiliaryBase& stateIn, mtDifVec& vecOut) const{}
  virtual void print() const{}
  virtual void setIdentity(){}
  virtual void setRandom(unsigned int& s){}
  virtual void fix(){}
  mtGet& get(unsigned int i){
    return static_cast<DERIVED&>(*this);
  }
  const mtGet& get(unsigned int i) const{
    return static_cast<const DERIVED&>(*this);
  }
  virtual void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str){}
};

class ScalarElement: public ElementBase<ScalarElement,double,1>{
 public:
  double s_;
  ScalarElement(){}
  ScalarElement(const ScalarElement& other){
    s_ = other.s_;
  }
  void boxPlus(const mtDifVec& vecIn, ScalarElement& stateOut) const{
    stateOut.s_ = s_ + vecIn(0);
  }
  void boxMinus(const ScalarElement& stateIn, mtDifVec& vecOut) const{
    vecOut(0) = s_ - stateIn.s_;
  }
  void print() const{
    std::cout << s_ << std::endl;
  }
  void setIdentity(){
    s_ = 0.0;
  }
  void setRandom(unsigned int& s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    s_ = distribution(generator);
    s++;
  }
  void fix(){
  }
  mtGet& get(unsigned int i = 0){
    assert(i==0);
    return s_;
  }
  const mtGet& get(unsigned int i = 0) const{
    assert(i==0);
    return s_;
  }
  void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str){
    mpPropertyHandler->doubleRegister_.registerScalar(str + name_,s_);
  }
};

template<unsigned int N>
class VectorElement: public ElementBase<VectorElement<N>,Eigen::Matrix<double,N,1>,N>{
 public:
  typedef ElementBase<VectorElement<N>,Eigen::Matrix<double,N,1>,N> Base;
  using typename  Base::mtDifVec;
  using typename  Base::mtGet;
  using typename  Base::name_;
  static const unsigned int N_ = N;
  Eigen::Matrix<double,N_,1> v_;
  VectorElement(){}
  VectorElement(const VectorElement<N>& other){
    v_ = other.v_;
  }
  void boxPlus(const mtDifVec& vecIn, VectorElement<N>& stateOut) const{
    stateOut.v_ = v_ + vecIn;
  }
  void boxMinus(const VectorElement<N>& stateIn, mtDifVec& vecOut) const{
    vecOut = v_ - stateIn.v_;
  }
  void print() const{
    std::cout << v_.transpose() << std::endl;
  }
  void setIdentity(){
    v_.setZero();
  }
  void setRandom(unsigned int& s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    for(unsigned int i=0;i<N_;i++){
      v_(i) = distribution(generator);
    }
    s++;
  }
  void fix(){
  }
  mtGet& get(unsigned int i = 0){
    assert(i==0);
    return v_;
  }
  const mtGet& get(unsigned int i = 0) const{
    assert(i==0);
    return v_;
  }
  void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str){
    for(unsigned int i = 0;i<N_;i++){
      mpPropertyHandler->doubleRegister_.registerScalar(str + name_ + "_" + std::to_string(i), v_(i));
    }
  }
};

class QuaternionElement: public ElementBase<QuaternionElement,rot::RotationQuaternionPD,3>{
 public:
  rot::RotationQuaternionPD q_;
  QuaternionElement(){}
  QuaternionElement(const QuaternionElement& other){
    q_ = other.q_;
  }
  void boxPlus(const mtDifVec& vecIn, QuaternionElement& stateOut) const{
    stateOut.q_ = q_.boxPlus(vecIn);
  }
  void boxMinus(const QuaternionElement& stateIn, mtDifVec& vecOut) const{
    vecOut = q_.boxMinus(stateIn.q_);
  }
  void print() const{
    std::cout << q_ << std::endl;
  }
  void setIdentity(){
    q_.setIdentity();
  }
  void setRandom(unsigned int& s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    q_.toImplementation().w() = distribution(generator);
    q_.toImplementation().x() = distribution(generator);
    q_.toImplementation().y() = distribution(generator);
    q_.toImplementation().z() = distribution(generator);
    fix();
    s++;
  }
  void fix(){
    q_.fix();
  }
  mtGet& get(unsigned int i = 0){
    assert(i==0);
    return q_;
  }
  const mtGet& get(unsigned int i = 0) const{
    assert(i==0);
    return q_;
  }
  void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str){
    mpPropertyHandler->doubleRegister_.registerQuaternion(str + name_, q_);
  }
};

class NormalVectorElement: public ElementBase<NormalVectorElement,Eigen::Vector3d,2>{
 public:
  Eigen::Vector3d n_;
  NormalVectorElement(){}
  NormalVectorElement(const NormalVectorElement& other){
    n_ = other.n_;
  }
  void boxPlus(const mtDifVec& vecIn, NormalVectorElement& stateOut) const{
    Eigen::Vector3d m0;
    Eigen::Vector3d m1;
    getTwoNormals(m0,m1);
    rot::RotationQuaternionPD q = q.exponentialMap(vecIn(0)*m0+vecIn(1)*m1);
    stateOut.n_ = q.rotate(n_);
  }
  void boxMinus(const NormalVectorElement& stateIn, mtDifVec& vecOut) const{
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
  void setRandom(unsigned int& s){
    std::default_random_engine generator (s);
    std::normal_distribution<double> distribution (0.0,1.0);
    n_(0) = distribution(generator);
    n_(1) = distribution(generator);
    n_(2) = distribution(generator);
    fix();
    s++;
  }
  void fix(){
    n_.normalize();
  }
  mtGet& get(unsigned int i = 0){
    assert(i==0);
    return n_;
  }
  const mtGet& get(unsigned int i = 0) const{
    assert(i==0);
    return n_;
  }
  void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str){
    mpPropertyHandler->doubleRegister_.registerVector(str + name_, n_);
  }
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
  Eigen::Matrix<double,3,2> getM() const {
    Eigen::Vector3d m0;
    Eigen::Vector3d m1;
    getTwoNormals(m0,m1);
    Eigen::Matrix<double,3,2> M;
    M.col(0) = m0;
    M.col(1) = m1;
    return M;
  }
};

template<typename Element, unsigned int M>
class ArrayElement: public ElementBase<ArrayElement<Element,M>,typename Element::mtGet,M*Element::D_,Element::D_>{
 public:
  typedef ElementBase<ArrayElement<Element,M>,typename Element::mtGet,M*Element::D_,Element::D_> Base;
  using typename Base::mtDifVec;
  using typename Base::mtGet;
  using typename Base::name_;
  static const unsigned int M_ = M;
  Element array_[M_];
  ArrayElement(){
    static_assert((M>0),"size of array must be larger than 0, please use TH_multiple_elements otherwise");
    for(unsigned int i=0; i<M_;i++){
      array_[i].name_ = "";
    }
  }
  ArrayElement(const ArrayElement& other){
    for(unsigned int i=0; i<M_;i++){
      array_[i] = other.array_[i];
    }
  }
  void boxPlus(const mtDifVec& vecIn, ArrayElement& stateOut) const{
    if(Element::D_>0){
      for(unsigned int i=0; i<M_;i++){
        array_[i].boxPlus(vecIn.template block<Element::D_,1>(Element::D_*i,0),stateOut.array_[i]);
      }
    }
  }
  void boxMinus(const ArrayElement& stateIn, mtDifVec& vecOut) const{
    if(Element::D_>0){
      typename Element::mtDifVec difVec;
      for(unsigned int i=0; i<M_;i++){
        array_[i].boxMinus(stateIn.array_[i],difVec);
        vecOut.template block<Element::D_,1>(Element::D_*i,0) = difVec;
      }
    }
  }
  void print() const{
    for(unsigned int i=0; i<M_;i++){
      array_[i].print();
    }
  }
  void setIdentity(){
    for(unsigned int i=0; i<M_;i++){
      array_[i].setIdentity();
    }
  }
  void setRandom(unsigned int& s){
    for(unsigned int i=0; i<M_;i++){
      array_[i].setRandom(s);
    }
  }
  void fix(){
    for(unsigned int i=0; i<M_;i++){
      array_[i].fix();
    }
  }
  mtGet& get(unsigned int i){
    assert(i<M_);
    return array_[i].get();
  }
  const mtGet& get(unsigned int i) const{
    assert(i<M_);
    return array_[i].get();
  }
  void registerElementToPropertyHandler(PropertyHandler* mpPropertyHandler, const std::string& str){
    for(unsigned int i=0; i<M_;i++){
      array_[i].registerElementToPropertyHandler(mpPropertyHandler,str + name_ + "_" + std::to_string(i));
    }
  }
};

template <typename Element>
class TH_convert{
 public:
  typedef std::tuple<Element> t;
};

template <typename Arg, unsigned int N>
class TH_multiple_elements{
 private:
 public:
  typedef decltype(std::tuple_cat(typename TH_convert<Arg>::t(),typename TH_multiple_elements<Arg,N-1>::t())) t;
};
template <typename Arg>
class TH_multiple_elements<Arg,1>{
 public:
  typedef typename TH_convert<Arg>::t t;
};
template <typename Arg>
class TH_multiple_elements<Arg,0>{
 public:
  typedef std::tuple<> t;
};

template <typename Element>
class TH_getDimension{
 public:
  static const unsigned int D_ = Element::D_;
};
template <typename Element>
class TH_getDimension<std::tuple<Element>>{
 public:
  static const unsigned int D_ = Element::D_;
};
template <typename Element, typename... Elements>
class TH_getDimension<std::tuple<Element, Elements...>>{
 public:
  static const unsigned int D_ = Element::D_ + TH_getDimension<std::tuple<Elements...>>::D_;
};

template<typename... Elements>
class State{
 public:
  typedef decltype(std::tuple_cat(typename TH_convert<Elements>::t()...)) t;
  static const unsigned int D_ = TH_getDimension<t>::D_;
  static const unsigned int E_ = std::tuple_size<t>::value;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  t mElements_;
  State(){}
  State(const State<Elements...>& other): mElements_(other.mElements_){}
  void boxPlus(const mtDifVec& vecIn, State<Elements...>& stateOut) const{
    boxPlus_(vecIn,stateOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void boxPlus_(const mtDifVec& vecIn, State<Elements...>& stateOut) const{
    if(std::tuple_element<i,decltype(mElements_)>::type::D_>0){
      std::get<i>(mElements_).boxPlus(vecIn.template block<std::tuple_element<i,decltype(mElements_)>::type::D_,1>(j,0),std::get<i>(stateOut.mElements_));
    } else { // Required for auxiliary states
      Eigen::Matrix<double,std::tuple_element<i,decltype(mElements_)>::type::D_,1> dummyVec;
      std::get<i>(mElements_).boxPlus(dummyVec,std::get<i>(stateOut.mElements_));
    }
    boxPlus_<i+1,j+std::tuple_element<i,decltype(mElements_)>::type::D_>(vecIn,stateOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void boxPlus_(const mtDifVec& vecIn, State<Elements...>& stateOut) const{
    if(std::tuple_element<i,decltype(mElements_)>::type::D_>0){
      std::get<i>(mElements_).boxPlus(vecIn.template block<std::tuple_element<i,decltype(mElements_)>::type::D_,1>(j,0),std::get<i>(stateOut.mElements_));
    } else { // Required for auxiliary states
      Eigen::Matrix<double,std::tuple_element<i,decltype(mElements_)>::type::D_,1> dummyVec;
      std::get<i>(mElements_).boxPlus(dummyVec,std::get<i>(stateOut.mElements_));
    }
  }
  void boxMinus(const State<Elements...>& stateIn, mtDifVec& vecOut) const{
    boxMinus_(stateIn,vecOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void boxMinus_(const State<Elements...>& stateIn, mtDifVec& vecOut) const{
    if(std::tuple_element<i,decltype(mElements_)>::type::D_>0){
      typename std::tuple_element<i,decltype(mElements_)>::type::mtDifVec difVec;
      std::get<i>(mElements_).boxMinus(std::get<i>(stateIn.mElements_),difVec);
      vecOut.template block<std::tuple_element<i,decltype(mElements_)>::type::D_,1>(j,0) = difVec;
    }
    boxMinus_<i+1,j+std::tuple_element<i,decltype(mElements_)>::type::D_>(stateIn,vecOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void boxMinus_(const State<Elements...>& stateIn, mtDifVec& vecOut) const{
    if(std::tuple_element<i,decltype(mElements_)>::type::D_>0){
      typename std::tuple_element<i,decltype(mElements_)>::type::mtDifVec difVec;
      std::get<i>(mElements_).boxMinus(std::get<i>(stateIn.mElements_),difVec);
      vecOut.template block<std::tuple_element<i,decltype(mElements_)>::type::D_,1>(j,0) = difVec;
    }
  }
  void print() const{
    print_();
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void print_() const{
    std::get<i>(mElements_).print();
    print_<i+1>();
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void print_() const{
    std::get<i>(mElements_).print();
  }
  void setIdentity(){
    setIdentity_();
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void setIdentity_(){
    std::get<i>(mElements_).setIdentity();
    setIdentity_<i+1>();
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void setIdentity_(){
    std::get<i>(mElements_).setIdentity();
  }
  void setRandom(unsigned int s){
    setRandom_(s);
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void setRandom_(unsigned int s){
    std::get<i>(mElements_).setRandom(s);
    setRandom_<i+1>(s);
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void setRandom_(unsigned int s){
    std::get<i>(mElements_).setRandom(s);
  }
  void fix(){
    fix_();
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void fix_(){
    std::get<i>(mElements_).fix();
    fix_<i+1>();
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void fix_(){
    std::get<i>(mElements_).fix();
  }
  void registerElementsToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    registerElementsToPropertyHandler_(mtPropertyHandler,str);
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void registerElementsToPropertyHandler_(PropertyHandler* mtPropertyHandler, const std::string& str){
    std::get<i>(mElements_).registerElementToPropertyHandler(mtPropertyHandler,str);
    registerElementsToPropertyHandler_<i+1>(mtPropertyHandler,str);
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void registerElementsToPropertyHandler_(PropertyHandler* mtPropertyHandler, const std::string& str){
    std::get<i>(mElements_).registerElementToPropertyHandler(mtPropertyHandler,str);
  }
  void registerCovarianceToPropertyHandler(Eigen::Matrix<double,D_,D_>& cov, PropertyHandler* mpPropertyHandler, const std::string& str){
    registerCovarianceToPropertyHandler_(cov,mpPropertyHandler,str);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void registerCovarianceToPropertyHandler_(Eigen::Matrix<double,D_,D_>& cov, PropertyHandler* mpPropertyHandler, const std::string& str){
    std::get<i>(mElements_).template registerCovarianceToPropertyHandler<D_,j>(cov,mpPropertyHandler,str);
    registerCovarianceToPropertyHandler_<i+1,j+std::tuple_element<i,decltype(mElements_)>::type::D_>(cov,mpPropertyHandler,str);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void registerCovarianceToPropertyHandler_(Eigen::Matrix<double,D_,D_>& cov, PropertyHandler* mpPropertyHandler, const std::string& str){
    std::get<i>(mElements_).template registerCovarianceToPropertyHandler<D_,j>(cov,mpPropertyHandler,str);
  }
  void createDefaultNames(const std::string& str = "def"){
    createDefaultNames_(str);
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void createDefaultNames_(const std::string& str){
    std::get<i>(mElements_).name_ = str + "_" + std::to_string(i);
    createDefaultNames_<i+1>(str);
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void createDefaultNames_(const std::string& str){
    std::get<i>(mElements_).name_ = str + "_" + std::to_string(i);
  }
  template<unsigned int i>
  auto get(unsigned int j = 0) -> decltype (std::get<i>(mElements_).get(j))& {
    return std::get<i>(mElements_).get(j);
  };
  template<unsigned int i>
  const auto get(unsigned int j = 0) const -> decltype (std::get<i>(mElements_).get(j))& {
    return std::get<i>(mElements_).get(j);
  };
  template<unsigned int i,unsigned int D=0,typename std::enable_if<(i==0)>::type* = nullptr>
  static constexpr unsigned int getId(unsigned int j = 0){
    return D+j*std::tuple_element<i,decltype(mElements_)>::type::E_;
  };
  template<unsigned int i,unsigned int D=0,typename std::enable_if<(i>0 & i<E_)>::type* = nullptr>
  static constexpr unsigned int getId(unsigned int j = 0){
    return getId<i-1,D+std::tuple_element<i-1,decltype(mElements_)>::type::D_>(0)+j*std::tuple_element<i,decltype(mElements_)>::type::E_;
  };
  template<unsigned int i>
  std::string& getName(){
    return std::get<i>(mElements_).name_;
  };
  template<unsigned int i>
  const std::string& getName() const{
    return std::get<i>(mElements_).name_;
  };
  static State Identity(){
    State identity;
    identity.setIdentity();
    return identity;
  }
};

template <typename... Args>
class TH_convert<State<Args...>>: public State<Args...>{
};
template <typename Arg, unsigned int N>
class TH_convert<TH_multiple_elements<Arg,N>>: public TH_multiple_elements<Arg,N>{
};



template<typename... States>
class ComposedState{ // TODO: test completely
 public:
  static const unsigned int D_ = TH_getDimension<decltype(std::tuple_cat(typename TH_convert<States>::t()...))>::D_;
  typedef Eigen::Matrix<double,D_,1> mtDifVec;
  typedef Eigen::Matrix<double,D_,D_> mtCovMat;
  std::tuple<States...> mStates_;
  static const unsigned int E_ = std::tuple_size<decltype(mStates_)>::value;
  ComposedState(){}
  ComposedState(const ComposedState<States...>& other): mStates_(other.mStates_){}
  void boxPlus(const mtDifVec& vecIn, ComposedState<States...>& stateOut) const{
    boxPlus_(vecIn,stateOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void boxPlus_(const mtDifVec& vecIn, ComposedState<States...>& stateOut) const{
    if(std::tuple_element<i,decltype(mStates_)>::type::D_>0){
      std::get<i>(mStates_).boxPlus(vecIn.template block<std::tuple_element<i,decltype(mStates_)>::type::D_,1>(j,0),std::get<i>(stateOut.mStates_));
    } else { // Required for auxiliary states
      Eigen::Matrix<double,std::tuple_element<i,decltype(mStates_)>::type::D_,1> dummyVec;
      std::get<i>(mStates_).boxPlus(dummyVec,std::get<i>(stateOut.mStates_));
    }
    boxPlus_<i+1,j+std::tuple_element<i,decltype(mStates_)>::type::D_>(vecIn,stateOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void boxPlus_(const mtDifVec& vecIn, ComposedState<States...>& stateOut) const{
    if(std::tuple_element<i,decltype(mStates_)>::type::D_>0){
      std::get<i>(mStates_).boxPlus(vecIn.template block<std::tuple_element<i,decltype(mStates_)>::type::D_,1>(j,0),std::get<i>(stateOut.mStates_));
    } else { // Required for auxiliary states
      Eigen::Matrix<double,std::tuple_element<i,decltype(mStates_)>::type::D_,1> dummyVec;
      std::get<i>(mStates_).boxPlus(dummyVec,std::get<i>(stateOut.mStates_));
    }
  }
  void boxMinus(const ComposedState<States...>& stateIn, mtDifVec& vecOut) const{
    boxMinus_(stateIn,vecOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void boxMinus_(const ComposedState<States...>& stateIn, mtDifVec& vecOut) const{
    if(std::tuple_element<i,decltype(mStates_)>::type::D_>0){
      typename std::tuple_element<i,decltype(mStates_)>::type::mtDifVec difVec;
      std::get<i>(mStates_).boxMinus(std::get<i>(stateIn.mStates_),difVec);
      vecOut.template block<std::tuple_element<i,decltype(mStates_)>::type::D_,1>(j,0) = difVec;
    }
    boxMinus_<i+1,j+std::tuple_element<i,decltype(mStates_)>::type::D_>(stateIn,vecOut);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void boxMinus_(const ComposedState<States...>& stateIn, mtDifVec& vecOut) const{
    if(std::tuple_element<i,decltype(mStates_)>::type::D_>0){
      typename std::tuple_element<i,decltype(mStates_)>::type::mtDifVec difVec;
      std::get<i>(mStates_).boxMinus(std::get<i>(stateIn.mStates_),difVec);
      vecOut.template block<std::tuple_element<i,decltype(mStates_)>::type::D_,1>(j,0) = difVec;
    }
  }
  void print() const{
    print_();
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void print_() const{
    std::get<i>(mStates_).print();
    print_<i+1>();
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void print_() const{
    std::get<i>(mStates_).print();
  }
  void setIdentity(){
    setIdentity_();
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void setIdentity_(){
    std::get<i>(mStates_).setIdentity();
    setIdentity_<i+1>();
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void setIdentity_(){
    std::get<i>(mStates_).setIdentity();
  }
  void setRandom(unsigned int s){
    setRandom_(s);
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void setRandom_(unsigned int s){
    std::get<i>(mStates_).setRandom(s);
    setRandom_<i+1>(s);
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void setRandom_(unsigned int s){
    std::get<i>(mStates_).setRandom(s);
  }
  void fix(){
    fix_();
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void fix_(){
    std::get<i>(mStates_).fix();
    fix_<i+1>();
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void fix_(){
    std::get<i>(mStates_).fix();
  }
  void registerElementsToPropertyHandler(PropertyHandler* mtPropertyHandler, const std::string& str){
    registerElementsToPropertyHandler_(mtPropertyHandler,str);
  }
  template<unsigned int i=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void registerElementsToPropertyHandler_(PropertyHandler* mtPropertyHandler, const std::string& str){
    std::get<i>(mStates_).registerElementToPropertyHandler(mtPropertyHandler,str);
    registerElementsToPropertyHandler_<i+1>(mtPropertyHandler,str);
  }
  template<unsigned int i=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void registerElementsToPropertyHandler_(PropertyHandler* mtPropertyHandler, const std::string& str){
    std::get<i>(mStates_).registerElementToPropertyHandler(mtPropertyHandler,str);
  }
  void registerCovarianceToPropertyHandler(Eigen::Matrix<double,D_,D_>& cov, PropertyHandler* mpPropertyHandler, const std::string& str){
    registerCovarianceToPropertyHandler_(cov,mpPropertyHandler,str);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i<E_-1)>::type* = nullptr>
  void registerCovarianceToPropertyHandler_(Eigen::Matrix<double,D_,D_>& cov, PropertyHandler* mpPropertyHandler, const std::string& str){
    std::get<i>(mStates_).template registerCovarianceToPropertyHandler<D_,j>(cov,mpPropertyHandler,str);
    registerCovarianceToPropertyHandler_<i+1,j+std::tuple_element<i,decltype(mStates_)>::type::D_>(cov,mpPropertyHandler,str);
  }
  template<unsigned int i=0,unsigned int j=0,typename std::enable_if<(i==E_-1)>::type* = nullptr>
  void registerCovarianceToPropertyHandler_(Eigen::Matrix<double,D_,D_>& cov, PropertyHandler* mpPropertyHandler, const std::string& str){
    std::get<i>(mStates_).template registerCovarianceToPropertyHandler<D_,j>(cov,mpPropertyHandler,str);
  }
  template<unsigned int i>
  auto getState(unsigned int j = 0) -> decltype (std::get<i>(mStates_))& {
    return std::get<i>(mStates_);
  };
  template<unsigned int i>
  const auto getState(unsigned int j = 0) const -> decltype (std::get<i>(mStates_))& {
    return std::get<i>(mStates_);
  };
  static ComposedState Identity(){
    ComposedState identity;
    identity.setIdentity();
    return identity;
  }
};

static Eigen::Matrix3d Lmat (Eigen::Vector3d a) {
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

template<unsigned int S, unsigned int V, unsigned int Q>
class StateSVQ: public State<ArrayElement<ScalarElement,S>,ArrayElement<VectorElement<3>,V>,ArrayElement<QuaternionElement,Q>>{
 public:
  static const unsigned int S_ = S;
  static const unsigned int V_ = V;
  static const unsigned int Q_ = Q;
  typedef State<ArrayElement<ScalarElement,S>,ArrayElement<VectorElement<3>,V>,ArrayElement<QuaternionElement,Q>> Base;
  StateSVQ(){}
  StateSVQ(const StateSVQ& other): Base(other){}
  const double& s(unsigned int i) const{
    return this->template get<0>(i);
  };
  double& s(unsigned int i) {
    return this->template get<0>(i);
  };
  const Eigen::Matrix<double,3,1>& v(unsigned int i) const{
    return this->template get<1>(i);
  };
  Eigen::Matrix<double,3,1>& v(unsigned int i) {
    return this->template get<1>(i);
  };
  const rot::RotationQuaternionPD& q(unsigned int i) const{
    return this->template get<2>(i);
  };
  rot::RotationQuaternionPD& q(unsigned int i) {
    return this->template get<2>(i);
  };
};

}

#endif /* STATE_HPP_ */
