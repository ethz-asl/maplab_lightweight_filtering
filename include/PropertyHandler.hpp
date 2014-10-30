/*
 * PropertyHandler.hpp
 *
 *  Created on: Oct 27, 2014
 *      Author: Bloeschm
 */

#ifndef PropertyHandler_HPP_
#define PropertyHandler_HPP_

#include <Eigen/Dense>
#include "kindr/rotations/RotationEigen.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <unordered_map>
#include <unordered_set>
#include "State.hpp"

namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<typename TYPE>
class Register{
 public:
  typedef boost::property_tree::ptree ptree;
  Register(){};
  ~Register(){};
  std::map<TYPE*,std::string> registerMap_;
  std::unordered_set<TYPE*> zeros_;
  void registerZero(TYPE& var){
    if(zeros_.count(&var)!=0) std::cout << "Property Handler Error: Zero variable already registered." << std::endl;
    zeros_.insert(&var);
  }
  void registerScalar(std::string str, TYPE& var){
    if(registerMap_.count(&var)!=0) std::cout << "Property Handler Error: Variable already registered to " << str << "." << std::endl;
    registerMap_[&var] = str;
  }
  void registerVector(std::string str, Eigen::Matrix<TYPE,3,1>& var){
    registerScalar(str + "x",var(0));
    registerScalar(str + "y",var(1));
    registerScalar(str + "z",var(2));
  }
  void registerQuaternion(std::string str, rot::RotationQuaternion<TYPE, kindr::rotations::RotationUsage::PASSIVE>& var){
    registerScalar(str + "w",var.toImplementation().w());
    registerScalar(str + "x",var.toImplementation().x());
    registerScalar(str + "y",var.toImplementation().y());
    registerScalar(str + "z",var.toImplementation().z());
  }
  template<int N, int M>
  void registerMatrix(std::string str, Eigen::Matrix<TYPE,N,M>& var){
    for(unsigned int i=0;i<N;i++){
      for(unsigned int j=0;j<M;j++){
        registerScalar(str + std::to_string(i) + "_" + std::to_string(j),var(i,j));
      }
    }
  }
  template<int N>
  void registerDiagonalMatrix(std::string str, Eigen::Matrix<TYPE,N,N>& var){
    for(unsigned int i=0;i<N;i++){
      registerScalar(str + std::to_string(i),var(i,i));
    }
    for(unsigned int i=0;i<N;i++){
      for(unsigned int j=0;j<N;j++){
        if(i!=j) registerZero(var(i,j));
      }
    }
  }
  template<int N>
  void registerScaledUnitMatrix(std::string str, Eigen::Matrix<TYPE,N,N>& var){
    for(unsigned int i=0;i<N;i++){
      registerScalar(str,var(i,i));
    }
    for(unsigned int i=0;i<N;i++){
      for(unsigned int j=0;j<N;j++){
        if(i!=j) registerZero(var(i,j));
      }
    }
  }
  void buildPropertyTree(ptree& pt){
    for(typename std::map<TYPE*,std::string>::iterator it=registerMap_.begin(); it != registerMap_.end(); ++it){
      pt.put(it->second, *(it->first));
    }
  }
  void readPropertyTree(ptree& pt){
    for(typename std::map<TYPE*,std::string>::iterator it=registerMap_.begin(); it != registerMap_.end(); ++it){
      *(it->first) = pt.get<TYPE>(it->second);
    }
    for(typename std::unordered_set<TYPE*>::iterator it=zeros_.begin(); it != zeros_.end(); ++it){
      **it = 0;
    }
  }
};

class PropertyHandler{
 public:
  typedef boost::property_tree::ptree ptree;
  PropertyHandler(){};
  ~PropertyHandler(){};
  Register<bool> boolRegister_;
  Register<int> intRegister_;
  Register<double> doubleRegister_;
  std::unordered_map<std::string,PropertyHandler*> subHandlers_;
  void buildPropertyTree(ptree& pt){
    boolRegister_.buildPropertyTree(pt);
    intRegister_.buildPropertyTree(pt);
    doubleRegister_.buildPropertyTree(pt);
    for(typename std::unordered_map<std::string,PropertyHandler*>::iterator it=subHandlers_.begin(); it != subHandlers_.end(); ++it){
      ptree ptsub;
      it->second->buildPropertyTree(ptsub);
      pt.add_child(it->first,ptsub);
    }
  }
  void readPropertyTree(ptree& pt){
    boolRegister_.readPropertyTree(pt);
    intRegister_.readPropertyTree(pt);
    doubleRegister_.readPropertyTree(pt);
    for(typename std::unordered_map<std::string,PropertyHandler*>::iterator it=subHandlers_.begin(); it != subHandlers_.end(); ++it){ // TODO
      ptree ptsub;
      ptsub = pt.get_child(it->first);
      it->second->readPropertyTree(ptsub);
    }
  }
  void registerSubHandler(std::string str,PropertyHandler& subHandler){
    if(subHandlers_.count(str)!=0) std::cout << "Property Handler Error: subHandler with name " << str << " already exists" << std::endl;
    subHandlers_[str] = &subHandler;
  }
  template<unsigned int S, unsigned int V, unsigned int Q>
  void registerStateSVQ(std::string str,StateSVQ<S,V,Q>& state){
    for(unsigned int i=0;i<S;i++){
      doubleRegister_.registerScalar(state.sName(i), state.s(i));
    }
    for(unsigned int i=0;i<V;i++){
      doubleRegister_.registerVector(state.vName(i), state.v(i));
    }
    for(unsigned int i=0;i<Q;i++){
      doubleRegister_.registerQuaternion(state.qName(i), state.q(i));
    };
  }
  template<unsigned int N>
  void registerVectorState(std::string str,VectorState<N>& state){
    for(unsigned int i=0;i<N;i++){
      doubleRegister_.registerScalar(state.sName(i), "v" + std::to_string(i));
    }
  }
  void writeToInfo(const std::string &filename){
    ptree pt;
    buildPropertyTree(pt);
    write_info(filename,pt);
  }
  void readFromInfo(const std::string &filename){
    ptree pt;
    read_info(filename,pt);
    readPropertyTree(pt);
  }
};

}

#endif /* PropertyHandler_HPP_ */
