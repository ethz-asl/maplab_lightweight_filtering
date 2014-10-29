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
#include "State.hpp"

namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<typename TYPE>
class Register{
 public:
  typedef boost::property_tree::ptree ptree;
  Register(){};
  ~Register(){};
  std::unordered_map<std::string,TYPE*> registerMap_;
  void registerScalar(std::string str, TYPE& var){
    if(registerMap_.count(str)!=0) std::cout << "Property Handler Error: Property with name " << str << " already exists" << std::endl;
    registerMap_[str] = &var;
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
  void buildPropertyTree(ptree& pt){
    for(typename std::unordered_map<std::string,TYPE*>::iterator it=registerMap_.begin(); it != registerMap_.end(); ++it){
      pt.put(it->first, *(it->second));
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
  void registerSubHandler(std::string str,PropertyHandler& subHandler){
    if(subHandlers_.count(str)!=0) std::cout << "Property Handler Error: subHandler with name " << str << " already exists" << std::endl;
    subHandlers_[str] = &subHandler;
  }
  template<unsigned int S, unsigned int V, unsigned int Q>
  void registerStateSVQ(std::string str,StateSVQ<S,V,Q>& state){
    for(unsigned int i=0;i<S;i++){
      doubleRegister_.registerScalar(str + "s" + std::to_string(i), state.s(i));
    }
    for(unsigned int i=0;i<V;i++){
      doubleRegister_.registerVector(str + "v" + std::to_string(i), state.v(i));
    }
    for(unsigned int i=0;i<Q;i++){
      doubleRegister_.registerQuaternion(str + "q" + std::to_string(i), state.q(i));
    };
  }
  void writeToInfo(const std::string &filename){
    ptree pt;
    buildPropertyTree(pt);
    write_info(filename,pt);
  }
  void readFromInfo(const std::string &filename){
  }
};

}

#endif /* PropertyHandler_HPP_ */
