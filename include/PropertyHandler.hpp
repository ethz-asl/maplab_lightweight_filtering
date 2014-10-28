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
#include <list>
namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<typename TYPE>
class Register{
 public:
  typedef boost::property_tree::ptree ptree;
  Register(){};
  ~Register(){};
  typedef std::pair<std::string,TYPE*> mtEntry;
  std::list<mtEntry> register_; // TODO make unique
  void registerScalar(std::string str, TYPE& var){
    register_.push_back(mtEntry(str,&var));
  }
  void registerVector(std::string str, Eigen::Matrix<TYPE,3,0>& var){
    register_.push_back(mtEntry(str + "x",&var(0)));
    register_.push_back(mtEntry(str + "y",&var(1)));
    register_.push_back(mtEntry(str + "z",&var(2)));
  }
  void registerQuaternion(std::string str, rot::RotationQuaternion<TYPE, kindr::rotations::RotationUsage::PASSIVE>& var){
    register_.push_back(mtEntry(str + "w",&var.w()));
    register_.push_back(mtEntry(str + "x",&var.x()));
    register_.push_back(mtEntry(str + "y",&var.y()));
    register_.push_back(mtEntry(str + "z",&var.z()));
  }
  void buildPropertyTree(ptree& pt){
    for(typename std::list<mtEntry>::iterator it=register_.begin(); it != register_.end(); ++it){
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
  std::list<std::pair<std::string,PropertyHandler*>> subHandlers_;
  void buildPropertyTree(ptree& pt){
    boolRegister_.buildPropertyTree(pt);
    intRegister_.buildPropertyTree(pt);
    doubleRegister_.buildPropertyTree(pt);
    for(typename std::list<std::pair<std::string,PropertyHandler*>>::iterator it=subHandlers_.begin(); it != subHandlers_.end(); ++it){
      ptree ptsub;
      it->second->buildPropertyTree(ptsub);
      ptsub.put("pat", 3.14f);
      pt.add_child(it->first,ptsub);
    }
  }
  void registerSubHandler(std::string str,PropertyHandler& subHandler){
    subHandlers_.push_back(std::pair<std::string,PropertyHandler*>(str,&subHandler));
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
