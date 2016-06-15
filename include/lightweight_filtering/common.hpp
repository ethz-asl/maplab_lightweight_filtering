/*
 * Common.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_COMMON_HPP_
#define LWF_COMMON_HPP_

#include <Eigen/Dense>
#include <iostream>
#include "kindr/Core"
#include "lightweight_filtering/PropertyHandler.hpp"
#include <type_traits>
#include <tuple>

namespace rot = kindr;

typedef rot::RotationQuaternionPD QPD;
typedef rot::RotationMatrixPD MPD;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
static M3D gSM(const V3D& vec){
  return kindr::getSkewMatrixFromVector(vec);
}

template<int nRow, int nCol, bool isDynamic = false>
class LWFMatrix;

template<int nRow, int nCol>
class LWFMatrix<nRow,nCol,true>: public Eigen::MatrixXd{
 public:
  LWFMatrix():Eigen::MatrixXd(nRow,nCol){}
  typedef Eigen::MatrixXd Base;
  template<typename OtherDerived>
  LWFMatrix(const Eigen::MatrixBase<OtherDerived>& other): Eigen::MatrixXd(other){}
  template<typename OtherDerived>
  LWFMatrix & operator= (const Eigen::MatrixBase <OtherDerived>& other){
    this->Base::operator=(other);
    return *this;
  }
};

template<int nRow, int nCol>
class LWFMatrix<nRow,nCol,false>: public Eigen::Matrix<double,nRow,nCol>{
 public:
  LWFMatrix(){}
  typedef Eigen::Matrix<double,nRow,nCol> Base;
  template<typename OtherDerived>
  LWFMatrix(const Eigen::MatrixBase<OtherDerived>& other): Eigen::Matrix<double,nRow,nCol>(other){}
  template<typename OtherDerived>
  LWFMatrix & operator= (const Eigen::MatrixBase <OtherDerived>& other){
    this->Base::operator=(other);
    return *this;
  }
};

template<int nRow, int nCol, bool isDynamic = false>
static void enforceSymmetry(LWFMatrix<nRow,nCol,true>& mat){
  mat = 0.5*(mat+mat.transpose()).eval();
}

static M3D Lmat (V3D a) {
  double aNorm = a.norm();
  double factor1 = 0;
  double factor2 = 0;
  M3D ak;
  M3D ak2;
  M3D G_k;

  // Get sqew matrices
  ak = kindr::getSkewMatrixFromVector(a);
  ak2 = ak*ak;

  // Compute factors
  if(aNorm >= 1e-10){
    factor1 = (1 - cos(aNorm))/pow(aNorm,2);
    factor2 = (aNorm-sin(aNorm))/pow(aNorm,3);
  } else {
    factor1 = 1/2;
    factor2 = 1/6;
  }

  return M3D::Identity()-factor1*ak+factor2*ak2;
}

#endif /* LWF_COMMON_HPP_ */
