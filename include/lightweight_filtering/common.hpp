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
#include "kindr/rotations/RotationEigen.hpp"
#include <type_traits>
#include <tuple>
#include <map>

namespace rot = kindr::rotations::eigen_impl;

typedef rot::RotationQuaternionPD QPD;
typedef rot::RotationMatrixPD MPD;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::VectorXd VXD;
typedef Eigen::MatrixXd MXD;
inline M3D gSM(const V3D& vec){
  return kindr::linear_algebra::getSkewMatrixFromVector(vec);
}

static void enforceSymmetry(MXD& mat){
  mat = 0.5*(mat+mat.transpose()).eval();
}

inline M3D Lmat (const V3D& a) {
  double aNorm = a.norm();
  double factor1 = 1.0/2.0;
  double factor2 = 1.0/6.0;
  // Get sqew matrices
  M3D ak(kindr::linear_algebra::getSkewMatrixFromVector(a));
  M3D ak2(ak*ak);

  // Compute factors
  if(aNorm >= 1e-10){
    factor1 = (1.0 - cos(aNorm))/pow(aNorm,2);
    factor2 = (aNorm-sin(aNorm))/pow(aNorm,3);
  }

  return M3D::Identity()-factor1*ak+factor2*ak2;
}

namespace LWF{
  enum FilteringMode{
    ModeEKF,
    ModeUKF,
    ModeIEKF
  };
}

#endif /* LWF_COMMON_HPP_ */
