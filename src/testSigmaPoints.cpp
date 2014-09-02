#include "SigmaPoints.hpp"
#include "gtest/gtest.h"
#include <assert.h>

// The fixture for testing class ScalarElement.
class SigmaPointTest : public ::testing::Test {
 protected:
  SigmaPointTest() {
    assert(V_>=Q_-1);
    unsigned int L = mtState::D_;
    double lambda = 1e-6*(L)-L;
    double gamma = sqrt(lambda + L);
    double wm = 1/(2*(L+lambda));
    double wc = wm;
    double wc0 = lambda/(L+lambda)+(1-1e-6+2);
    sigmaPoints_.resize(2*L+1,2*L+1,0);
    sigmaPoints_.setParameter(wm,wc,wc0,gamma);
    mean_.s(0) = 4.5;
    for(int i=1;i<S_;i++){
      mean_.s(i) = mean_.s(i-1) + i*i*46.2;
    }
    mean_.v(0) << 2.1, -0.2, -1.9;
    for(int i=1;i<V_;i++){
      mean_.v(i) = mean_.v(i-1) + Eigen::Vector3d(0.3,10.9,2.3);
    }
    mean_.q(0) = rot::RotationQuaternionPD(4.0/sqrt(30.0),3.0/sqrt(30.0),1.0/sqrt(30.0),2.0/sqrt(30.0));
    for(int i=1;i<Q_;i++){
      mean_.q(i) = mean_.q(i-1).boxPlus(mean_.v(i-1));
    }
    // Easy way to obtain a pseudo random positive definite matrix
    P_ = mtState::D_*mtCovMat::Identity();
    double randValue;
    for(int i=0;i<mtState::D_;i++){
      for(int j=i;j<mtState::D_;j++){
        randValue = (cos((double)(123456*(i+j+1)))+1.0)/2.0;
        P_(i,j) += randValue;
        P_(j,i) += randValue;
      }
    }
  }
  virtual ~SigmaPointTest() {
  }
  static const unsigned int S_ = 4;
  static const unsigned int V_ = 3;
  static const unsigned int Q_ = 2;
  const unsigned int N_ = 2;
  const unsigned int L_ = 6;
  const unsigned int offset_ = 2;
  typedef LightWeightUKF::State<S_,V_,Q_> mtState;
  typedef mtState::DiffVec mtDiffVec;
  typedef mtState::CovMat mtCovMat;
  LightWeightUKF::SigmaPoints<mtState> sigmaPoints_;
  mtState mean_;
  mtCovMat P_;

  typedef LightWeightUKF::State<0,1,0> mtStateVector;
};

// Test constructors
TEST_F(SigmaPointTest, constructors) {
  LightWeightUKF::SigmaPoints<mtState> sigmaPoints;
  ASSERT_TRUE(sigmaPoints.sigmaPoints_.empty());
  ASSERT_EQ(sigmaPoints.N_,0);
  ASSERT_EQ(sigmaPoints.L_,0);
  ASSERT_EQ(sigmaPoints.offset_,0);

  LightWeightUKF::SigmaPoints<mtState> sigmaPoints2(N_,L_,offset_);
  ASSERT_EQ(sigmaPoints2.sigmaPoints_.size(),sigmaPoints2.N_);
  ASSERT_EQ(sigmaPoints2.N_,N_);
  ASSERT_EQ(sigmaPoints2.L_,L_);
  ASSERT_EQ(sigmaPoints2.offset_,offset_);
}

// Test resize
TEST_F(SigmaPointTest, resize) {
  sigmaPoints_.resize(N_,L_,offset_);
  ASSERT_EQ(sigmaPoints_.sigmaPoints_.size(),sigmaPoints_.N_);
  ASSERT_EQ(sigmaPoints_.N_,N_);
  ASSERT_EQ(sigmaPoints_.L_,L_);
  ASSERT_EQ(sigmaPoints_.offset_,offset_);
}

// Test setParameter
TEST_F(SigmaPointTest, setParameter) {
  LightWeightUKF::SigmaPoints<mtState> sigmaPoints;
  unsigned int L = mtState::D_;
  double lambda = 1e-6*(L)-L;
  double gamma = sqrt(lambda + L);
  double wm = 1/(2*(L+lambda));
  double wc = wm;
  double wc0 = lambda/(L+lambda)+(1-1e-6+2);
  sigmaPoints.setParameter(wm,wc,wc0,gamma);
  ASSERT_EQ(sigmaPoints_.wm_,wm);
  ASSERT_EQ(sigmaPoints_.wc_,wc);
  ASSERT_EQ(sigmaPoints_.wc0_,wc0);
  ASSERT_EQ(sigmaPoints_.gamma_,gamma);
}

// Test computeFromGaussian, getMean, getCovariance, computeFromZeroMeanGaussian
TEST_F(SigmaPointTest, computeFromGaussianPlusPlus) {
  // computeFromGaussian
  sigmaPoints_.computeFromGaussian(mean_,P_);

  // Check mean is same
  mtState mean = sigmaPoints_.getMean();
  mtDiffVec vec;
  mean.boxminus(mean_,vec);
  ASSERT_NEAR(vec.norm(),0.0,1e-8);

  // Check covariance is same
  mtCovMat P = sigmaPoints_.getCovarianceMatrix(mean);
  for(int i=0;i<mtState::D_;i++){
    for(int j=0;j<mtState::D_;j++){
      ASSERT_NEAR(P_(i,j),P(i,j),1e-8);
    }
  }

  // computeFromZeroMeanGaussian
  sigmaPoints_.computeFromZeroMeanGaussian(P_);

  // Check mean is same
  mean = sigmaPoints_.getMean();
  ASSERT_TRUE(mean.isNearIdentity(1e-8));

  // Check covariance is same
  P = sigmaPoints_.getCovarianceMatrix(mean);
  for(int i=0;i<mtState::D_;i++){
    for(int j=0;j<mtState::D_;j++){
      ASSERT_NEAR(P_(i,j),P(i,j),1e-8);
    }
  }
}

// Test getMean, getCovariance2
TEST_F(SigmaPointTest, getCovariance2) {
  // computeFromGaussian
  sigmaPoints_.computeFromGaussian(mean_,P_);

  unsigned int L = mtState::D_*2+1;
  LightWeightUKF::SigmaPoints<mtStateVector> sigmaPointsVector(L,L,0);
  sigmaPointsVector.setParameter(sigmaPoints_.wm_,sigmaPoints_.wc_,sigmaPoints_.wc0_,sigmaPoints_.gamma_);

  // Apply simple linear transformation
  for(int i=0;i<L;i++){
    sigmaPointsVector(i).v(0) = sigmaPoints_(i).v(0)*2.45+Eigen::Vector3d::Ones()*sigmaPoints_(i).s(0)*0.51;
  }

  Eigen::Matrix<double,mtStateVector::D_,mtState::D_> M = sigmaPointsVector.getCovarianceMatrix(sigmaPoints_);
  Eigen::Matrix<double,mtStateVector::D_,mtState::D_> H; // Jacobian of linear transformation
  H.setZero();
  H.block(0,S_,3,3) = Eigen::Matrix3d::Identity()*2.45;
  H.block(0,0,3,1) = Eigen::Vector3d::Ones()*0.51;
  Eigen::Matrix<double,mtStateVector::D_,mtState::D_> Mref = H*P_;
  for(int i=0;i<mtStateVector::D_;i++){
    for(int j=0;j<mtState::D_;j++){
      ASSERT_NEAR(Mref(i,j),M(i,j),1e-8);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
