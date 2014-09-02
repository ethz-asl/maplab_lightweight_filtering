#include "UKF.hpp"
#include "EKF.hpp"
#include "gtest/gtest.h"
#include <assert.h>

class SimpleUKF:public LightWeightUKF::UKF<LightWeightUKF::State<2,0,0>,LightWeightUKF::State<1,0,0>,LightWeightUKF::State<1,0,0>,LightWeightUKF::State<1,0,0>,2,1>{
 public:
  mtState evalPrediction(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtProcessNoise pNoise, const double dt) const{
    mtState x;
    x.s(0) = mpState->s(0) + mpState->s(1)*dt + pNoise(0)*sqrt(dt);
    x.s(1) = mpState->s(1) + mpPredictionMeas->s(0)*dt + pNoise(1)*sqrt(dt);
    return x;
  }
  mtInnovation evalInnovation(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtUpdateMeas* mpUpdateMeas, const mtProcessNoise pNoise,const mtUpdateNoise uNoise, const double dt) const{
    mtInnovation y;
    y.s(0) = mpState->s(0)-mpUpdateMeas->s(0)+uNoise(0);
    return y;
  }
};

class SimpleUKFAugmented:public LightWeightUKF::UKF<LightWeightUKF::State<3,0,0>,LightWeightUKF::State<2,0,0>,LightWeightUKF::State<0,0,0>,LightWeightUKF::State<2,0,0>,3,2>{
 public:
  mtState evalPrediction(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtProcessNoise pNoise, const double dt) const{
    mtState x;
    x.s(0) = mpState->s(0) + mpState->s(1)*dt + pNoise(0)*sqrt(dt);
    x.s(1) = mpState->s(1) + mpState->s(2)*dt;// + pNoise(1)*sqrt(dt);
    x.s(2) = pNoise(2)*sqrt(dt);
    return x;
  }
  mtInnovation evalInnovation(const mtState* mpState, const mtPredictionMeas* mpPredictionMeas, const mtUpdateMeas* mpUpdateMeas, const mtProcessNoise pNoise,const mtUpdateNoise uNoise, const double dt) const{
    mtInnovation y;
    y.s(0) = mpState->s(0)-mpUpdateMeas->s(0)+uNoise(0);
    y.s(1) = mpState->s(2)-mpUpdateMeas->s(1)+uNoise(1)/sqrt(dt);
    return y;
  }
};

// The fixture for testing class ScalarElement.
class UKFTest : public ::testing::Test {
 protected:
  UKFTest() {
    mpPredictionMeas_ = new SimpleUKF::mtPredictionMeas();
    mpUpdateMeas_ = new SimpleUKF::mtUpdateMeas();
    mpPredictionMeasAugmented_ = new SimpleUKFAugmented::mtPredictionMeas();
    mpUpdateMeasAugmented_ = new SimpleUKFAugmented::mtUpdateMeas();
    tData_(0,0) = 0;
    tData_(0,1) = -tAcc_*tN_*dt_*0.5;
    tData_(0,2) = tAcc_;
    tData_(0,3) = 0;
    tData_(0,4) = tAcc_;
    for(unsigned int i=1;i<tN_;i++){
      tData_(i,0) = tData_(i-1,0)+tData_(i-1,1)*dt_; // Position
      tData_(i,1) = tData_(i-1,1)+tData_(i-1,2)*dt_; // Velocity
      tData_(i,2) = tAcc_; // Acceleration
      tData_(i,3) = tData_(i,0); // Position Measurement
      tData_(i,4) = tData_(i,2); // Acceleration Measurement
    }
    refFx_ << 1.0, dt_, 0.0, 1.0;
    refFn_ << sqrt(dt_), 0.0, 0.0, sqrt(dt_);
    refFu_ << 0.0, dt_;
    refGx_ << 1.0, 0.0;
    refGn_ << 1.0;
  }
  virtual ~UKFTest() {
    delete mpPredictionMeas_;
    delete mpUpdateMeas_;
  }
  SimpleUKF simpleUKF_;
  SimpleUKF::mtPredictionMeas* mpPredictionMeas_;
  SimpleUKF::mtUpdateMeas* mpUpdateMeas_;
  SimpleUKFAugmented simpleUKFAugmented_;
  SimpleUKFAugmented::mtPredictionMeas* mpPredictionMeasAugmented_;
  SimpleUKFAugmented::mtUpdateMeas* mpUpdateMeasAugmented_;
  const double dt_ = 0.1;

  // Trajectory
  static const unsigned int tN_ = 100;
  Eigen::Matrix<double,tN_,5> tData_;
  const double tAcc_ = 1.0;

  // Variables for reference KF
  Eigen::Matrix<double,2,1> refX_;
  Eigen::Matrix<double,2,2> refP_;
  Eigen::Matrix<double,2,2> refFx_;
  Eigen::Matrix<double,2,2> refFn_;
  Eigen::Matrix<double,2,1> refFu_;
  Eigen::Matrix<double,1,2> refGx_;
  Eigen::Matrix<double,1,1> refGn_;
  Eigen::Matrix<double,1,1> refY_;
  Eigen::Matrix<double,2,1> refK_;
};

// Test constructors
TEST_F(UKFTest, constructors) {
  SimpleUKF simpleUKF;

  // test size of sigmapoints
  ASSERT_EQ(simpleUKF.stateSigmaPoints1_.N_,2*SimpleUKF::D_+1);
  ASSERT_EQ(simpleUKF.stateSigmaPoints2_.N_,2*(SimpleUKF::D_+SimpleUKF::pD_)+1);
  ASSERT_EQ(simpleUKF.innSigmaPoints3_.N_,2*(SimpleUKF::D_+SimpleUKF::pD_+SimpleUKF::uD_)+1);
  ASSERT_EQ(simpleUKF.processNoiseSP_.N_,2*SimpleUKF::pD_+1);
  ASSERT_EQ(simpleUKF.updateNoiseSP_.N_,2*SimpleUKF::uD_+1);

  // Test state_ is identity
  ASSERT_TRUE(simpleUKF.state_.isNearIdentity(1e-8));

  // Test covariance matrices initialized to identity
  Eigen::Matrix<double,simpleUKF.D_,simpleUKF.D_> testP = simpleUKF.stateP_ - Eigen::Matrix<double,simpleUKF.D_,simpleUKF.D_>::Identity();
  ASSERT_NEAR(testP.norm(),0.0,1e-10);
  Eigen::Matrix<double,simpleUKF.pD_,simpleUKF.pD_> testPreP = simpleUKF.prenoiP_ - Eigen::Matrix<double,simpleUKF.pD_,simpleUKF.pD_>::Identity();
  ASSERT_NEAR(testPreP.norm(),0.0,1e-10);
  Eigen::Matrix<double,simpleUKF.uD_,simpleUKF.uD_> testUpdP = simpleUKF.updnoiP_ - Eigen::Matrix<double,simpleUKF.uD_,simpleUKF.uD_>::Identity();
  ASSERT_NEAR(testUpdP.norm(),0.0,1e-10);
}

// Test reset
TEST_F(UKFTest, reset) {
  double var1 = 0.01;
  double var2 = 0.02;
  double var3 = 0.03;
  simpleUKF_.initState_.s(0) = tData_(0,0);
  simpleUKF_.initState_.s(1) = tData_(0,1);
  simpleUKF_.initStateP_ = Eigen::Matrix2d::Identity()*var1;
  simpleUKF_.prenoiP_ = Eigen::Matrix2d::Identity()*var2;
  simpleUKF_.updnoiP_(0,0) = var3;
  simpleUKF_.reset();

  // Test state_ is identity
  SimpleUKF::mtState::DiffVec vec;
  simpleUKF_.state_.boxminus(simpleUKF_.initState_,vec);
  ASSERT_NEAR(vec.norm(),0.0,1e-10);

  // Test covariance matrices initialized to identity
  Eigen::Matrix<double,simpleUKF_.D_,simpleUKF_.D_> testP = simpleUKF_.stateP_ - Eigen::Matrix<double,simpleUKF_.D_,simpleUKF_.D_>::Identity()*var1;
  ASSERT_NEAR(testP.norm(),0.0,1e-10);
}

// Test setUKFParameter
TEST_F(UKFTest, setUKFParameter) {
  double alpha = 1e-4;
  double beta = 2.1;
  double kappa = 0.03;
  simpleUKF_.setUKFParameter(alpha,beta,kappa);

  // Compute parameters
  unsigned int L = simpleUKF_.D_+simpleUKF_.pD_+simpleUKF_.uD_;
  double lambda = alpha*alpha*(L+kappa)-L;
  double gamma = sqrt(lambda + L);
  double wm = 1/(2*(L+lambda));
  double wc = wm;
  double wc0 = lambda/(L+lambda)+(1-alpha*alpha+beta);

  // Test parameters are properly set
  ASSERT_NEAR(simpleUKF_.innSigmaPoints3_.wm_,wm,1e-10);
  ASSERT_NEAR(simpleUKF_.innSigmaPoints3_.wc_,wc,1e-10);
  ASSERT_NEAR(simpleUKF_.innSigmaPoints3_.wc0_,wc0,1e-10);
  ASSERT_NEAR(simpleUKF_.innSigmaPoints3_.gamma_,gamma,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints2_.wm_,wm,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints2_.wc_,wc,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints2_.wc0_,wc0,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints2_.gamma_,gamma,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints1_.wm_,wm,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints1_.wc_,wc,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints1_.wc0_,wc0,1e-10);
  ASSERT_NEAR(simpleUKF_.stateSigmaPoints1_.gamma_,gamma,1e-10);
  ASSERT_NEAR(simpleUKF_.processNoiseSP_.wm_,wm,1e-10);
  ASSERT_NEAR(simpleUKF_.processNoiseSP_.wc_,wc,1e-10);
  ASSERT_NEAR(simpleUKF_.processNoiseSP_.wc0_,wc0,1e-10);
  ASSERT_NEAR(simpleUKF_.processNoiseSP_.gamma_,gamma,1e-10);
  ASSERT_NEAR(simpleUKF_.updateNoiseSP_.wm_,wm,1e-10);
  ASSERT_NEAR(simpleUKF_.updateNoiseSP_.wc_,wc,1e-10);
  ASSERT_NEAR(simpleUKF_.updateNoiseSP_.wc0_,wc0,1e-10);
  ASSERT_NEAR(simpleUKF_.updateNoiseSP_.gamma_,gamma,1e-10);
}

// Test predict (only forward integration)
TEST_F(UKFTest, predict) {
  double var = 0.01;
  simpleUKF_.initState_.s(0) = tData_(0,0);
  simpleUKF_.initState_.s(1) = tData_(0,1);
  simpleUKF_.initStateP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.prenoiP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.updnoiP_(0,0) = var;
  simpleUKF_.reset();
  simpleUKF_.computeProcessAndUpdateNoiseSP();
  refX_(0) = tData_(0,0);
  refX_(1) = tData_(0,1);
  refP_ = simpleUKF_.stateP_;
  for(unsigned int i=1;i<tN_;i++){
    // Get Measurements
    mpPredictionMeas_->s(0) = tData_(i-1,4);

    // Compute reference with KF
    refX_ = refFx_*refX_+refFu_*mpPredictionMeas_->s(0);
    refP_ = refFx_*refP_*refFx_.transpose()+refFn_*simpleUKF_.prenoiP_*refFn_.transpose();

    // Compare
    simpleUKF_.predict(mpPredictionMeas_,dt_);
    ASSERT_NEAR(simpleUKF_.state_.s(0),refX_(0),1e-8);
    ASSERT_NEAR(simpleUKF_.state_.s(1),refX_(1),1e-8);
    ASSERT_NEAR((refP_-simpleUKF_.stateP_).norm(),0.0,1e-8);
  }
}

// Test predictAndUpdate and compare with real data
TEST_F(UKFTest, predictAndUpdate) {
  double var = 0.01;
  simpleUKF_.initState_.s(0) = tData_(0,0);
  simpleUKF_.initState_.s(1) = tData_(0,1);
  simpleUKF_.initStateP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.prenoiP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.updnoiP_(0,0) = var;
  simpleUKF_.reset();
  simpleUKF_.computeProcessAndUpdateNoiseSP();
  refX_(0) = tData_(0,0);
  refX_(1) = tData_(0,1);
  refP_ = simpleUKF_.stateP_;
  for(unsigned int i=1;i<tN_;i++){
    // Get Measurements
    mpPredictionMeas_->s(0) = tData_(i-1,4);
    mpUpdateMeas_->s(0) = tData_(i,3);

    // Compute reference with KF
    refX_ = refFx_*refX_+refFu_*mpPredictionMeas_->s(0);
    refP_ = refFx_*refP_*refFx_.transpose()+refFn_*simpleUKF_.prenoiP_*refFn_.transpose();
    refY_ = Eigen::Matrix<double,1,1>::Identity()*mpUpdateMeas_->s(0)-refGx_*refX_;
    refK_ = refP_*refGx_.transpose()*(refGx_*refP_*refGx_.transpose()+refGn_*simpleUKF_.updnoiP_*refGn_.transpose()).inverse();
    refX_ = refX_+refK_*refY_;
    refP_ = (Eigen::Matrix<double,2,2>::Identity()-refK_*refGx_)*refP_;

    // Compare
    simpleUKF_.predictAndUpdate(mpPredictionMeas_,mpUpdateMeas_,dt_);
    ASSERT_NEAR(simpleUKF_.state_.s(0),refX_(0),1e-8);
    ASSERT_NEAR(simpleUKF_.state_.s(1),refX_(1),1e-8);
    ASSERT_NEAR((refP_-simpleUKF_.stateP_).norm(),0.0,1e-8);
    ASSERT_NEAR(simpleUKF_.state_.s(0),tData_(i,0),1e-8);
    ASSERT_NEAR(simpleUKF_.state_.s(1),tData_(i,1),1e-8);
  }
}

// Test predictAndUpdate and compare with real data (with noise)
TEST_F(UKFTest, predictAndUpdateWithNoise) {
  double var = 0.01;
  simpleUKF_.initState_.s(0) = tData_(0,0);
  simpleUKF_.initState_.s(1) = tData_(0,1);
  simpleUKF_.initStateP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.prenoiP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.updnoiP_(0,0) = var;
  simpleUKF_.reset();
  simpleUKF_.computeProcessAndUpdateNoiseSP();
  refX_(0) = tData_(0,0);
  refX_(1) = tData_(0,1);
  refP_ = simpleUKF_.stateP_;
  for(unsigned int i=1;i<tN_;i++){
    // Get Measurements
    mpPredictionMeas_->s(0) = tData_(i-1,4)+cos(123456.0*(double)i)*var/sqrt(dt_);
    mpUpdateMeas_->s(0) = tData_(i,3)+cos(345678.0*(double)i)*var;

    // Compute reference with KF
    refX_ = refFx_*refX_+refFu_*mpPredictionMeas_->s(0);
    refP_ = refFx_*refP_*refFx_.transpose()+refFn_*simpleUKF_.prenoiP_*refFn_.transpose();
    refY_ = Eigen::Matrix<double,1,1>::Identity()*mpUpdateMeas_->s(0)-refGx_*refX_;
    refK_ = refP_*refGx_.transpose()*(refGx_*refP_*refGx_.transpose()+refGn_*simpleUKF_.updnoiP_*refGn_.transpose()).inverse();
    refX_ = refX_+refK_*refY_;
    refP_ = (Eigen::Matrix<double,2,2>::Identity()-refK_*refGx_)*refP_;

    // Compare
    simpleUKF_.predictAndUpdate(mpPredictionMeas_,mpUpdateMeas_,dt_);
    ASSERT_NEAR(simpleUKF_.state_.s(0),refX_(0),1e-8);
    ASSERT_NEAR(simpleUKF_.state_.s(1),refX_(1),1e-8);
    ASSERT_NEAR((refP_-simpleUKF_.stateP_).norm(),0.0,1e-8);
  }
}

// Test predictAndUpdate and compare with real data (with noise and outliers)
TEST_F(UKFTest, predictAndUpdateWithNoiseAndOutliers) {
  double var = 0.01;
  simpleUKF_.initState_.s(0) = tData_(0,0);
  simpleUKF_.initState_.s(1) = tData_(0,1);
  simpleUKF_.initStateP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.prenoiP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.updnoiP_(0,0) = var;
  simpleUKF_.reset();
  simpleUKF_.computeProcessAndUpdateNoiseSP();
  refX_(0) = tData_(0,0);
  refX_(1) = tData_(0,1);
  simpleUKF_.outlierDetection_.push_back(SimpleUKF::mtOutlierDetection(0,0,6.64));
  refP_ = simpleUKF_.stateP_;
  for(unsigned int i=1;i<tN_;i++){
    // Get Measurements
    mpPredictionMeas_->s(0) = tData_(i-1,4)+cos(123456.0*(double)i)*var/sqrt(dt_);
    mpUpdateMeas_->s(0) = tData_(i,3)+cos(345678.0*(double)i)*var+(i==tN_/2)*100;

    // Compute reference with KF
    refX_ = refFx_*refX_+refFu_*mpPredictionMeas_->s(0);
    refP_ = refFx_*refP_*refFx_.transpose()+refFn_*simpleUKF_.prenoiP_*refFn_.transpose();
    if(i!=tN_/2){
      refY_ = Eigen::Matrix<double,1,1>::Identity()*mpUpdateMeas_->s(0)-refGx_*refX_;
      refK_ = refP_*refGx_.transpose()*(refGx_*refP_*refGx_.transpose()+refGn_*simpleUKF_.updnoiP_*refGn_.transpose()).inverse();
      refX_ = refX_+refK_*refY_;
      refP_ = (Eigen::Matrix<double,2,2>::Identity()-refK_*refGx_)*refP_;
    }

    // Compare
    simpleUKF_.predictAndUpdate(mpPredictionMeas_,mpUpdateMeas_,dt_);
    ASSERT_NEAR(simpleUKF_.state_.s(0),refX_(0),1e-8);
    ASSERT_NEAR(simpleUKF_.state_.s(1),refX_(1),1e-8);
    ASSERT_NEAR((refP_-simpleUKF_.stateP_).norm(),0.0,1e-8);

    // Test outlier Flag
    if(i!=tN_/2){
      ASSERT_FALSE(simpleUKF_.outlierDetection_[0].outlier_);
    } else {
      ASSERT_TRUE(simpleUKF_.outlierDetection_[0].outlier_);
    }
  }
}

// Test predictAndUpdate and compare with real data (with noise and outliers), with alternative update equations
TEST_F(UKFTest, predictAndUpdateWithNoiseAndOutliersAlternative) {
  double var = 0.01;
  simpleUKF_.initState_.s(0) = tData_(0,0);
  simpleUKF_.initState_.s(1) = tData_(0,1);
  simpleUKF_.initStateP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.prenoiP_ = Eigen::Matrix2d::Identity()*var;
  simpleUKF_.updnoiP_(0,0) = var;
  simpleUKF_.reset();
  simpleUKF_.computeProcessAndUpdateNoiseSP();
  simpleUKF_.useAlternativeUpdate_ = true; // Use alternative update equations
  refX_(0) = tData_(0,0);
  refX_(1) = tData_(0,1);
  simpleUKF_.outlierDetection_.push_back(SimpleUKF::mtOutlierDetection(0,0,6.64));
  refP_ = simpleUKF_.stateP_;
  for(unsigned int i=1;i<tN_;i++){
    // Get Measurements
    mpPredictionMeas_->s(0) = tData_(i-1,4)+cos(123456.0*(double)i)*var/sqrt(dt_);
    mpUpdateMeas_->s(0) = tData_(i,3)+cos(345678.0*(double)i)*var+(i==tN_/2)*100;

    // Compute reference with KF
    refX_ = refFx_*refX_+refFu_*mpPredictionMeas_->s(0);
    refP_ = refFx_*refP_*refFx_.transpose()+refFn_*simpleUKF_.prenoiP_*refFn_.transpose();
    if(i!=tN_/2){
      refY_ = Eigen::Matrix<double,1,1>::Identity()*mpUpdateMeas_->s(0)-refGx_*refX_;
      refK_ = refP_*refGx_.transpose()*(refGx_*refP_*refGx_.transpose()+refGn_*simpleUKF_.updnoiP_*refGn_.transpose()).inverse();
      refX_ = refX_+refK_*refY_;
      refP_ = (Eigen::Matrix<double,2,2>::Identity()-refK_*refGx_)*refP_;
    }

    // Compare
    simpleUKF_.predictAndUpdate(mpPredictionMeas_,mpUpdateMeas_,dt_);
    ASSERT_NEAR(simpleUKF_.state_.s(0),refX_(0),1e-8);
    ASSERT_NEAR(simpleUKF_.state_.s(1),refX_(1),1e-8);
    ASSERT_NEAR((refP_-simpleUKF_.stateP_).norm(),0.0,1e-8);

    // Test outlier Flag
    if(i!=tN_/2){
      ASSERT_FALSE(simpleUKF_.outlierDetection_[0].outlier_);
    } else {
      ASSERT_TRUE(simpleUKF_.outlierDetection_[0].outlier_);
    }
  }
}

// Test predictAndUpdate and compare with real data (with noise) for augmented filter
TEST_F(UKFTest, predictAndUpdateWithNoiseAugmented) {
  double var = 0.01;
  simpleUKFAugmented_.initState_.s(0) = tData_(0,0);
  simpleUKFAugmented_.initState_.s(1) = tData_(0,1);
  simpleUKFAugmented_.initState_.s(2) = tData_(0,4)+cos(123456.0)*var/sqrt(dt_);
  simpleUKFAugmented_.initStateP_ = Eigen::Matrix3d::Identity()*var;
  simpleUKFAugmented_.initStateP_(2,2) = var/dt_;
  simpleUKFAugmented_.prenoiP_(0,0) = var;
  simpleUKFAugmented_.prenoiP_(1,1) = var;
  simpleUKFAugmented_.prenoiP_(2,2) = var;
  simpleUKFAugmented_.updnoiP_(0,0) = var;
  simpleUKFAugmented_.updnoiP_(1,1) = var;
  simpleUKFAugmented_.reset();
  simpleUKFAugmented_.computeProcessAndUpdateNoiseSP();
  simpleUKFAugmented_.useAlternativeUpdate_ = true; // Use alternative update equations
  simpleUKFAugmented_.infinitePredictionNoise_[2] = true;
  refX_(0) = tData_(0,0);
  refX_(1) = tData_(0,1);
  refP_ = simpleUKFAugmented_.stateP_.block<2,2>(0,0);
  for(unsigned int i=1;i<tN_;i++){
    // Get Measurements
    double aLast = tData_(i-1,4)+cos(123456.0*(double)i)*var/sqrt(dt_);
    double aNext = tData_(i,4)+cos(123456.0*(double)(i+1))*var/sqrt(dt_);
    mpUpdateMeasAugmented_->s(0) = tData_(i,3)+cos(345678.0*(double)i)*var;
    mpUpdateMeasAugmented_->s(1) = aNext;

    // Compute reference with KF
    refX_ = refFx_*refX_+refFu_*aLast;
    refP_ = refFx_*refP_*refFx_.transpose()+refFn_*simpleUKFAugmented_.prenoiP_.block<2,2>(0,0)*refFn_.transpose();
    refY_ = Eigen::Matrix<double,1,1>::Identity()*mpUpdateMeasAugmented_->s(0)-refGx_*refX_;
    refK_ = refP_*refGx_.transpose()*(refGx_*refP_*refGx_.transpose()+refGn_*simpleUKFAugmented_.updnoiP_.block<1,1>(0,0)*refGn_.transpose()).inverse();
    refX_ = refX_+refK_*refY_;
    refP_ = (Eigen::Matrix<double,2,2>::Identity()-refK_*refGx_)*refP_;

    // Compare
    simpleUKFAugmented_.predictAndUpdate(mpPredictionMeasAugmented_,mpUpdateMeasAugmented_,dt_);
    ASSERT_NEAR(simpleUKFAugmented_.state_.s(0),refX_(0),1e-8);
    ASSERT_NEAR(simpleUKFAugmented_.state_.s(1),refX_(1),1e-8);
    ASSERT_NEAR((refP_-simpleUKFAugmented_.stateP_.block<2,2>(0,0)).norm(),0.0,1e-8);
  }
}

// Test LMat
TEST_F(UKFTest, LMat) {
  double d = 0.00001;
  LightWeightUKF::State<0,0,1> att;
  LightWeightUKF::State<0,1,0> vec;
  vec.v(0) = Eigen::Vector3d(0.4,-0.2,1.7);
  Eigen::Matrix3d J;
  LightWeightUKF::State<0,0,1> attDisturbed;
  LightWeightUKF::State<0,1,0> vecDisturbed;
  Eigen::Matrix3d I;
  I.setIdentity();
  Eigen::Vector3d dif;
  I = d*I;
  att.q(0) = att.q(0).exponentialMap(vec.v(0));
  for(unsigned int i=0;i<3;i++){
    vec.boxplus(I.col(i),vecDisturbed);
    attDisturbed.q(0) = attDisturbed.q(0).exponentialMap(vecDisturbed.v(0));
    attDisturbed.boxminus(att,dif);
    J.col(i) = dif*1/d;
  }
  std::cout << J << std::endl;
  std::cout << LightWeightEKF::Lmat(vec.v(0)) << std::endl;
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
