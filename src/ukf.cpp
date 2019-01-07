#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 1, 0,
      0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = .5;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // set initialization to false at the start
  is_initialized_ = false;

  // time when tate is true , in us
  time_us_ = 0;

  //state dimension
  n_x_ = 5;

  // Augumented state dimension
  n_aug_ = 7;

  // Spreading Parameter Lambda
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  // weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = (double)lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  {
    weights_(i) = (double)(0.5 / (n_aug_ + lambda_));
  }

  // the current NIS for radar
  NIS_radar_ = 0.0;

  // the current NIS for laser
  NIS_lidar_ = 0.0;

  NIS_writer_radar_.open("nis_radar.txt", ios::out);
  NIS_writer_lidar_.open("nis_lidar.txt", ios::out);
}

UKF::~UKF()
{
  NIS_writer_radar_.close();
  NIS_writer_lidar_.close();
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_)
  {
    if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_)
    {
      x_(0) = meas_package.raw_measurements_(0) > 0.0001 ? meas_package.raw_measurements_(0) : 0.0001;
      x_(1) = meas_package.raw_measurements_(1) > 0.0001 ? meas_package.raw_measurements_(1) : 0.0001;
      x_(2) = 0;
      x_(3) = 0.5;
      x_(4) = 0.5;
    }
    else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_)
    {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double x, y, vx, vy;

      x = fabs(rho) > 0.0001 ? rho * cos(phi) : 0.0001;
      y = fabs(rho) > 0.0001 ? rho * sin(phi) : 0.0001;
      vx = fabs(rho_dot) > 0.0001 ? rho_dot * cos(phi) : 0.0001;
      vy = fabs(rho_dot) > 0.0001 ? rho_dot * sin(phi) : 0.0001;

      x_(0) = x;
      x_(1) = y;
      x_(2) = sqrt(vx * vx + vy * vy);
      x_(3) = 1;
      x_(4) = .1;
    }
    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  /*****************************************************************************
     *  Prediction
     ****************************************************************************/
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
     *  Update
     ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    UpdateRadar(meas_package);
  }
}

/**
 * Normalizes the angle value  of provided vector
 * @param {VectorXd &} res  Vector to be look for
 * @param {int} index       Vector index to be normalized
 */
void UKF::NormalizeAngle(VectorXd &res, int index)
{
  while (res(index) > M_PI)
    res(index) -= 2.0 * M_PI;
  while (res(index) < -M_PI)
    res(index) += 2.0 * M_PI;
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
     * TODO:
     * Complete this function! Use lidar data to update the belief about the object's
     * position. Modify the state vector, x_, and covariance, P_.
     *
     * You'll also need to calculate the lidar NIS.
     */

  // extract measurement
  int n_laser_ = 2;
  VectorXd z = VectorXd(n_laser_);
  z(0) = meas_package.raw_measurements_(0);
  z(1) = meas_package.raw_measurements_(1);

  //create example matrix with sigma points in measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_laser_, 2 * n_aug_ + 1);

  //create example vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_laser_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred += weights_(i) * Zsig.col(i);

  //create example matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_laser_, n_laser_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    NormalizeAngle(z_diff, 1);

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  // Laser noise covariance matrix R_laser
  MatrixXd R_laser_;
  R_laser_ = MatrixXd(n_laser_, n_laser_);
  R_laser_ << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;
  S += R_laser_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_laser_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    NormalizeAngle(z_diff, 1);

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    NormalizeAngle(x_diff, 3);

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  // angle normalization
  NormalizeAngle(z_diff, 1);

  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  //calculate NIS
  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;

  // dump to file
  NIS_writer_lidar_ << NIS_lidar_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
     * TODO:
     * Complete this function! Use radar data to update the belief about the object's
     * position. Modify the state vector, x_, and covariance, P_.
     *
     * You'll also need to calculate the radar NIS.
     */

  // extract measurement
  int n_radar_ = 3;
  VectorXd z = VectorXd(n_radar_);
  z(0) = meas_package.raw_measurements_(0);
  z(1) = meas_package.raw_measurements_(1);
  z(2) = meas_package.raw_measurements_(2);

  // Radar noise covariance matrix
  MatrixXd R_radar_ = MatrixXd(n_radar_, n_radar_);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;

  //create example matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);

  ///transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         //r
    Zsig(1, i) = atan2(p_y, p_x);                                     //phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
  }

  //create example vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_radar_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred += weights_(i) * Zsig.col(i);

  //create example matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_radar_, n_radar_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    NormalizeAngle(z_diff, 1);

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  S += R_radar_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_radar_);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    NormalizeAngle(z_diff, 1);

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    NormalizeAngle(x_diff, 3);

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  // angle normalization
  NormalizeAngle(z_diff, 1);

  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  //calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  NIS_writer_radar_ << NIS_radar_ << std::endl;
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  /***********************************************************
   * Generate Sigma Points
  ***********************************************************/
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);
  AugmentedSigmaPoints(&Xsig_aug);

  /***********************************************************
     * Predict Sigma Points
     ***********************************************************/
  SigmaPointPrediction(Xsig_aug, delta_t);

  /***********************************************************
     * Predict Mean and Covariance
     ***********************************************************/
  PredictMeanAndCovariance(&x_, &P_);
}

/**
 * Populates Augmented Sigma Points
 * @param {MatrixXd *} Xsig_out    Augmented Sigma Points
 */
void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out)
{
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  // augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  double lambda_plus_n_aug_sqrt = sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + lambda_plus_n_aug_sqrt * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - lambda_plus_n_aug_sqrt * L.col(i);
  }

  *Xsig_out = Xsig_aug;
}

/**
 * Predicts Sigma Points
 * @param {MatrixXd *} Xsig_out    Augmented Sigma Points
 * @param {double} delta_t         timestamp difference
 */
void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t)
{
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.0001)
    {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p += nu_a * delta_t;

    yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p += nu_yawdd * delta_t;

    //write predicted sigma points into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}
/**
 * Predicts mean and covariance matrix for UKF
 * @param {VectorXd *} x_pred   predicted mean matrix
 * @param {MatrixXd *} P_pred   predicted covariance matrix
 */
void UKF::PredictMeanAndCovariance(VectorXd *x_pred, MatrixXd *P_pred)
{
  VectorXd x = VectorXd(n_x_);
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predict state mean
  x.fill(0.0);
  x = Xsig_pred_ * weights_;

  //predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    // angle normalization
    NormalizeAngle(x_diff, 3);

    P += weights_(i) * x_diff * x_diff.transpose();
  }
  *x_pred = x;
  *P_pred = P;
}
