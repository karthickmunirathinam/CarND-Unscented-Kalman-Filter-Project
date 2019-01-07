#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include <fstream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF
{
public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;
  // NIS of Laser and Lidar
  double NIS_lidar_;
  double NIS_radar_;

private:
  ///* NIS Value writer (used to store NIS values of Radar on csv file)
  std::fstream NIS_writer_radar_;

  ///* NIS Value writer (used to store NIS values of Laser on csv file)
  std::fstream NIS_writer_lidar_;

  /**
   * Normalizes the angle value (provided index) of provided vector
   * @param {VectorXd &} res  Vector to be look for
   * @param {int} index       Vector index to be normalized
   */
  inline void NormalizeAngle(VectorXd &res, int index);

  /**
         * Populates Augmented Sigma Points
         * @param {MatrixXd *} Xsig_out    Augmented Sigma Points
         */
  void AugmentedSigmaPoints(MatrixXd *Xsig_out);

  /**
         * Predicts Sigma Points
         * @param {MatrixXd *} Xsig_out    Augmented Sigma Points
         * @param {double} delta_t         timestamp difference
         */
  void SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t);

  /**
         * Predicts mean and covariance matrix for UKF
         * @param {VectorXd *} x_pred   predicted mean matrix
         * @param {MatrixXd *} P_pred   predicted covariance matrix
         */
  void PredictMeanAndCovariance(VectorXd *x_pred, MatrixXd *P_pred);
};

#endif // UKF_H