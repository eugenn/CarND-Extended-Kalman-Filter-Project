#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);

    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ <<
             0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ <<
             0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    //laser measurement matrix
    H_laser_ <<
             1, 0, 0, 0,
            0, 1, 0, 0;

    //the initial transition matrix F_
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    //set the acceleration noise components
    noise_ax = 9.0;
    noise_ay = 9.0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {

        //state covariance matrix P
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ <<
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;

        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */

            float rho = measurement_pack.raw_measurements_[0]; // range
            float phi = measurement_pack.raw_measurements_[1]; // bearing
            float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho

            // Coordinates conversion from polar to cartesian
            float x = rho * cos(phi);
            float y = rho * sin(phi);
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);

            ekf_.x_ << x, y, vx, vy;

        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            // set the state with the initial location and zero velocity
            ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
        }

        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    //compute the time elapsed between the current and previous measurements
    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;


    cout << dt << endl;

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    ekf_.F_ <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;

    ekf_.Q_ <<
            (pow(dt, 4) / 4 * noise_ax), 0, (pow(dt, 3) / 2 * noise_ax), 0,
            0, (pow(dt, 4) / 4 * noise_ay), 0, (pow(dt, 3) / 2 * noise_ay),
            (pow(dt, 3) / 2 * noise_ax), 0, pow(dt, 2) * noise_ax, 0,
            0, (pow(dt, 3) / 2 * noise_ay), 0, pow(dt, 2) * noise_ay;


    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
