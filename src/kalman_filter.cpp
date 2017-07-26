#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;

    KF(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    double z_ro = z[0];
    double z_theta = z[1];
    double z_ro_dot = z[2];

    if (z[0] < 0.001) {
        return;
    }

    double px = x_[0];
    double py = x_[1];
    double vx = x_[2];
    double vy = x_[3];

    double pxy = pow(px, 2) + pow(py, 2);
    double pxy_sqrt = pow(pxy, 0.5);

    VectorXd z_pred = VectorXd(3);
    z_pred << pxy_sqrt, atan2(py, px), (px * vx + py * vy) / pxy_sqrt;

    VectorXd y = z - z_pred;

    // Adjust angle representation to keep between -pi and pi.
    double pi = 3.14159265358;

    while (y[1] > pi) {
        y[1] -= 2 * pi;
    }
    while (y[1] < -pi) {
        y[1] += 2 * pi;
    }

    double c = (vx * py - vy * px) / pow(pxy, 1.5);

    H_ <<
       px / pxy_sqrt, py / pxy_sqrt, 0, 0,
            -py / pxy, px / pxy, 0, 0,
            py * c, -px * c, px / pxy_sqrt, py / pxy_sqrt;

    KF(y);
}


// Universal update Kalman Filter step. Equations from the lectures
void KalmanFilter::KF(const VectorXd &y) {
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K = P_ * Ht * Si;

    // New state
    x_ = x_ + (K * y);

    long x_size = x_.size();

    MatrixXd I = MatrixXd::Identity(x_size, x_size);

    P_ = (I - K * H_) * P_;
}