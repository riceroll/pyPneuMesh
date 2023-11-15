//
// Created by 莱洛 on 2/15/21.
//

#ifndef GEODESY_MODEL_H
#define GEODESY_MODEL_H

#include <iostream>
#include <set>
#include <vector>
#include <tuple>
#include <cmath>
#include <fstream>
#include <regex>

#include "pybind11/eigen.h"


class Model {

public:
  Eigen::MatrixXd V0;
  Eigen::MatrixXd V;
  Eigen::MatrixXi E;
  Eigen::VectorXd L0;   // target length of mass spring

  Eigen::MatrixXd Vel;
  Eigen::MatrixXd Force;

  double h;
  Eigen::VectorXd K;
  double damping;
  double gravity;
  double friction;
  double CONTRACTION_SPEED;

  Model(Eigen::VectorXd K, double h, double gravity, double damping, double friction,
        Eigen::MatrixXd v0, Eigen::MatrixXi e, double CONTRACTION_SPEED);

  Eigen::VectorXd getL(Eigen::MatrixXd V, Eigen::MatrixXi E);

  Eigen::MatrixXi getE();

  std::pair<Eigen::VectorXd, Eigen::VectorXd> step(Eigen::VectorXd times, Eigen::MatrixXd lengths, int numSteps);

  Eigen::MatrixXd stepForGym(Eigen::VectorXd lengths, int numSteps);

};


#endif //GEODESY_MODEL_H
