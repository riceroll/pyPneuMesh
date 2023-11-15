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

#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/cgal/points_inside_component.h>
#include <igl/point_mesh_squared_distance.h>


class Model {

public:
  Eigen::MatrixXd V0;
  Eigen::MatrixXd V;
  Eigen::MatrixXi E;
  Eigen::VectorXd L0;   // target length of mass spring

  // define Vo, Fo from obj
  Eigen::MatrixXd Vo;
  Eigen::MatrixXi Fo;

  Eigen::MatrixXd Vel;
  Eigen::MatrixXd Force;

  double h;
  Eigen::VectorXd K;
  double damping;
  double gravity;
  double friction;
  double CONTRACTION_SPEED;

  bool hasObstacle;

//  Model(Eigen::VectorXd K, double h, double gravity, double damping, double friction,
//        Eigen::MatrixXd v0, Eigen::MatrixXi e, double CONTRACTION_SPEED);

// define the constructor with objDir
  Model(Eigen::VectorXd K, double h, double gravity, double damping, double friction,
        Eigen::MatrixXd v0, Eigen::MatrixXi e, double CONTRACTION_SPEED, std::string objDir);

  Eigen::VectorXd getL(Eigen::MatrixXd V, Eigen::MatrixXi E);

  Eigen::MatrixXi getE();

  std::pair<Eigen::VectorXd, Eigen::VectorXd> step(Eigen::VectorXd times, Eigen::MatrixXd lengths, int numSteps);

  Eigen::MatrixXd stepForGym(Eigen::VectorXd lengths, int numSteps);

  void handleMeshCollision(const Eigen::MatrixXd &Vo, const Eigen::MatrixXi &Fo, Eigen::MatrixXd &V, Eigen::MatrixXd &Vel, Eigen::MatrixXd &Force, double delta, double mu, double h);

};


#endif //GEODESY_MODEL_H
