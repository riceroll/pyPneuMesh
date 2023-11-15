#include "Model.h"
#include "pybind11/pybind11.h"

using namespace std;
using namespace Eigen;
namespace py = pybind11;

Model::Model(VectorXd K, double h, double gravity, double damping, double friction,
             MatrixXd v0, MatrixXi e, double CONTRACTION_SPEED)
   {

  this->V0 = v0;
  this->E = e;

  Vel.resize(V0.rows(), 3);
  Vel.setZero();
  Force.resize(V0.rows(), 3);

  this->h = h;
  this->K = K;
  this->gravity = gravity;
  this->damping = damping;
  this->friction = friction;
  this->CONTRACTION_SPEED = CONTRACTION_SPEED;

  this->L0 = getL(V0, E);
}

VectorXd Model::getL(MatrixXd V, MatrixXi E) {
  auto vec = V(E(all, 0), all) - V(E(all, 1), all);
  auto L = vec.rowwise().norm();
  return L;
}

MatrixXi Model::getE() {
  return this->E;
}

std::pair<VectorXd, VectorXd> Model::step(VectorXd times, MatrixXd lengths, int numSteps) {
  MatrixXd V = V0;
  Vel.setZero();

//  Vel.setOnes();
//  Vel(all, 2) *= 0;
//  Vel(all, 1) *= 0;
//  Vel(all, 0) *= 100;

  VectorXd LTarget = lengths.row(0);   // target length of penumatic actuation

  VectorXd Fs((numSteps + 1) * E.rows() * 1);

  VectorXd Vs((numSteps + 1) * V.rows() * 3);
  for (int iRow = 0; iRow < V.rows(); iRow++) {
    for (int iCol=0; iCol < V.cols(); iCol++) {
      Vs((0 * V.rows() + iRow) * 3 + iCol) = V(iRow, iCol);
    }
  }

  for (int iStep=0; iStep < numSteps; iStep++) {
    Force.setZero();

    double time = h * iStep;

    // set LTarget
    for (int iTime = 0; iTime < times.size(); iTime++) {
      if (time > times[iTime]) {
        LTarget = lengths.row(iTime);
      }
    }

    // set L0
    for (int iE = 0; iE < E.rows(); iE++) {
      if (L0[iE] != LTarget[iE]) {
        double diff = LTarget[iE] - L0[iE];
        double sign = diff / abs(diff);
        double dL0 = sign * CONTRACTION_SPEED * h;
        if (abs(dL0) > abs(diff)) {   // L0 goes over LTarget
          L0[iE] = LTarget[iE];
        } else {
          L0[iE] += dL0;
        }
      }
    }

    // add edge force
    VectorXd FEdge;
    FEdge.resize(E.size());
    auto L = getL(V, E);
    auto Vec01 = V(E(all, 1), all) - V(E(all, 0), all); // vectors from V[e[0]] to V[e[1]]

    for (int iE = 0; iE < E.rows(); iE++) {
      FEdge[iE] = (L[iE] - L0[iE]) * K[iE];   // contraction force is positive

      Force.row(E(iE, 0)) += Vec01.row(iE) / L[iE] * FEdge[iE];
      Force.row(E(iE, 1)) -= Vec01.row(iE) / L[iE] * FEdge[iE];
    }

    // add gravity force
    Force(all, 2).array() -= gravity;

    // contact with ground
    for (int iV = 0; iV < V.rows(); iV++) {

      // support and friction
      if (V(iV, 2) < 0 and Force(iV, 2) < 0) {
        double forceSupport = -Force(iV, 2);
        Force(iV, 2) = 0;

        double fFrictionMaxMag = forceSupport * friction;
        VectorXd velHorizontal(3);
        velHorizontal << Vel(iV, 0), Vel(iV, 1), 0;

        if (fFrictionMaxMag * h > velHorizontal.norm()) {
          fFrictionMaxMag = velHorizontal.norm() / h;
        }

        Force.row(iV) += -fFrictionMaxMag * velHorizontal / velHorizontal.norm();
      }

      // velocity on ground
      if (V(iV, 2) < 0 and Vel(iV, 2) < 0) {
        Vel(iV, 2) = 0;
        V(iV, 2) = 0;
      }
    }

    // contact with objects
    Vo, Fo, V, Vel, Force, del

    Vo, Fo: object mesh
    V, Vo, Fo:=> inside  boolean array of which V is inside
    point_mesh_squared_distance: V, Vo, Fo => IF, Vc, D , indices of faces, closest points

    for (int iV = 0; iV < V.rows(); iV++) {

      the closest point
      vector from v to closest point
      get the normal from vector assuming outside

      if inside[iV] {
        invert the normal
        extend the vector
        that is the new position of v
      }

      // on mesh
      if D[iV] < 2 * delta {
        get the velocity component along normal
        if normal velocity towards surface
          totalvelocity - that dot product

        get the normal force
        if normal force towards surface
          get support force
          total force plus support force
          velocity is now parallel velocity -> velocity magnitude and unit vector

          calculate the friction magnitude by taking the min between support * miu * t and velocity
          add friction force to total force




      }

    }




    Vel = Vel + Force * h;
    Vel = Vel * damping;

    V += Vel * h;

    for (int iRow = 0; iRow < V.rows(); iRow++) {
      for (int iCol=0; iCol < V.cols(); iCol++) {
        Vs(( (iStep + 1) * V.rows() + iRow) * 3 + iCol) = V(iRow, iCol);
      }
    }

    for (int iE = 0; iE < E.rows(); iE++) {
        Fs[iStep * E.rows() + iE] = FEdge[iE];
    }
  }

  return std::make_pair(Vs, Fs);
}


PYBIND11_MODULE(model, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
//
//  m.def("add", &add, "A function that adds two numbers");

  py::class_<Model>(m, "Model")
    .def(py::init<VectorXd, double, double, double, double, MatrixXd, MatrixXi, double>())
    .def("getE", &Model::getE)
    .def("step", &Model::step)
    ;
}
