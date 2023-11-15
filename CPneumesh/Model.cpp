#include "Model.h"
#include "pybind11/pybind11.h"

using namespace std;
using namespace Eigen;
namespace py = pybind11;

Model::Model(VectorXd K, double h, double gravity, double damping, double friction,
             MatrixXd v0, MatrixXi e, double CONTRACTION_SPEED, std::string objDir)
   {

  this->V0 = v0;
  this->V = v0;
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

  // check if objDir length zero
  if (objDir.length() == 0) {
    this->hasObstacle = false;
  }
  else {
    this->hasObstacle = true;
    igl::readOBJ(objDir, Vo, Fo);
  }


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

    // calculate edge force
    VectorXd FEdge;
    FEdge.resize(E.size());
    auto L = getL(V, E);
    auto Vec01 = V(E(all, 1), all) - V(E(all, 0), all); // vectors from V[e[0]] to V[e[1]]

    for (int iE = 0; iE < E.rows(); iE++) {
      FEdge[iE] = (L[iE] - L0[iE]) * K[iE];   // contraction force is positive

      Force.row(E(iE, 0)) += Vec01.row(iE) / L[iE] * FEdge[iE];
      Force.row(E(iE, 1)) -= Vec01.row(iE) / L[iE] * FEdge[iE];
    }

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

MatrixXd Model::stepForGym(VectorXd lengths, int numSteps) {
  VectorXd LTarget = lengths;   // target length of pneumatic actuators

  // define Vs in MatrixXd that stores all Vs of every time step
  MatrixXd Vs((numSteps + 1) * V.rows(), 3);

  // store V0 in Vs
  for (int iRow = 0; iRow < V.rows(); iRow++) {
    for (int iCol=0; iCol < V.cols(); iCol++) {
      Vs((0 * V.rows() + iRow), iCol) = V(iRow, iCol);
    }
  }


  for (int iStep=0; iStep < numSteps; iStep++) {
    Force.setZero();

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

    // calculate edge force
    VectorXd FEdge;
    FEdge.resize(E.size());
    auto L = getL(V, E);
    auto Vec01 = V(E(all, 1), all) - V(E(all, 0), all); // vectors from V[e[0]] to V[e[1]]

    for (int iE = 0; iE < E.rows(); iE++) {
      FEdge[iE] = (L[iE] - L0[iE]) * K[iE];   // contraction force is positive

      Force.row(E(iE, 0)) += Vec01.row(iE) / L[iE] * FEdge[iE];
      Force.row(E(iE, 1)) -= Vec01.row(iE) / L[iE] * FEdge[iE];
    }

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

    // calculate the number of steps every 0.1 seconds
//    int numStepsPer0_1s = 0.1 / h;

    // handle collision every n steps
    if (iStep % 1 == 0 and hasObstacle) {
      handleMeshCollision(Vo, Fo, V, Vel, Force, 1e-4, 0.9, h);
    }

    Vel = Vel + Force * h;
    Vel = Vel * damping;

    V += Vel * h;

    // store V in Vs
    for (int iRow = 0; iRow < V.rows(); iRow++) {
      for (int iCol=0; iCol < V.cols(); iCol++) {
        Vs(( (iStep + 1) * V.rows() + iRow), iCol) = V(iRow, iCol);
      }
    }

  }

  return Vs;
}

void Model::handleMeshCollision(const Eigen::MatrixXd &Vo, const Eigen::MatrixXi &Fo, Eigen::MatrixXd &VCollision, Eigen::MatrixXd &VelCollision, Eigen::MatrixXd &ForceCollision, double delta, double mu, double h) {
    // inputs:
        // Vo, Fo: mesh vertices and faces
        // V, Vel, Force: point, velocity and force
        // delta: distance considered for the collision
        // mu: friction coefficient
        // h: time step

    // check if V inside mesh
    // define inside as a bool Eigen Array

    const int nV = VCollision.rows();
    Eigen::Array<bool, Eigen::Dynamic, 1> inside(nV, 1);


    // set V to be const
    const Eigen::MatrixXd V_const = VCollision;

    try {
//        igl::copyleft::cgal::points_inside_component(Vo, Fo, V_const, inside);

//
//      /// @param[in] source  3-vector origin of ray
//      /// @param[in] dir  3-vector direction of ray
//      /// @param[in] V  #V by 3 list of mesh vertex positions
//      /// @param[in] F  #F by 3 list of mesh face indices into V
//      /// @param[out] hits  **sorted** list of hits
//      /// @return true if there were any hits (hits.size() > 0)
//      ///
//      /// \see AABB
//      template <
//        typename Derivedsource,
//        typename Deriveddir,
//        typename DerivedV,
//        typename DerivedF>
//      IGL_INLINE bool ray_mesh_intersect(
//        const Eigen::MatrixBase<Derivedsource> & source,
//        const Eigen::MatrixBase<Deriveddir> & dir,
//        const Eigen::MatrixBase<DerivedV> & V,
//        const Eigen::MatrixBase<DerivedF> & F,
//        std::vector<igl::Hit> & hits);

        // use ray mesh intersect to check if V inside mesh
        std::vector<igl::Hit> hits;
        for (int iV = 0; iV < VCollision.rows(); iV++) {
          Eigen::RowVector3d source = VCollision.row(iV);
          // any direction
          Eigen::RowVector3d dir = Eigen::RowVector3d(1, 0, 0);

          bool hit = igl::ray_mesh_intersect(source, dir, Vo, Fo, hits);
          // if hits number is odd, then V is inside mesh
          if (hits.size() % 2 == 1) {
            inside(iV) = true;
          }
          else {
            inside(iV) = false;
          }

        }


    }
    catch (const std::exception& e) {
         std::cout << "Exception caught: " << e.what() << std::endl;
    }

    // check the distance between V and mesh
    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    igl::point_mesh_squared_distance(VCollision, Vo, Fo, sqrD, I, C);

    for (int iV = 0; iV < VCollision.rows(); iV++) {
        // get the closest point
        Eigen::RowVector3d closestPoint = C.row(iV);

        // get the distance
        double d = sqrt(sqrD(iV));

        // get inside
        bool isInside = inside(iV);

        // get unit normal vector from V to closestPoint (assuming it's insid)
        Eigen::RowVector3d normal = (closestPoint - VCollision.row(iV)).normalized();

        // if inside, move V outside of the mesh by delta (on mesh)
        if (isInside) {
//          std::cout<<"inside"<<std::endl;
//          std::cout<<VCollision.row(iV)<<std::endl;
            // move V along the normal by d + delta
            VCollision.row(iV) += (d + delta) * normal;
        }
        else {
            // invert the normal (V is from outside)
            normal = -normal;
        }

        // if on mesh (d < 2 * delta)
        if ( d < 2 * delta or isInside) {
//          std::cout<<"on mesh"<<d<<std::endl;

            // compute the velocity component along the normal
            double velNormal = VelCollision.row(iV).dot(normal);

            // compute the parallel velocity
            Eigen::RowVector3d velParallel = VelCollision.row(iV) - velNormal * normal;

            // if velNormal is towards the mesh, remove it from total velocity, calculate parallel velocity
            if (velNormal < 0) {
                // remove velNormal from total velocity
                VelCollision.row(iV) -= velNormal * normal;
            }

            // compute the force component along the normal
            double forceNormal = ForceCollision.row(iV).dot(normal);

            // if forceNormal is towards the mesh
            if (forceNormal < 0) {
                // calculate support force
                double supportForce = -forceNormal;

                // add support force to total force
                ForceCollision.row(iV) += supportForce * normal;

                // compute friction force
                double frictionForceMag = mu * supportForce;

                // if friction force * h is greater than parallel velocity, set friction force * h to parallel velocity
                if (frictionForceMag * h > velParallel.norm()) {
                    // set friction force to parallel velocity / h
                    frictionForceMag = velParallel.norm() / h;
                }

                // compute friction force
                Eigen::RowVector3d frictionForce = -frictionForceMag * velParallel.normalized();

                // add friction force to total force
                ForceCollision.row(iV) += frictionForce;

            }

        }

    }
    }



PYBIND11_MODULE(model, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  py::register_exception<std::exception>(m, "PyExp");

  py::class_<Model>(m, "Model")
    .def(py::init<VectorXd, double, double, double, double, MatrixXd, MatrixXi, double, std::string>())
    .def("getE", &Model::getE)
    .def("step", &Model::step)
    .def("stepForGym", &Model::stepForGym)
    .def("handleMeshCollision", &Model::handleMeshCollision)
    ;
}
