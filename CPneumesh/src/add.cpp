#include <iostream>
#include <pybind11/eigen.h>

Eigen::RowVector3d add(int i, int j) {
  auto a = Eigen::RowVector3d(1,2,3);

  return a;
}


class MyClass {
public:
  Eigen::MatrixXd big_mat = Eigen::MatrixXd::Zero(10000, 10000);
  Eigen::MatrixXd getMatrix(Eigen::MatrixXd& a) {
    std::cout<<a<<std::endl;
    auto b = Eigen::MatrixXd::Ones(a.rows(),a.cols());
    std::cout<<b<<std::endl;
    a = a + b;
    std::cout<<a<<std::endl;
    std::cout<<a(0,0)<<std::endl;
    return a;
  }
  const Eigen::MatrixXd &viewMatrix() { return big_mat; }
};
