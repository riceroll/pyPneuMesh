#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
//#include <pybind11/eigen.h>
#include "add.cpp"

namespace py = pybind11;

PYBIND11_MODULE(greet, m) {
m.doc() = "pybind11 example plugin"; // optional module docstring

m.def("add", &add, "A function that adds two numbers");

py::class_<MyClass>(m, "MyClass")
  .def(py::init<>())
  .def("copy_matrix", &MyClass::getMatrix) // Makes a copy!
  .def("get_matrix", &MyClass::getMatrix, py::return_value_policy::reference_internal)
  .def("view_matrix", &MyClass::viewMatrix, py::return_value_policy::reference_internal)
  ;
}


