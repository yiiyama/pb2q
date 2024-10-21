#ifndef pyarray_h
#define pyarray_h

#include <memory>
#include <utility>
#include <exception>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

// helper function to avoid making a copy when returning a py::array_t
// author: https://github.com/YannickJadoul
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
// copied from: https://github.com/ssciwr/pybind11-numpy-example/blob/main/src/pybind11_numpy_example_python.cpp
template <typename Sequence>
inline py::array_t<typename Sequence::value_type>
as_pyarray(Sequence &&seq) {
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr =
      std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
  });
  seq_ptr.release();
  return py::array(size, data, capsule);
}

// version with shape specification
template <typename Sequence>
inline py::array_t<typename Sequence::value_type>
as_pyarray(Sequence &&seq, py::array::ShapeContainer shape) {
  unsigned ssize(1);
  for (auto n : *shape)
    ssize *= n;
  if (ssize != seq.size())
    throw std::runtime_error("Shape mismatch");
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr =
      std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
  });
  seq_ptr.release();
  return py::array(shape, data, capsule);
}

#endif
