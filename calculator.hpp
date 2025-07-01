// calculator.hpp
#pragma once
#include <vector>
#include <pybind11/numpy.h>

namespace calculator {
    // Scalar operations
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);

    // Matrix operations
    pybind11::array_t<double> matrix_add(pybind11::array_t<double> a, pybind11::array_t<double> b);
    pybind11::array_t<double> matrix_subtract(pybind11::array_t<double> a, pybind11::array_t<double> b);
    pybind11::array_t<double> matrix_multiply(pybind11::array_t<double> a, pybind11::array_t<double> b);

    // Server
    void start_server(int port, int num_clients);
}