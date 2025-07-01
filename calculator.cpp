#include "calculator.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <thread>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace calculator {

// Scalar operations

double add(double a, double b) {
    return a + b;
}

double subtract(double a, double b) {
    return a - b;
}

double multiply(double a, double b) {
    return a * b;
}

double divide(double a, double b) {
    if (b == 0) throw std::runtime_error("Division by zero");
    return a / b;
}

// Matrix operations

py::array_t<double> matrix_add(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info buf1 = a.request(), buf2 = b.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
        throw std::runtime_error("Number of dimensions must be 2");
    if (buf1.shape[0] != buf2.shape[0] || buf1.shape[1] != buf2.shape[1])
        throw std::runtime_error("Input shapes must match");

    auto result = py::array_t<double>(buf1.shape);
    py::buffer_info buf3 = result.request();

    double* ptr1 = static_cast<double*>(buf1.ptr);
    double* ptr2 = static_cast<double*>(buf2.ptr);
    double* ptr3 = static_cast<double*>(buf3.ptr);

    for (ssize_t i = 0; i < buf1.size; i++)
        ptr3[i] = ptr1[i] + ptr2[i];

    return result;
}

py::array_t<double> matrix_subtract(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info buf1 = a.request(), buf2 = b.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
        throw std::runtime_error("Number of dimensions must be 2");
    if (buf1.shape[0] != buf2.shape[0] || buf1.shape[1] != buf2.shape[1])
        throw std::runtime_error("Input shapes must match");

    auto result = py::array_t<double>(buf1.shape);
    py::buffer_info buf3 = result.request();

    double* ptr1 = static_cast<double*>(buf1.ptr);
    double* ptr2 = static_cast<double*>(buf2.ptr);
    double* ptr3 = static_cast<double*>(buf3.ptr);

    for (ssize_t i = 0; i < buf1.size; i++)
        ptr3[i] = ptr1[i] - ptr2[i];

    return result;
}

py::array_t<double> matrix_multiply(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info buf1 = a.request(), buf2 = b.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
        throw std::runtime_error("Number of dimensions must be 2");
    if (buf1.shape[1] != buf2.shape[0])
        throw std::runtime_error("Matrix shapes are not aligned for multiplication");

    auto result = py::array_t<double>({buf1.shape[0], buf2.shape[1]});
    py::buffer_info buf3 = result.request();

    double* ptr1 = static_cast<double*>(buf1.ptr);
    double* ptr2 = static_cast<double*>(buf2.ptr);
    double* ptr3 = static_cast<double*>(buf3.ptr);

    ssize_t rows = buf1.shape[0];
    ssize_t cols = buf2.shape[1];
    ssize_t inner = buf1.shape[1];

    for (ssize_t i = 0; i < rows; ++i) {
        for (ssize_t j = 0; j < cols; ++j) {
            ptr3[i * cols + j] = 0;
            for (ssize_t k = 0; k < inner; ++k) {
                ptr3[i * cols + j] += ptr1[i * inner + k] * ptr2[k * cols + j];
            }
        }
    }

    return result;
}

// Server functions

std::vector<double> receive_vector(int client_sock) {
    int32_t vec_size;
    recv(client_sock, &vec_size, sizeof(int32_t), 0);
    std::vector<double> vec(vec_size);
    recv(client_sock, vec.data(), vec_size * sizeof(double), 0);
    return vec;
}

void send_vector(int client_sock, const std::vector<double>& vec) {
    int32_t vec_size = vec.size();
    send(client_sock, &vec_size, sizeof(int32_t), 0);
    send(client_sock, vec.data(), vec_size * sizeof(double), 0);
}

void start_server(int port, int num_clients) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    bind(server_fd, (struct sockaddr*)&address, sizeof(address));
    listen(server_fd, num_clients);

    std::vector<std::vector<double>> client_vectors;
    std::vector<int> client_sockets;

    std::cout << "Esperando " << num_clients << " clientes...\n";

    for (int i = 0; i < num_clients; ++i) {
        int client_sock = accept(server_fd, nullptr, nullptr);
        client_sockets.push_back(client_sock);
        std::cout << "Cliente " << i + 1 << " conectado.\n";
        client_vectors.push_back(receive_vector(client_sock));
    }

    size_t vec_size = client_vectors[0].size();
    std::vector<double> mean(vec_size, 0.0);

    for (const auto& vec : client_vectors) {
        for (size_t i = 0; i < vec_size; ++i)
            mean[i] += vec[i] / num_clients;
    }

    for (int client_sock : client_sockets) {
        send_vector(client_sock, mean);
        close(client_sock);
    }

    close(server_fd);
    std::cout << "Proceso completado.\n";
}

} // namespace calculator
