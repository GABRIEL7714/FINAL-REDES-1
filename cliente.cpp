#include <iostream>
#include <fstream>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

#define PORT 5000
#define SERVER_IP "127.0.0.1"

std::vector<double> load_vector_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(double);
    file.seekg(0);
    std::vector<double> vec(size);
    file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(double));
    return vec;
}

void save_vector_to_file(const std::string& filename, const std::vector<double>& vec) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
}

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Error al crear socket\n";
        return 1;
    }

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        std::cerr << "Dirección inválida\n";
        return 1;
    }

    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Conexión fallida\n";
        return 1;
    }

    std::vector<double> vec = load_vector_from_file("model_vector.bin");
    int32_t vec_size = vec.size();
    send(sock, &vec_size, sizeof(int32_t), 0);
    send(sock, vec.data(), vec_size * sizeof(double), 0);

    recv(sock, &vec_size, sizeof(int32_t), 0);
    std::vector<double> received(vec_size);
    recv(sock, received.data(), vec_size * sizeof(double), 0);

    save_vector_to_file("averaged_vector.bin", received);
    std::cout << "Cliente finalizado.\n";
    close(sock);
    return 0;
}
