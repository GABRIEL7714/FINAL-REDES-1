#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>

#define VECTOR_SIZE 10566
#define SERVER_IP "127.0.0.1"
#define SERVER_PORT 12345

int main() {
    // Leer vector desde archivo binario
    std::ifstream infile("model_vector.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "No se pudo abrir model_vector.bin\n";
        return 1;
    }

    std::vector<float> vec(VECTOR_SIZE);
    infile.read(reinterpret_cast<char*>(vec.data()), VECTOR_SIZE * sizeof(float));
    infile.close();

    // Crear socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket error");
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

    if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connect error");
        return 1;
    }

    // Enviar vector
    send(sock, vec.data(), VECTOR_SIZE * sizeof(float), 0);

    // Recibir vector promedio
    std::vector<float> avg_vec(VECTOR_SIZE);
    size_t total_received = 0;
    while (total_received < avg_vec.size() * sizeof(float)) {
        ssize_t bytes = recv(sock,
                             reinterpret_cast<char*>(avg_vec.data()) + total_received,
                             avg_vec.size() * sizeof(float) - total_received,
                             0);
        if (bytes <= 0) {
            perror("Receive error");
            close(sock);
            return 1;
        }
        total_received += bytes;
    }

    close(sock);

    // Mostrar primeros 10 elementos recibidos
    std::cout << "Vector promedio recibido (primeros 10 elementos):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << avg_vec[i] << " ";
    }
    std::cout << "\n";

    return 0;
}

