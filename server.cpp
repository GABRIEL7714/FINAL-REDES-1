// server.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

std::mutex mtx;
int VECTOR_SIZE = 10566;
int NUM_CLIENTS = 1;

void handle_client(int client_sock) {
    std::vector<float> buffer(VECTOR_SIZE);
    recv(client_sock, buffer.data(), VECTOR_SIZE * sizeof(float), 0);

    // Calcular promedio (con solo un cliente es el mismo vector)
    std::vector<float> avg = buffer;

    // Enviar al cliente
    send(client_sock, avg.data(), avg.size() * sizeof(float), 0);
    close(client_sock);
}

int main() {
    int port = 12345;

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);  // aceptar hasta 5 conexiones pendientes

    std::cout << "Servidor corriendo en el puerto " << port << "...\n";

    while (true) {
        int client_sock = accept(server_fd, nullptr, nullptr);
        if (client_sock >= 0) {
            std::thread(handle_client, client_sock).detach();
        }
    }

    close(server_fd);
    return 0;
}
