import torch
import numpy as np
import struct
import socket
from model import MulticlassClassifier

def model_to_vector(model):
    """
    Convierte todos los parámetros del modelo (pesos y biases) en un solo vector 1D.
    Además guarda este vector en un archivo binario.
    """
    with torch.no_grad():
        params = []
        for param in model.parameters():
            params.append(param.view(-1))  # flatten
        vector = torch.cat(params)
        vector_np = vector.cpu().numpy().astype(np.float32)

        # Guardar en archivo binario para inspección si se desea
        with open("model_vector.bin", "wb") as f:
            f.write(struct.pack(f'{len(vector_np)}f', *vector_np))

        print(f"Vector de tamaño {len(vector_np)} guardado en model_vector.bin")
        return vector_np


def call_cpp_server(vector, host='127.0.0.1', port=12345):
    """
    Envía el vector al servidor C++ y recibe el vector medio procesado.
    """
    with socket.create_connection((host, port)) as sock:
        # Enviar el tamaño del vector como entero (4 bytes)
        sock.sendall(struct.pack('I', len(vector)))

        # Enviar los datos como floats
        sock.sendall(struct.pack(f'{len(vector)}f', *vector))

        # Recibir el vector promedio (float32)
        data = b''
        expected_bytes = len(vector) * 4  # 4 bytes por float
        while len(data) < expected_bytes:
            packet = sock.recv(expected_bytes - len(data))
            if not packet:
                raise ConnectionError("Conexión cerrada prematuramente por el servidor")
            data += packet

        mean_vector = struct.unpack(f'{len(vector)}f', data)
        return np.array(mean_vector, dtype=np.float32)
