import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import pandas as pd

from model import MulticlassClassifier
from export_model import model_to_vector, call_cpp_server  # funciones que exportan pesos y llaman al servidor

# Parámetros
input_dim = 14
num_classes = 3
batch_size = 50
num_epochs = 360

# Leer CSV
csv_path = "Dataset of Diabetes .csv"
df = pd.read_csv(csv_path, header=None, skiprows=1)
X_np = df.iloc[:, :input_dim].values.astype(np.float32)
y_onehot_np = df.iloc[:, -num_classes:].values.astype(np.float32)

X = torch.tensor(X_np)
y = torch.tensor(y_onehot_np)

# Dataset y DataLoaders
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelo
model = MulticlassClassifier(input_dim=input_dim, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_tracker, test_tracker, accuracy_tracker = [], [], []
y_true, y_pred = [], []

# Entrenamiento por épocas
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits, log_vars = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_tracker.append(train_loss)

    # 🔄 Extraer pesos y llamar servidor C++ por cada época
    vector = model_to_vector(model)
    new_vector = call_cpp_server(vector)  # <- recibe la media desde C++
    model.load_state_dict(model.state_dict())  # <-- aquí puedes actualizar si se requiere con `new_vector`

    # Evaluación
    model.eval()
    test_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits, log_vars = model(batch_x)
            loss = criterion(logits, batch_y)
            test_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == torch.argmax(batch_y, dim=1)).sum().item()
            total += batch_x.size(0)
            y_true.extend(torch.argmax(batch_y, dim=1).tolist())
            y_pred.extend(pred.tolist())

    acc = correct / total
    test_tracker.append(test_loss / len(test_loader))
    accuracy_tracker.append(acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {acc:.4f}")

# Gráficos
plt.plot(train_tracker, label='Train Loss')
plt.plot(test_tracker, label='Test Loss')
plt.plot(accuracy_tracker, label='Accuracy')
plt.legend()
plt.title("Loss & Accuracy over Epochs")
plt.grid(True)
plt.savefig("loss_accuracy.png")
# plt.show()


# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
# plt.show()

# Reporte
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))
