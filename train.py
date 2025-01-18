import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from torchvision import transforms, models
from torchinfo import summary

import matplotlib.pyplot as plt


train_dir  = "./FruitsClassification/train"
valid_dir = "./FruitsClassification/valid"


# Transformaciones (equivalente a rescale=1./255 en Keras)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        # Convertir a tensor
    # Preprocesamiento estilo ResNet (imagenet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Dataset para imágenes en PyTorch
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)

# Dataloaders para cargar datos por lotes (batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)


#%%
# Red neruonal con transfer learning

# Definir modelo con ResNet50 preentrenado:
# Cargar ResNet50 preentrenado en PyTorch
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')


# Imprime un resumen utilizando torchinfo 
summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


# Congelar todas las capas excepto las de batch normalization
for name, param in model.named_parameters():
    if 'bn' not in name:  # 'bn' se refiere a las capas BatchNorm
        param.requires_grad = False  # Congelar todas las demás capas
    else:
        param.requires_grad = True   # Asegurar que BatchNorm sea entrenable

# Modificar la última capa de ResNet50 para nuestro número de clases (6)
num_ftrs = model.fc.in_features


model.fc = nn.Linear(num_ftrs, 5)  # 5 clases


# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Entrenamiento y evaluación:
# Definir optimizador y función de pérdida
#optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


# Entrenamiento
num_epochs = 20
best_val_acc = 0.0
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        running_corrects += torch.sum(outputs.argmax(dim=1) == labels)
    
    train_loss = running_loss / total
    train_accuracy = 100.0 * running_corrects.item() / total
    train_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)
    

    
    
    # Validación
    model.eval() 
    corrects = 0
    val_loss = 0.0
    total = 0

    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += loss.item()
            total += labels.size(0)
            corrects += torch.sum(outputs.argmax(dim=1) == labels)
    
    val_loss /= len(valid_loader)
    val_accuracy = 100.0 * corrects.item() / total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_accuracy)  
    val_acc = corrects.double() / len(valid_dataset)
    
    
 
    print(f'Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}' )

    # Guardamos el mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_fruit_model.pt")
   

# Gráficas de evolución de Accuracy y Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

    

