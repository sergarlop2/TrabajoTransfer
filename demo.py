import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import picamera
import io
import time

# Lista de frutas
fruits = ["Manzana", "Platano", "Uva", "Mango", "Fresa"]

# Configuración de transformaciones para las imágenes capturadas
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Cargamos nuestro modelo ResNet50 reentrenado
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 5 clases
model.load_state_dict(torch.load("best_fruit_model.pt", weights_only=True, map_location=torch.device('cpu')))
model.eval()

# Inicializamos la cámara
with picamera.PICamera() as camera:
    camera.resolution = (224, 224)  # Configuramos la resolución para que coincida con la entrada del modelo
    camera.framerate = 30
    time.sleep(2)  # Esperamos para la estabilización de la cámara

    print("Comenzando demo:\n")
    
    while True:
        # Capturamos una imagen en memoria
        stream = io.BytesIO()
        camera.capture(stream, format='jpeg')
        stream.seek(0)
        
        # Abrimos la imagen desde el flujo de bytes y ls convertimos a un objeto PIL
        image = Image.open(stream)
        
        # Aplicamos las transformaciones
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Hacemos la inferencia
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
        
        # Obtenemos la etiqueta de la clase tras la prediccion
        predicted_label = fruits[predicted_class.item()]
        
        # Mostramos el resultado
        print(f"Predicción: {predicted_label}")

        # Esperamos un poco antes de capturar la siguiente imagen
        time.sleep(1)
