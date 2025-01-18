import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import time
from picamera2 import Picamera2

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
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()  # Iniciamos la cámara

# Configuramos la ventana de OpenCV para mostrar el video
cv2.namedWindow("Cámara en Tiempo Real", cv2.WINDOW_NORMAL)

print("Comenzando demo:\n")

while True:
    # Capturamos una imagen
    image_array = picam2.capture_array()  # Capturamos la imagen como un array numpy
    
    # Convertimos la imagen numpy (RGB) a formato PIL para las transformaciones
    image = Image.fromarray(image_array)
    
    # Aplicamos las transformaciones
    image_tensor = transform(image).unsqueeze(0) 

    # Hacemos la inferencia
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    
    # Obtenemos la etiqueta de la clase tras la predicción
    predicted_label = fruits[predicted_class.item()]
    
    # Mostramos el resultado por consola
    print(f"Predicción: {predicted_label}")
    
    # Convertimos la imagen de RGB a BGR para que OpenCV la maneje correctamente
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Mostramos la imagen en la ventana de OpenCV
    cv2.imshow("Cámara en Tiempo Real", image_bgr)
    
    # Si el usuario presiona 'q', salimos del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerramos la ventana de OpenCV y liberamos la cámara
cv2.destroyAllWindows()
picam2.stop()
