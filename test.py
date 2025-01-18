import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

fruits = ["Manzana", "Platano", "Uva", "Mango", "Fresa"]

test_dir = "./FruitsClassification/test"


# Transformaciones (equivalente a rescale=1./255 en Keras)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        # Convertir a tensor
    # Preprocesamiento estilo ResNet (imagenet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Dataset para imágenes en PyTorch
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Dataloaders para cargar datos por lotes (batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Cargamos nuestro modelo ResNet50 para frutas
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5) # 5 clases
model.load_state_dict(torch.load("best_fruit_model.pt", weights_only=True, map_location=torch.device('cpu')))

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test
model.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        test_predictions.extend(predictions)
        test_labels.extend(batch_y.cpu().numpy())

# Calculate and display test metrics
test_acc = accuracy_score(test_labels, test_predictions)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Display confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=fruits,
    yticklabels=fruits,
)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(test_labels, test_predictions, target_names=fruits))
