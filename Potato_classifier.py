import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# model = models.mobilenet_v2(pretrained=True)
model = models.mobilenet_v3_small(pretrained=True)

# ИЗМЕНЕНИЕ: Количество классов
num_classes = 4 
# model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5374630689620972, 0.46548762917518616, 0.36194294691085815],
                         [0.21799011528491974, 0.1950533241033554, 0.16310831904411316]),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.ColorJitter(brightness=.5),
])

data_transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5374630689620972, 0.46548762917518616, 0.36194294691085815],
                         [0.21799011528491974, 0.1950533241033554, 0.16310831904411316]),
])

train_dataset = ImageFolder('data/train', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = ImageFolder('data/test', transform=data_transforms_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ИЗМЕНЕНИЕ: Получаем имена классов для визуализации
class_names = train_dataset.classes

device = torch.device("cuda")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 200
eff = 0

def visualize_metrics(y_true, y_pred, class_names):
    """
    Визуализирует матрицу ошибок, Precision и Recall.
    """
    # Расчет метрик
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- Отчет по метрикам ---")
    for i, name in enumerate(class_names):
        print(f"Класс '{name}':")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
    print("-------------------------")

    # Визуализация матрицы ошибок
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.show()

# Основной цикл обучения
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    if epoch % 10 == 0:
        model.eval()
        correct_answers = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                pred_classes = outputs.argmax(dim=1)
                
                correct_answers += (pred_classes == labels).sum()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(pred_classes.cpu().numpy())
        
        correct_answers_percentage = correct_answers / len(test_loader.dataset) * 100.0
        
        # ИЗМЕНЕНИЕ: Вызов функции визуализации
        print("\nОценка на тестовом наборе:")
        visualize_metrics(all_labels, all_predictions, class_names)
        
        if correct_answers_percentage >= eff:
            torch.save(model.state_dict(), f"best{epoch}.pth")
            eff = correct_answers_percentage
            
        print(f"Correct answers percentage: {correct_answers_percentage:.4f}")

    scheduler.step()

# Финальная оценка после обучения
model.eval()
correct_answers = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pred_classes = outputs.argmax(dim=1)
        correct_answers += (pred_classes == labels).sum()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(pred_classes.cpu().numpy())

correct_answers_percentage = correct_answers / len(test_loader.dataset) * 100.0
print(f"\nИтоговая оценка после обучения:")
print(f"Correct answers percentage: {correct_answers_percentage:.4f}")
visualize_metrics(all_labels, all_predictions, class_names)