import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset, random_split
import kagglehub
from sklearn.model_selection import train_test_split

# Function to find correct file path
def find_correct_path(base_path, target_name):
    for root, dirs, files in os.walk(base_path):
        for name in dirs + files:
            if name.lower() == target_name.lower():
                return os.path.join(root, name)
    return None

# Function to get actual dataset path
def get_actual_path(base_path, parts):
    current_path = base_path
    for part in parts:
        next_path = find_correct_path(current_path, part)
        if next_path is None:
            raise FileNotFoundError(f"Directory or file {part} not found under {current_path}")
        current_path = next_path
    return current_path

# Download dataset
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Path to dataset files:", path)

base_data_dir = path
image_dir_part_1 = get_actual_path(base_data_dir, ['HAM10000_images_part_1'])
image_dir_part_2 = get_actual_path(base_data_dir, ['HAM10000_images_part_2'])
metadata_csv_path = os.path.join(base_data_dir, 'HAM10000_metadata.csv')

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define custom dataset class
class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, root_dir1, root_dir2, transform=None):
        self.frame = dataframe
        self.label_mapping = {label: idx for idx, label in enumerate(self.frame['dx'].unique())}
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform
    
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        img_name = self.frame.iloc[idx, 1] + '.jpg'
        img_path1 = os.path.join(self.root_dir1, img_name)
        img_path2 = os.path.join(self.root_dir2, img_name)
        
        image = datasets.folder.default_loader(img_path1 if os.path.exists(img_path1) else img_path2)
        label = self.label_mapping[self.frame.iloc[idx, 2]]

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load dataset
metadata_df = pd.read_csv(metadata_csv_path)
train_df, test_val_df = train_test_split(metadata_df, test_size=0.2, stratify=metadata_df['dx'])
val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['dx'])

print(f"Training Samples: {len(train_df)}, Validation Samples: {len(val_df)}, Testing Samples: {len(test_df)}")

train_dataset = SkinCancerDataset(train_df, image_dir_part_1, image_dir_part_2, data_transforms['train'])
val_dataset = SkinCancerDataset(val_df, image_dir_part_1, image_dir_part_2, data_transforms['val'])
test_dataset = SkinCancerDataset(test_df, image_dir_part_1, image_dir_part_2, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Define model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.label_mapping))
model.to(device)
print(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train function
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} started")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    return model

trained_model = train_model(model, criterion, optimizer)

# Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

test_accuracy = evaluate_model(trained_model, test_loader)

# Save model
model_path = 'resnet50_skin_cancer_model.pth'
torch.save(trained_model.state_dict(), model_path)
print(f'Model saved to {model_path}')