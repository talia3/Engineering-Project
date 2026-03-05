import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 160

# -----------------------------
# Load images
# -----------------------------

def load_images(base_path):

    X = []
    y = []

    classes = {
        "no_manipulation":0,
        "mouth":1,
        "eyes":2
    }

    for cls in classes:
        folder = os.path.join(base_path, cls)

        for img_name in os.listdir(folder):

            path = os.path.join(folder, img_name)

            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            X.append(img)
            y.append(classes[cls])

    X = np.array(X)
    y = np.array(y)

    return X,y


# -----------------------------
# Dataset class
# -----------------------------

class FaceDataset(Dataset):

    def __init__(self,X,y,transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        img = self.X[idx]

        if self.transform:
            img = self.transform(img)

        label = self.y[idx]

        return img,label


# -----------------------------
# CNN model
# -----------------------------

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(64*20*20,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x


# -----------------------------
# Load data
# -----------------------------

X,y = load_images("dataset")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = FaceDataset(X_train,y_train,transform)
test_dataset = FaceDataset(X_test,y_test,transform)

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=16)


# -----------------------------
# Train
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)


EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# -----------------------------
# Test accuracy
# -----------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (predicted==labels).sum().item()

print("Accuracy:",correct/total)