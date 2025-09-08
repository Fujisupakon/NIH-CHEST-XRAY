import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from PIL import Image
import pandas as pd
import os
import csv
from tqdm import tqdm

# --- Config ---
IMAGE_DIR = "images"
CSV_PATH = "Data_Entry_2017.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "nih_model.pth"
METRICS_CSV = "training_metrics.csv" 
PREDICTIONS_CSV = "val_predictions.csv"
THRESHOLD = 0.5

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None): #__init__ เป็นการบอกว่าให้ไปอ่านไฟล์อะไรบ้าง
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data) #อ่านมาแล้วเป็นยังไงบ้าง

    def __getitem__(self, idx): #ดึงจากที่อ่านมาแล้ว
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"]) #อ่านแถวก่อนแล้วถ้าเจอImage Index ให้เปลี่ยนเป็นอ่านแนวตั้งแล้วแทนเลขในช่อง = img_path
        image = Image.open(img_path).convert("RGB") #เปิดไฟล์img เปลี่ยนเป็นสีRGB(windows อ่านได้ดีที่สุด)
        if self.transform:
            image = self.transform(image)

        labels = torch.zeros(len(LABELS)) #ให้สร้างตัวเลขให้แต่ละmuti labels
        for tag in row["Finding Labels"].split("|"): #กำหนดให้อ่านแถวFinding labels
            if tag in LABEL_MAP: 
                labels[LABEL_MAP[tag]] = 1.0 #ถ้าเจอLables ให้แทนเป็น1
        return image, labels #คืนค่ากลับมา

def train_and_save(): #
    dataset = NIHChestXrayDataset(CSV_PATH, IMAGE_DIR, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE, #กำหนดให้ส่งชุดข้อมูลตามbatch ที่กำหนดไว้ได้เลย
        shuffle=True, #ให้สุ่มเพื่อให้เจอคำตอบได้เร็วขึ้น
        num_workers=2, #กำหนดว่าโมเดลใช้ได้ 1 คอล 2 เทรด
        pin_memory=True #move data to gpu #ถ้าเจอข้อมูลที่มันส่งผ่านgpu= true
    ) 

    model = models.resnet50(pretrained=True)#เรียกmodel resnet50 ออกมา #เป็นการให้โมเดลเรียนรู้ไปเลื่อยๆ
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss() #บอกตัวโมเดลว่าผิดจำนวนเท่าไหร่จากทั้งหมด14อัน
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #ให้โมเดลไปคิดใหม่จนกว่าจะถูก แล้วดูอัตราการเรียนรู้ของโมเดล

    metrics = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0 #ดูว่าผิดกี่อัน และ ถูกกี่อัน
        correct = 0
        total = 0 #ทดลองกี่ครั้งจนกว่าจะถูก

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE, non_blocking=True) #ไม่ต้องรอเสร็จ32รูปทั้งหมดรูปไหนเสร็จแล้วส่งไปได้เลย
            labels = labels.to(DEVICE, non_blocking=True) #ไม่ต้องรออ่านเสร็จ32โรค อ่านโรคไหนเสร็จส่งไปได้เลย
            outputs = model(images) #ส่งภาพที่แปลงเข้าโมเดล
            loss = criterion(outputs, labels) #กำหนดค่าloss ว่าผิดพลาดไปเท่าไหร่ คิดแค่รูปภาพเดียว
            loss.backward() #คำณวนค่าผิดพลาดจากท้ายสุดกับตัวก่อนหน้าและทำการไล่ไปเลื่อยๆ
            optimizer.step() #อัพเดตน้ำหนักโมเดลหลังจากผ่านไป1epoch

            running_loss += loss.item() #ผลรวมของของlossทุกอัน=running loss
            preds = (torch.sigmoid(outputs) >= THRESHOLD).float() #เก็บค่าที่ได้จากsigmoidไว้ในpreds
            correct += (preds == labels).all(dim=1).sum().item() #ทั้งหมดในlabelsต้่องถูกถึงจะเท่ากับ= 1
            total += labels.size(0) 
 
        avg_loss = running_loss / len(dataloader) #ค่าความผิดทั้งหมดหารด้วยจำนวนภาพทั้งหมด
        accuracy = correct / total 
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f}") 
        metrics.append([epoch + 1, avg_loss, accuracy]) 

    torch.save(model.state_dict(), MODEL_SAVE_PATH) #เซฟไฟล์น้ำหนักออกมา
    print(f"Model saved to {MODEL_SAVE_PATH}") 

    with open(METRICS_CSV, mode='w', newline='') as f: 
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Accuracy"])
        writer.writerows(metrics)
    print(f"Training metrics saved to {METRICS_CSV}")

    model.eval() #เป็นฟังค์ชันที่บอกว่าหยุดtrain ให้นำข้อมูลที่trainมาทำนาย
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Saving predictions"):
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) >= THRESHOLD).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    with open(PREDICTIONS_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["Index"] + [f"{label}_pred" for label in LABELS] + [f"{label}_true" for label in LABELS]
        writer.writerow(header)
        for i in range(len(all_preds)):
            row = [i + 1]
            row.extend(all_preds[i].astype(int))
            row.extend(all_labels[i].astype(int))
            writer.writerow(row)
    print(f"Validation predictions saved to {PREDICTIONS_CSV}")

if __name__ == "__main__":
    train_and_save()