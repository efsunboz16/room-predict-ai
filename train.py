import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# --- 1. KAGGLE İÇİN YENİ HİPERPARAMETRELER VE YOLLAR ---
BATCH_SIZE = 64 # ResNet18 hafiftir, T4x2 GPU'ları tam kapasite doyurmak için ideal.
IMG_SIZE = 224  # ResNet'in dünyayı en net gördüğü standart çözünürlük.
EPOCHS = 20     # Transfer learning çok hızlı öğrenir.
LEARNING_RATE = 0.0005 

# GÜNCELLENEN VERİ SETİ YOLU
DATA_DIR = "/kaggle/input/datasets/efsunboz/oda-dataset/room-dataset"
SAVE_PATH = "/kaggle/working/best_real_estate_resnet.pth"

# --- 2. TRANSFER LEARNING MİMARİSİ (ResNet-18) ---
def get_pretrained_model(num_classes):
    # Önceden eğitilmiş devasa modeli indir
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Modelin son karar katmanını (fully connected layer) bul
    num_ftrs = model.fc.in_features
    
    # Kendi sınıflarımıza göre son katmanı baştan yaz
    # GÜNCELLEME 1: Dropout %50'den %60'a çıkarıldı. Modelin ezberlemesi zorlaştı.
    model.fc = nn.Sequential(
        nn.Dropout(0.6), 
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# --- 3. ANA EĞİTİM DÖNGÜSÜ ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan donanım birimi: {device}")

    # GÜÇLENDİRİLMİŞ EĞİTİM VERİSİ (Overfitting'i engeller)
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop(IMG_SIZE), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # GÜNCELLEME 2: Random Erasing eklendi. Resmin belli kısımlarını rastgele silerek ezberi bozar.
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)) 
    ])

    # DOĞRULAMA VERİSİ (Temiz, hilesiz sınav formatı)
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    class_names = full_dataset.classes
    print(f"Sınıflar ({len(class_names)} adet): {class_names}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # CPU Aşçılarını Tam Kapasite Çalıştırıyoruz
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Modeli Yükle ve Çift GPU (T4 x2) Ayarını Yap
    model = get_pretrained_model(num_classes=len(class_names)).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Harika! {torch.cuda.device_count()} adet GPU kullanılıyor.")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    
    # GÜNCELLEME 3: weight_decay (Ağırlık Cezası) 1e-4'ten 1e-2'ye çıkarıldı. Frene sert basıldı.
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # GÜNCELLEME 4: Öğrenme hızı zamanlayıcısı (Scheduler) eklendi. 
    # Eğitim sonuna yaklaştıkça öğrenme hızını (Learning Rate) kosinüs eğrisi gibi yavaşça düşürür.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda') 

    best_val_acc = 0.0

    print("\n--- Profesyonel Transfer Learning Eğitimi Başlıyor ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            # tqdm bar'ında güncel öğrenme hızını da görebilmek için lr eklendi
            current_lr = scheduler.get_last_lr()[0]
            train_bar.set_postfix(loss=running_loss/total_train, acc=100.*correct_train/total_train, lr=current_lr)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix(loss=val_loss/total_val, acc=100.*correct_val/total_val)
                
        epoch_val_acc = 100. * correct_val / total_val
        print(f"Epoch {epoch+1} Özet -> Eğitim Başarısı: %{100.*correct_train/total_train:.2f} | Doğrulama Başarısı: %{epoch_val_acc:.2f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f">>> Yeni zirve! Model '{SAVE_PATH}' olarak kaydedildi!")
            
        # GÜNCELLEME 4'ÜN DEVAMI: Her Epoch sonunda scheduler'ı bir adım ilerlet (hızı düşür)
        scheduler.step()

if __name__ == '__main__':
    main()