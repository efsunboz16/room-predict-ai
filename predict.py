import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- 1. SINIFLAR VE AYARLAR ---
CLASS_NAMES = [
    'bathroom', 'bedroom', 'dining', 'gaming', 'kitchen', 
    'laundry', 'living', 'office', 'terrace', 'yard'
]

MODEL_PATH = "best_real_estate_resnet.pth"

# --- 2. MODEL İSKELETİ ---
def get_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# --- 3. TAHMİN FONKSİYONU ---
def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: '{MODEL_PATH}' modeli bulunamadı! Lütfen modelin bu klasörde olduğundan emin olun.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modeli yükle
    model = get_model(len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.to(device)
    model.eval()

    # Fotoğrafı modele uygun hale getir
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        top_prob, top_catid = torch.topk(probabilities, 3)

        print("\n" + "="*50)
        print(f"📸 Analiz Edilen Fotoğraf: {os.path.basename(image_path)}")
        print("="*50)
        
        for i in range(top_prob.size(0)):
            score = top_prob[i].item() * 100
            room_name = CLASS_NAMES[top_catid[i]].upper()
            print(f"  {i+1}. Tahmin: %{score:.2f} ihtimalle {room_name}")
            
        print("="*50 + "\n")

    except Exception as e:
        print(f"Fotoğraf işlenirken bir hata oluştu: {e}")

# --- 4. OTOMATİK FOTOĞRAF BULMA ---
if __name__ == '__main__':
    # Mevcut klasördeki tüm dosyaları tara
    gecerli_uzantilar = ('.jpg', '.jpeg', '.png')
    klasordeki_fotograflar = [dosya for dosya in os.listdir() if dosya.lower().endswith(gecerli_uzantilar)]
    
    if not klasordeki_fotograflar:
        print("HATA: Bu klasörde hiç .jpg veya .png uzantılı fotoğraf bulunamadı!")
        print("Lütfen test etmek istediğiniz fotoğrafı bu klasörün içine atın ve tekrar deneyin.")
    else:
        # Klasördeki İLK fotoğrafı otomatik olarak seç
        secilen_fotograf = klasordeki_fotograflar[0]
        print(f"Otomatik bulunan fotoğraf: {secilen_fotograf} analize alınıyor...")
        predict(secilen_fotograf)