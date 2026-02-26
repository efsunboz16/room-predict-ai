# 🏠 Real Estate X-Ray: Room Classification AI (room-predict-ai)

Bu proje, emlak ilanlarındaki fotoğrafları analiz ederek odayı 10 farklı kategoriden birine otomatik olarak sınıflandıran derin öğrenme (Deep Learning) tabanlı bir yapay zeka modülüdür. 

Geliştirilmekte olan bir "Emlak SaaS" (Real Estate X-Ray) platformunun görüntü işleme motoru olarak tasarlanmıştır. Özellikle Türkiye (Bursa vb.) emlak piyasasındaki karmaşık, loş, dar açılı veya standart dışı ilan fotoğraflarını yüksek doğrulukla analiz edebilmesi için özel anti-overfitting stratejileriyle eğitilmiştir.

## 🚀 Proje Özeti ve Başarı Oranı
* **Mimari:** Transfer Learning (Google ResNet-18)
* **Eğitim Başarısı (Train Accuracy):** ~%99.8
* **Doğrulama Başarısı (Validation Accuracy):** ~%84.0 (İnsan seviyesi / Human-Level Performance)
* **Kullanılan Çerçeve (Framework):** PyTorch

## 📂 Veri Seti ve Sınıflar
Model, emlak fotoğraflarını aşağıdaki 10 sınıfa ayırmak üzere eğitilmiştir:
`bathroom`, `bedroom`, `dining`, `gaming`, `kitchen`, `laundry`, `living`, `office`, `terrace`, `yard`

## 🧠 Geliştirme Süreci ve Mühendislik Çözümleri
Eğitim aşamasında, 11 milyon parametreli ResNet-18 modelinin verileri ezberlemesini (Overfitting) engellemek ve gerçek dünya verilerinde (Validation) yüksek performans almasını sağlamak için ağır cezalandırma ve veri manipülasyonu teknikleri kullanılmıştır:

1.  **Agresif Veri Çoğaltma (Data Augmentation):**
    * `RandomCrop(224)` ve `Resize(256)` kombinasyonu ile odak noktası sürekli değiştirildi.
    * `RandomAffine` (Açı ve yakınlaştırma) ve `ColorJitter` (Parlaklık/Kontrast manipülasyonu) uygulandı.
2.  **Random Erasing (Rastgele Silme):**
    * Görsellerin %20'lik kısımları rastgele siyah kutularla kapatılarak (Sansürlenerek), modelin pikselleri ezberlemesi engellendi ve odanın genel yapısal mimarisini öğrenmeye zorlandı.
3.  **Ağır L2 Regularization & Dropout:**
    * Tam Bağlantılı Katmandaki (Fully Connected) `Dropout` oranı %50'den **%60**'a çıkarıldı.
    * `AdamW` optimizasyon algoritmasındaki `weight_decay` değeri **1e-2**'ye yükseltilerek modelin uçuk ağırlıklar ataması engellendi.
4.  **Dinamik Öğrenme Hızı (Cosine Annealing LR):**
    * Eğitim başındaki yüksek öğrenme hızı, epoch'lar ilerledikçe Kosinüs eğrisi formunda kademeli olarak düşürülerek yerel minimumlarda (local minima) mükemmel ince ayar (fine-tuning) yapılması sağlandı.

## ⚙️ Kurulum ve Gereksinimler

Projenin kendi izole ortamında çalışabilmesi için aşağıdaki adımları izleyin. Sisteminizde CUDA destekli bir GPU (örn. RTX 50 serisi) varsa GPU sürümünü, yoksa standart CPU sürümünü kurabilirsiniz.

**1. Virtual Environment (Sanal Ortam) Oluşturma ve Aktifleştirme:**
```bash
python -m venv venv

# Windows için:
.\venv\Scripts\activate

# Mac/Linux için:
source venv/bin/activate
