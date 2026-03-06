import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. AYARLAR VE VERİ YÜKLEME
# ==========================================

base_path = r'C:\Users\RANA NUR OKTAY\Desktop\PYTHON\CIFAR10\CIFAR10_Resimler'
classes = ['ucak', 'otomobil', 'kus', 'kedi', 'geyik', 'kopek', 'kurbaga', 'at', 'gemi', 'kamyon']

X_all = []
y_all = []

print("Resimler klasörden okunuyor, lütfen bekleyiniz...")

for idx, cls in enumerate(classes):
    cls_folder = os.path.join(base_path, cls)
    if not os.path.exists(cls_folder):
        continue

    # Her sınıftan 500 resim alalım (Hız ve denge için)
    file_list = os.listdir(cls_folder)[:500]

    for file_name in file_list:
        img_path = os.path.join(cls_folder, file_name)
        img = Image.open(img_path).convert('RGB')
        X_all.append(np.array(img).flatten())
        y_all.append(idx)

# Listeleri numpy dizisine çevir
X_all = np.array(X_all).astype("float32")
y_all = np.array(y_all)

# --- KRİTİK ADIM: VERİYİ KARIŞTIRMA (SHUFFLE) ---
# Eğer karıştırmazsak eğitimde kedi varken testte sadece kamyon olur!
indices = np.arange(X_all.shape[0])
np.random.shuffle(indices)
X_all = X_all[indices]
y_all = y_all[indices]

# Veriyi Eğitim ve Test olarak ayır (%80 eğitim, %20 test)
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

# ==========================================
# 2. KULLANICI ETKİLEŞİMİ
# ==========================================

print(f"\nToplam Veri: {len(X_all)} (Eğitim: {len(X_train)}, Test: {len(X_test)})")
dist_choice = input("Mesafe ölçütü seçin (L1 / L2): ").upper()
k_val = int(input("k değerini giriniz: "))
num_samples = int(input("Kaç resim görselleştirilsin?: "))

# Test setinden ilk 'num_samples' kadarını alalım
X_test_show = X_test[:num_samples]
y_test_show = y_test[:num_samples]

# ==========================================
# 3. HESAPLAMA VE GÖRSELLEŞTİRME
# ==========================================

plt.figure(figsize=(15, 5))
correct_count = 0

for i in range(num_samples):
    # MESAFE HESAPLAMA
    if dist_choice == "L1":
        distances = np.sum(np.abs(X_train - X_test_show[i]), axis=1)
    else:
        distances = np.sqrt(np.sum(np.square(X_train - X_test_show[i]), axis=1))

    # k komşuyu bul ve oyla
    closest_indices = np.argsort(distances)[:k_val]
    closest_labels = y_train[closest_indices]
    prediction = np.argmax(np.bincount(closest_labels))

    true_label = y_test_show[i]
    is_correct = (prediction == true_label)
    if is_correct:
        correct_count += 1

    print(f"Örnek {i + 1}: Tahmin: {classes[prediction]}, Gerçek: {classes[true_label]}")

    # Görselleştirme
    img_display = X_test_show[i].reshape(32, 32, 3).astype("uint8")
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(img_display)
    plt.title(f"T: {classes[prediction]}\nG: {classes[true_label]}", color="green" if is_correct else "red")
    plt.axis('off')

accuracy = (correct_count / num_samples) * 100
print(f"\nSeçilen {num_samples} örnek içindeki başarı: %{accuracy}")
plt.tight_layout()
plt.show()