import numpy as np
import os
from PIL import Image

# 1. VERİLERİ YÜKLE VE HAZIRLA
base_path = r'C:\Users\RANA NUR OKTAY\Desktop\PYTHON\CIFAR10\CIFAR10_Resimler'
classes = ['ucak', 'otomobil', 'kus', 'kedi', 'geyik', 'kopek', 'kurbaga', 'at', 'gemi', 'kamyon']

X_all = []
y_all = []

# Her sınıftan 200 resim alalım (5-Fold CV uzun sürdüğü için örneklem sayısını düşük tuttuk)
for idx, cls in enumerate(classes):
    cls_folder = os.path.join(base_path, cls)
    if os.path.exists(cls_folder):
        file_list = os.listdir(cls_folder)[:200]
        for file_name in file_list:
            img = Image.open(os.path.join(cls_folder, file_name)).convert('RGB')
            X_all.append(np.array(img).flatten())
            y_all.append(idx)

X_all = np.array(X_all).astype("float32")
y_all = np.array(y_all)

# Veriyi karıştır
indices = np.arange(len(X_all))
np.random.shuffle(indices)
X_all, y_all = X_all[indices], y_all[indices]

# 2. 5-FOLD CROSS VALIDATION AYARLARI
num_folds = 5
k_choices = [3, 5]  # Karşılaştırılacak k değerleri
fold_size = len(X_all) // num_folds

print(f"--- 5-Fold Cross Validation Başlatıldı (Toplam Veri: {len(X_all)}) ---\n")

# Her k değeri için sonuçları tutacak sözlük
k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []
    print(f"k = {k} için hesaplanıyor...")

    for i in range(num_folds):
        # Katmanları ayır (Validation ve Training)
        start, end = i * fold_size, (i + 1) * fold_size
        X_val_fold = X_all[start:end]
        y_val_fold = y_all[start:end]

        X_train_fold = np.concatenate([X_all[:start], X_all[end:]])
        y_train_fold = np.concatenate([y_all[:start], y_all[end:]])

        # k-NN Hesaplama (L2 Mesafesi ile)
        fold_correct = 0
        for j in range(len(X_val_fold)):
            # Öklid Mesafesi
            dists = np.sqrt(np.sum(np.square(X_train_fold - X_val_fold[j]), axis=1))
            closest_y = y_train_fold[np.argsort(dists)[:k]]
            prediction = np.argmax(np.bincount(closest_y))

            if prediction == y_val_fold[j]:
                fold_correct += 1

        acc = (fold_correct / fold_size) * 100
        k_to_accuracies[k].append(acc)
        print(f"  Fold {i + 1}: Başarı %{acc:.2f}")

# 3. HİPOTEZ KANITI: SONUÇLARI KIYASLA
print("\n--- NİHAİ ANALİZ ---")
for k in k_choices:
    mean_acc = np.mean(k_to_accuracies[k])
    std_acc = np.std(k_to_accuracies[k])
    print(f"k={k} için Ortalama Başarı: %{mean_acc:.2f} (+/- %{std_acc:.2f})")

print("\nHipotez Sonucu: Eğer standart sapma (+/-) yüksekse, CV yapmanın şart olduğu kanıtlanmış olur.")