import pickle
import numpy as np
import os
from PIL import Image

# 1. AYARLAR
data_folder = r'C:\Users\RANA NUR OKTAY\Desktop\PYTHON\CIFAR10\cifar-10-batches-py'
output_folder = r'C:\Users\RANA NUR OKTAY\Desktop\PYTHON\CIFAR10\CIFAR10_Resimler'

classes = ['ucak', 'otomobil', 'kus', 'kedi', 'geyik', 'kopek', 'kurbaga', 'at', 'gemi', 'kamyon']

# Klasörleri oluştur (Yoksa oluşturur)
for cls in classes:
    os.makedirs(os.path.join(output_folder, cls), exist_ok=True)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# İşlenecek tüm dosyaların listesi (5 eğitim + 1 test batch)
batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

print("Tüm CIFAR-10 verileri resim olarak kaydediliyor...")

for batch_name in batch_files:
    batch_path = os.path.join(data_folder, batch_name)

    if not os.path.exists(batch_path):
        print(f"Uyarı: {batch_name} bulunamadı, atlanıyor.")
        continue

    print(f"Şu an işleniyor: {batch_name}")
    data_dict = unpickle(batch_path)
    images = data_dict[b'data']
    labels = data_dict[b'labels']

    # Her batch içindeki 10.000 resmi dönüştür
    for i in range(len(images)):
        # Veriyi görsel formatına getir
        img_rgb = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img_rgb.astype('uint8'))

        # Dosya ismi çakışmaması için batch adını da ekleyelim
        label_name = classes[labels[i]]
        img_name = f"{batch_name}_resim_{i}.png"
        save_path = os.path.join(output_folder, label_name, img_name)

        img.save(save_path)

print(f"\nİşlem Tamamlandı! Toplam 60.000 resim '{output_folder}' klasörüne kaydedildi.")