import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_object_and_merge_with_grabcut(foreground_path, background_path, rect=None):
    # Baca gambar objek (foreground) dan background
    foreground = cv2.imread(foreground_path)
    background = cv2.imread(background_path)

    # Pastikan ukuran background sama dengan foreground
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    # Jika rect belum diberikan, kita gunakan seluruh gambar
    if rect is None:
        rect = (10, 10, foreground.shape[1] - 30, foreground.shape[0] - 30)  # (x, y, width, height)

    # Inisialisasi mask dan model untuk GrabCut
    mask = np.zeros(foreground.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Terapkan GrabCut untuk memisahkan objek dari latar belakang
    cv2.grabCut(foreground, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Buat mask biner untuk objek
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    obj_only = foreground * mask2[:, :, np.newaxis]

    # Buat versi putih dari objek hanya untuk tampilan
    obj_white = np.zeros_like(foreground)
    obj_white[mask2 == 1] = [255, 255, 255]  # Set pixels to white where mask is 1

    # Ekstrak bagian background di gambar kedua yang akan ditempati objek
    mask_inv = cv2.bitwise_not(mask2 * 255)
    background_only = cv2.bitwise_and(background, background, mask=mask_inv)

    # Gabungkan objek asli dengan background
    combined_image = cv2.add(background_only, obj_only)

    # Tampilkan hasil menggunakan matplotlib
    plt.figure(figsize=(15, 5))

    # Tampilkan gambar asli
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Tampilkan objek yang diekstrak dalam warna putih
    plt.subplot(1, 3, 2)
    plt.title("Extracted")
    plt.imshow(cv2.cvtColor(obj_white, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Tampilkan hasil akhir gabungan dengan background
    plt.subplot(1, 3, 3)
    plt.title("Output")
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
    
    # Simpan hasil akhir
    cv2.imwrite("combined_output_grabcut.png", combined_image)

# Path ke gambar foreground dan background
foreground_path = "goat3.jpg"
background_path = "grass-bg.jpg"

# Panggil fungsi dengan menggunakan GrabCut
extract_object_and_merge_with_grabcut(foreground_path, background_path)