import os 
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import duckdb
from sklearn.decomposition import PCA


def load_images_from_folder(folder_path, category_name):
    """
    Dodano parametr category_name, aby przypisywać kategorię do każdego załadowanego zdjęcia.
    """
    images = []
    filenames = []
    categories_list = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = Path(folder_path) / filename
            preprocessed_image = preprocess_image(image_path)
            print(f"Załadowano i przetworzono: {filename} (kategoria: {category_name})")
            
            if preprocessed_image is not None:
                images.append(preprocessed_image)
                filenames.append(filename)
                categories_list.append(category_name)

    return np.array(images), filenames, categories_list

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Wczytuje obraz, konwertuje do odcieni szarości, aplikuje CLAHE,
    zmienia rozmiar oraz normalizuje do zakresu [0, 1].
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"[BŁĄD] Nie można wczytać obrazu: {image_path}")
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    img_resized = cv2.resize(img_clahe, target_size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized.flatten()

categories = ["NORMAL", "PNEUMONIA"]

all_photos = []
all_filenames = []
all_categories = []

for category in categories:
    folder_path = "./data/train/" + category + "/"

    print("Ładowanie zdjęć z folderu " + folder_path)
    loaded_images, filenames, cats = load_images_from_folder(folder_path, category)
    filecount = len(loaded_images)
    print(f"Załadowano {filecount} zdjęć z kategorii {category}")
    
    all_photos.extend(loaded_images)
    all_filenames.extend(filenames)
    all_categories.extend(cats)

print(f"Ładowanie zdjęć zakończone. Łączna liczba zdjęć: {len(all_photos)}")

n_components = min(50, len(all_photos)) 
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(all_photos)

con = duckdb.connect("photos.db")

# ZMIANA: Zamiast BLOB używamy typu FLOAT[]
con.execute("""
    CREATE TABLE IF NOT EXISTS photos (
        filename VARCHAR,
        category VARCHAR,
        pca_features FLOAT[]
    )
""")

con.execute("DELETE FROM photos")

dane_do_bazy = []
for i in range(len(all_filenames)):
    dane_do_bazy.append((all_filenames[i], all_categories[i], X_pca[i].tolist()))

con.executemany("INSERT INTO photos VALUES (?, ?, ?)", dane_do_bazy)
print("Dane zostały pomyślnie zapisane do bazy photos.db!")

print("\nPrzykładowy odczyt z bazy (jako Pandas DataFrame):")
df_odczyt = con.execute("SELECT * FROM photos LIMIT 3").df()
print(df_odczyt)

con.close()