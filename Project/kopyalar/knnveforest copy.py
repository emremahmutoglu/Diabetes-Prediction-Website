import pandas as pd  # Pandas kütüphanesini pd takma adı ile içe aktarma, veri işleme için kullanılır.
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test setlerine ayırmak için kullanılır.
from sklearn.preprocessing import StandardScaler  # Özellik ölçeklendirme (normalizasyon) için kullanılır.
from sklearn.neighbors import KNeighborsClassifier  # K-En Yakın Komşu sınıflandırıcı algoritmasını kullanmak için.
from sklearn.ensemble import RandomForestClassifier  # Random Forest sınıflandırıcı algoritmasını kullanmak için.
from sklearn.svm import SVC  # Destek Vektör Makineleri sınıflandırıcı algoritmasını kullanmak için.

# Verileri yükleme
data = pd.read_csv("diabetes.csv")  # 'diabetes.csv' adlı dosyadan verileri yükler.

# Verileri ayırma
X = data.drop('Sonuc', axis=1)  # 'Sonuc' sütunu hariç tüm verileri özellik matrisi olarak ayarlar.
y = data['Sonuc'].values  # 'Sonuc' sütununu hedef vektör olarak ayarlar.

# Eğitim ve test kümelerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Veriyi %20 test, %80 eğitim olarak ayırır.

# Verileri ölçeklendirme
scaler = StandardScaler()  # StandardScaler örneği oluşturur.
X_train_scaled = scaler.fit_transform(X_train)  # Eğitim verilerini ölçeklendirir ve scaler'ı eğitir.
X_test_scaled = scaler.transform(X_test)  # Test verilerini ölçeklendirir.

# KNN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)  # 5 komşu ile KNN sınıflandırıcı oluşturur.
knn.fit(X_train_scaled, y_train)  # Ölçeklendirilmiş eğitim verileri ile modeli eğitir.

# Random Forest modelini oluşturma ve eğitme
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 ağaç ile Random Forest sınıflandırıcı oluşturur.
rf.fit(X_train_scaled, y_train)  # Ölçeklendirilmiş eğitim verileri ile modeli eğitir.

# SVM modelini oluşturma ve eğitme
svm = SVC(kernel='linear')  # Lineer çekirdek ile SVM sınıflandırıcı oluşturur.
svm.fit(X_train_scaled, y_train)  # Ölçeklendirilmiş eğitim verileri ile modeli eğitir.

# Kullanıcıdan veri alma
gebelik = int(input("Gebelik sayısını girin: "))
glukoz = int(input("Glukoz değerini girin: "))
tansiyon = int(input("Tansiyon değerini girin: "))
deri_kalinligi = int(input("Deri Kalınlığı değerini girin: "))
insulin = int(input("İnsülin değerini girin: "))
bmi = float(input("Vücut Kitle Endeksi değerini girin: "))
diyabet_pedigree = float(input("Diyabet Soy Ağacı Fonksiyon değerini girin: "))
yas = int(input("Yaş değerini girin: "))  # Kullanıcıdan sağlık verileri alır.

# Kullanıcının girdileriyle veri oluşturma
yeni_veri = pd.DataFrame([[gebelik, glukoz, tansiyon, deri_kalinligi, insulin, bmi, diyabet_pedigree, yas]],
                         columns=X.columns)  # Kullanıcıdan alınan verileri DataFrame'e dönüştürür.

# Tahmin için ölçeklendirme
yeni_veri_scaled = scaler.transform(yeni_veri)  # Kullanıcıdan alınan verileri ölçeklendirir.

# Modellerden tahmin yapma
tahmin_knn = knn.predict(yeni_veri_scaled)[0]  # KNN modeli ile tahmin yapar.
tahmin_rf = rf.predict(yeni_veri_scaled)[0]  # Random Forest modeli ile tahmin yapar.
tahmin_svm = svm.predict(yeni_veri_scaled)[0]  # SVM modeli ile tahmin yapar.

# Tahminleri yazdırma
print(f"KNN Tahmini: {'Var' if tahmin_knn == 1 else 'Yok'}")
print(f"Random Forest Tahmini: {'Var' if tahmin_rf == 1 else 'Yok'}")
print(f"SVM Tahmini: {'Var' if tahmin_svm == 1 else 'Yok'}") 


# 3,150,90,35,200,33.5,1.2,50