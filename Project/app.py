from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# CSV verilerini yükleme ve işleme
data = pd.read_csv("data/diabetes.csv")
X = data.drop('Sonuc', axis=1)
y = data['Sonuc'].values

# Eğitim ve test kümelerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelleri eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/soru1', methods=['GET', 'POST'])
def soru1():
    return render_template('soru1.html')

@app.route('/soru2', methods=['POST'])
def soru2():
    session['gebelik'] = request.form['gebelik']
    return render_template('soru2.html')

@app.route('/soru3', methods=['POST'])
def soru3():
    session['glukoz'] = request.form['glukoz']
    return render_template('soru3.html')

@app.route('/soru4', methods=['POST'])
def soru4():
    session['tansiyon'] = request.form['tansiyon']
    return render_template('soru4.html')

@app.route('/soru5', methods=['POST'])
def soru5():
    session['deri_kalinligi'] = request.form['deri_kalinligi']
    return render_template('soru5.html')

@app.route('/soru6', methods=['POST'])
def soru6():
    session['insulin'] = request.form['insulin']
    return render_template('soru6.html')

@app.route('/soru7', methods=['POST'])
def soru7():
    session['diyabet_pedigree'] = request.form['diyabet_pedigree']
    return render_template('soru7.html')

@app.route('/soru8', methods=['POST'])
def soru8():
    session['bmi'] = request.form['bmi']
    return render_template('soru8.html')

@app.route('/predict', methods=['POST'])
def predict():
    session['yas'] = request.form['yas']
    
    # Kullanıcıdan gelen tüm verileri alma
    gebelik = int(session.get('gebelik'))
    glukoz = int(session.get('glukoz'))
    tansiyon = int(session.get('tansiyon'))
    deri_kalinligi = int(session.get('deri_kalinligi'))
    insulin = int(session.get('insulin'))
    bmi = float(session.get('bmi'))
    diyabet_pedigree = float(session.get('diyabet_pedigree'))
    yas = int(session.get('yas'))

    yeni_veri = pd.DataFrame([[gebelik, glukoz, tansiyon, deri_kalinligi, insulin, bmi, diyabet_pedigree, yas]], columns=X.columns)
    yeni_veri_scaled = scaler.transform(yeni_veri)

    # Modellerden tahmin yapma
    tahmin_knn = knn.predict(yeni_veri_scaled)[0]
    tahmin_rf = rf.predict(yeni_veri_scaled)[0]
    tahmin_svm = svm.predict(yeni_veri_scaled)[0]

    # Çoğunluk kararını hesaplama
    predictions = [tahmin_knn, tahmin_rf, tahmin_svm]
    result = 'Hasta' if predictions.count(1) >= 2 else 'Hasta Değil'

    # Sonuç resim dosyasını belirleme
    image_file = "yuksek.png" if result == 'Hasta' else "dusuk.png"

    # Sonucu ve resim dosyasını dönme
    return render_template('sonuc.html', image_file=image_file)

@app.route('/grafikler')
def grafikler():
    img = io.BytesIO()
    
    # Grafik 1: Tüm değişkenler arası ilişkiler
    sns.pairplot(data, hue='Sonuc', palette={0: 'green', 1: 'red'})
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Grafik 2: Glukoz ve İnsülin Değerlerine Göre Dağılım
    img = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.scatter(X[y == 0]['Glukoz'], X[y == 0]['Insulin'], color="green", label="Sağlıklı", alpha=0.5)
    plt.scatter(X[y == 1]['Glukoz'], X[y == 1]['Insulin'], color="red", label="Hasta", alpha=0.5)
    plt.title("Glukoz ve İnsülin Değerlerine Göre Dağılım")
    plt.xlabel("Glukoz")
    plt.ylabel("İnsülin")
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('grafikler.html', plot_url1=plot_url1, plot_url2=plot_url2)

@app.route('/download_pdf')
def download_pdf():
    # Grafik 1: Tüm değişkenler arası ilişkiler
    sns.pairplot(data, hue='Sonuc', palette={0: 'green', 1: 'red'})
    plt.savefig("static/grafik1.png", format='png')
    plt.close()

    # Grafik 2: Glukoz ve İnsülin Değerlerine Göre Dağılım
    plt.figure(figsize=(12, 6))
    plt.scatter(X[y == 0]['Glukoz'], X[y == 0]['Insulin'], color="green", label="Sağlıklı", alpha=0.5)
    plt.scatter(X[y == 1]['Glukoz'], X[y == 1]['Insulin'], color="red", label="Hasta", alpha=0.5)
    plt.title("Glukoz ve İnsülin Değerlerine Göre Dağılım")
    plt.xlabel("Glukoz")
    plt.ylabel("İnsülin")
    plt.legend()
    plt.savefig("static/grafik2.png", format='png')
    plt.close()

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Diyabet Tahmin Grafikleri", ln=True, align='C')
    pdf.image("static/grafik1.png", x=10, y=20, w=180)
    pdf.add_page()
    pdf.image("static/grafik2.png", x=10, y=20, w=180)
    pdf.output("static/grafikler.pdf")

    return redirect(url_for('static', filename='grafikler.pdf'))

if __name__ == '__main__':
    app.run(debug=True)
