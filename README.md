# 🏦 Banka Müşteri Kayıp (Churn) Analiz Paneli

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern_API-05998b.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)

Bu proje, banka şube müdürlerinin ve portföy yöneticilerinin müşteri kaybını (churn) önceden tahmin etmeleri için geliştirilmiş **uçtan uca (end-to-end)** bir karar destek sistemidir. Sadece tahmin yapmakla kalmaz, tahminin nedenini açıklar ve çözüm yolları simüle eder.

---

## 🚀 Canlı Uygulama
Uygulamayı tarayıcınızda hemen deneyin:
🔗 **[https://bank-churn-prediction-eaf.streamlit.app/]**

> **🔐 Giriş Bilgileri:**
> * **Kullanıcı Adı:** `admin`
> * **Şifre:** `123456` 

---

## ✨ Temel Özellikler

| Özellik | Açıklama |
| :--- | :--- |
| **🔍 Akıllı Tahmin** | Random Forest algoritması ile %86+ doğruluk oranı. |
| **💡 Neden Analizi (SHAP)** | Modelin kararlarını (Yaş, Bakiye, Aktiflik vb.) şeffaf şekilde açıklar. |
| **🧪 What-If Simülatörü** | Müşteriye uygulanacak aksiyonların (Kredi kartı verme vb.) risk üzerindeki etkisini ölçer. |
| **💰 Finansal Etki (CLTV)** | Ayrılma riski durumunda bankanın uğrayacağı **Beklenen Kayıp (€)** tutarını hesaplar. |
| **📂 Toplu Analiz** | CSV dosyası yükleyerek tüm portföyü finansal önceliğe göre sıralar. |
| **🔒 Güvenli Giriş** | Streamlit Secrets altyapısı ile şifrelenmiş kurumsal giriş ekranı. |

---

## 🛠️ Teknoloji Yığını (Tech Stack)

* **Model:** Scikit-Learn (Random Forest)
* **API:** FastAPI (Dockerized)
* **Arayüz:** Streamlit
* **Açıklanabilirlik:** SHAP (SHapley Additive exPlanations)
* **Görselleştirme:** Plotly & Graph Objects
* **Dağıtım:** Docker, GitHub, Streamlit Cloud

---

## 📦 Kurulum ve Çalıştırma

### 1. Docker ile API Sunucusunu Başlatma
docker build -t churn-api .
docker run -d -p 8000:8000 churn-api

### 2. Dashboard'u Yerel Olarak Çalıştırma
pip install -r requirements.txt
streamlit run dashboard.py

## 👨‍💻 Mühendislik Yaklaşımı
Bu proje bir bitirme tezi kapsamında geliştirilmiş olup şu mühendislik disiplinlerini içerir:

Mikroservis Mimarisi: FastAPI ve Docker ile API-Arayüz ayrımı.

XAI (Açıklanabilir Yapay Zeka): Kara kutu modellerin (Black-box) karar mekanizmalarını görselleştirme.

Güvenli Yazılım Geliştirme: Hassas verilerin Secrets Management ile yönetilmesi.

Veri Odaklı Karar Destek: Tahminlerin finansal metriklerle (CLTV) birleştirilmesi.
