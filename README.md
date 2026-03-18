🏦 Banka Müşteri Kayıp (Churn) Analiz Sistemi
Bu proje, banka şube müdürlerinin ve portföy yöneticilerinin müşterilerin bankayı terk etme (churn) riskini tahmin etmeleri, bu riskin nedenlerini anlamaları ve finansal kaybı minimize edecek aksiyonlar almaları için geliştirilmiş uçtan uca (end-to-end) bir karar destek sistemidir.

🚀 Canlı Demo
Uygulamayı tarayıcınızda hemen deneyin:
🔗 [https://bank-churn-prediction-eaf.streamlit.app/]
(Giriş Bilgileri: Kullanıcı: admin | Şifre: 123456)

✨ Temel Özellikler
🔍 Akıllı Tahminleme: Random Forest algoritması kullanarak müşterilerin ayrılma olasılığını %86+ doğrulukla tahmin eder.

💡 Açıklanabilir AI (SHAP): Modelin neden "Riskli" dediğini şeffaf bir şekilde gösterir. Hangi faktörün (yaş, bakiye, aktiflik vb.) riski ne kadar artırdığını görselleştirir.

🧪 What-If (Aksiyon) Simülatörü: "Müşteriye kredi kartı verirsek risk ne kadar düşer?" gibi senaryoları canlı olarak test etmenizi sağlar.

💰 Finansal Etki Analizi (CLTV): Müşterinin banka için değerini ve ayrılması durumunda oluşacak Beklenen Kayıp (€) tutarını hesaplar.

📂 Toplu Analiz & Önceliklendirme: Yüzlerce müşteriyi içeren CSV listelerini analiz eder ve şube müdürüne en çok para kaybettirecek müşteriden başlayarak bir arama listesi sunar.

🔒 Kurumsal Güvenlik: Şifre korumalı giriş ekranı ve güvenli "Secrets" yönetimi.

🛠️ Teknoloji Yığını (Tech Stack)
Model: Scikit-Learn (Random Forest Classifier)

API: FastAPI (Dockerized)

Frontend: Streamlit

Explainability: SHAP (SHapley Additive exPlanations)

Visuals: Plotly & Graph Objects

DevOps: Docker, GitHub Actions, Streamlit Cloud

📦 Kurulum ve Çalıştırma

Docker ile Çalıştırma (API)
Sistemin motorunu (FastAPI) izole bir konteynerde çalıştırmak için:
Bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api

Dashboard'u Çalıştırma
Bash
pip install -r requirements.txt
streamlit run dashboard.py

📊 Veri Seti Hakkında
Projede kullanılan veri seti, 10.000 banka müşterisinin demografik ve finansal bilgilerini içermektedir. Model eğitimi sırasında Geography, Gender gibi kategorik veriler One-Hot Encoding ile, sayısal veriler ise StandardScaler ile işlenmiştir.

👨‍💻 Yazar
[Enis Çelik]
4. Sınıf Bilgisayar Mühendisliği Öğrencisi
[https://www.linkedin.com/in/eniscelik16/]
[enisccelik@gmail.com]
