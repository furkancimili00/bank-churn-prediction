from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# 1. FastAPI uygulamasını başlatıyoruz
app = FastAPI(
    title="Banka Churn Tahmin API",
    description="Müşterilerin bankayı terk etme (churn) riskini hesaplayan XAI destekli kurumsal API",
    version="1.0.0"
)

# 2. Kaydettiğimiz modeli ve ön işleme araçlarını hafızaya yüklüyoruz.
# (API her çalıştığında sadece bir kere yüklenir, her istekte tekrar yüklenmez - Performans için kritik)
try:
    model_pack = joblib.load('churn_thesis_model.pkl')
    model = model_pack['model']
    scaler = model_pack['scaler']
    expected_features = model_pack['features']
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    model, scaler, expected_features = None, None, None


# 3. Pydantic ile Veri Doğrulama Şeması (Banka sisteminden gelecek verinin formatı)
# Banka, API'ye bir müşteri verisi gönderdiğinde tam olarak bu değişkenleri ve tipleri göndermek zorundadır.
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# 4. Endpoint: Sadece sistemin çalışıp çalışmadığını test etmek için
@app.get("/")
def health_check():
    return {"status": "success", "message": "Churn Tahmin API aktif olarak çalışıyor."}


# 5. Endpoint: Asıl tahmini yapacak olan POST isteği
@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Makine öğrenmesi modeli yüklenemedi.")

    # Gelen veriyi bir sözlüğe (dictionary), sonra da Pandas DataFrame'e çeviriyoruz
    customer_dict = customer.dict()
    df_input = pd.DataFrame([customer_dict])

    # VERİ ÖN İŞLEME (Senin notebook'ta yaptığın işlemlerin simülasyonu)
    # Kategori verilerini (Geography, Gender) One-Hot Encoding'e dönüştürme:
    df_input = pd.get_dummies(df_input, drop_first=True)

    # Modelin eğitiminde kullanılan sütun yapısı ile gelen verinin yapısını eşliyoruz.
    # Eksik dummy sütunlar varsa 0 olarak ekliyoruz.
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Sütun sırasını modelin eğitildiği sıraya diziyoruz
    df_input = df_input[expected_features]

    # Scaler ile sayısal verileri aynı eğitimdeki gibi ölçeklendiriyoruz
    scaled_input = scaler.transform(df_input)

    # TAHMİN (Prediction)
    # predict_proba ile sadece 0-1 değil, % kaç ihtimalle churn olacağını buluyoruz.
    churn_probability = model.predict_proba(scaled_input)[0][1]
    churn_prediction = int(model.predict(scaled_input)[0])

    # Riske göre kategori belirleme (Bankanın aksiyon alabilmesi için iş kuralı)
    if churn_probability >= 0.70:
        risk_level = "Çok Yüksek Riskli - Acil İletişime Geçilmeli"
    elif churn_probability >= 0.40:
        risk_level = "Orta Riskli - Kampanya Önerilebilir"
    else:
        risk_level = "Düşük Riskli - Sadık Müşteri"

    # Standart ve kurumsal bir JSON dönüşü
    return {
        "churn_tahmini": churn_prediction,
        "churn_ihtimali": round(float(churn_probability), 4),
        "risk_seviyesi": risk_level,
        "mesaj": "Tahmin başarıyla hesaplandı."
    }