import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
import numpy as np
import os

# Configuration
st.set_page_config(page_title="Banka Churn Risk Paneli", page_icon="🏦", layout="wide")

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ==========================================
# 1. OTURUM (SESSION) YÖNETİMİ
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'current_customer' not in st.session_state:
    st.session_state.current_customer = None
if 'base_risk' not in st.session_state:
    st.session_state.base_risk = None


# ==========================================
# 2. API REQUEST FUNCTION
# ==========================================
def make_prediction_api(data_dict):
    """
    Decoupled prediction mechanism for a single row.
    Sends a POST request to the Flask API.
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=data_dict, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Hatası: {e}")
        return None

def make_batch_prediction_api(data_list):
    """
    Decoupled prediction mechanism for multiple rows.
    """
    try:
        response = requests.post(f"{API_URL}/predict_batch", json=data_list, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Toplu API Hatası: {e}")
        return None

# ==========================================
# 3. GİRİŞ (LOGIN) EKRANI
# ==========================================
def login_screen():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🏦</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Kurumsal Yönetici Girişi</h3>", unsafe_allow_html=True)
        st.write("---")
        username = st.text_input("Kullanıcı Adı")
        password = st.text_input("Şifre", type="password")
        if st.button("Sisteme Giriş Yap", use_container_width=True):
            try:
                # Use secrets exclusively for security
                secret_user = st.secrets["auth"]["username"]
                secret_pass = st.secrets["auth"]["password"]
                if username == secret_user and password == secret_pass:
                    st.session_state.logged_in = True
                    st.success("Giriş Başarılı!")
                    st.rerun()
                else:
                    st.error("❌ Hatalı kullanıcı adı veya şifre!")
            except Exception as e:
                # If no secrets exist, inform the user how to set them up
                st.error("Giriş bilgileri yapılandırılamadı. Lütfen .streamlit/secrets.toml dosyasını kontrol edin.")

# ==========================================
# 4. ANA PANEL (DASHBOARD)
# ==========================================
def main_dashboard():
    st.sidebar.title("Yönetici Menüsü")
    st.sidebar.info("Hoş Geldiniz, **Şube Müdürü**")
    if st.sidebar.button("🚪 Güvenli Çıkış Yap"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🏦 Şube Müdürü Müşteri Risk Analiz Paneli")
    tab1, tab2, tab3 = st.tabs(["👤 Tekil Müşteri Analizi", "🧪 What-If Simülatörü", "📂 Toplu Analiz (CLTV Öncelikli)"])

    # --- SEKME 1: TEKİL MÜŞTERİ ANALİZİ ---
    with tab1:
        st.subheader("Müşteri Parametreleri")
        col_form1, col_form2, col_form3 = st.columns(3)
        with col_form1:
            credit_score = st.number_input("Kredi Notu", min_value=300, max_value=850, value=650)
            age = st.number_input("Yaş", min_value=18, max_value=100, value=40)
            tenure = st.number_input("Müşterilik Süresi (Yıl)", min_value=0, max_value=10, value=5)
        with col_form2:
            balance = st.number_input("Hesap Bakiyesi (€)", min_value=0.0, value=50000.0, step=1000.0)
            est_salary = st.number_input("Tahmini Maaş (€)", min_value=0.0, value=60000.0, step=1000.0)
            num_products = st.selectbox("Kullanılan Ürün Sayısı", [1, 2, 3, 4], index=1)
        with col_form3:
            geography = st.selectbox("Ülke", ["France", "Germany", "Spain"])
            gender = st.selectbox("Cinsiyet", ["Male", "Female"])
            has_cr_card = st.selectbox("Kredi Kartı Var mı?", [1, 0], format_func=lambda x: "Evet" if x == 1 else "Hayır")
            is_active = st.selectbox("Aktif Müşteri mi?", [1, 0], format_func=lambda x: "Evet" if x == 1 else "Hayır")

        if st.button("🔍 Risk Analizi Yap", use_container_width=True):
            customer_data = {
                "CreditScore": credit_score, "Geography": geography, "Gender": gender, "Age": age,
                "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
                "HasCrCard": has_cr_card, "IsActiveMember": is_active, "EstimatedSalary": est_salary
            }

            with st.spinner("Yapay Zeka Hesaplarken Lütfen Bekleyin..."):
                # REST API İSTEĞİ
                result = make_prediction_api(customer_data)

                if result and "error" not in result:
                    churn_probability = result["churn_ihtimali"] * 100
                    st.session_state.current_customer = customer_data
                    st.session_state.base_risk = churn_probability

                    customer_value = balance + (est_salary * 0.20)
                    expected_loss = customer_value * (churn_probability / 100)

                    st.write("---")
                    st.subheader("💰 Finansal Etki Analizi (CLTV)")
                    fin_col1, fin_col2, fin_col3 = st.columns(3)
                    fin_col1.metric(label="Müşterinin Bankaya Değeri", value=f"€{customer_value:,.2f}")
                    fin_col2.metric(label="Ayrılma İhtimali", value=f"%{churn_probability:.1f}")
                    fin_col3.metric(label="Beklenen Finansal Kayıp", value=f"€{expected_loss:,.2f}", delta="- Risk Tutarı", delta_color="inverse")

                    st.write("---")
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.subheader("📊 Model Çıktısı")
                        st.metric(label="Risk Kategorisi", value=result["risk_seviyesi"])

                        # SHAP ANALİZİ (API'den dönen verilerle çizilir)
                        if "shap_degerleri" in result:
                            st.markdown("### 💡 Neden Analizi (SHAP)")
                            shap_dict = result["shap_degerleri"]

                            # Sort for visualization
                            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]))
                            features = [k for k, v in sorted_shap]
                            values = [v for k, v in sorted_shap]
                            colors = ['salmon' if float(val) > 0 else 'lightgreen' for val in values]

                            fig_shap = go.Figure(go.Bar(x=values, y=features, orientation='h', marker_color=colors))
                            fig_shap.update_layout(xaxis_title="<- Riski Düşürenler | Riski Artıranlar ->", margin=dict(l=0, r=0, t=0, b=0), height=300)
                            st.plotly_chart(fig_shap, use_container_width=True)

                    with res_col2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=churn_probability,
                            title={'text': "Ayrılma İhtimali (%)", 'font': {'size': 24}},
                            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "black"},
                                   'steps': [{'range': [0, 40], 'color': "lightgreen"},
                                             {'range': [40, 70], 'color': "gold"},
                                             {'range': [70, 100], 'color': "salmon"}]}))
                        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- SEKME 2: WHAT-IF SİMÜLATÖRÜ ---
    with tab2:
        if st.session_state.current_customer is not None:
            cust = st.session_state.current_customer
            base_risk = st.session_state.base_risk
            st.metric(label="Mevcut Durumdaki Ayrılma Riski", value=f"%{base_risk:.1f}")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                new_balance = st.number_input("Yeni Hesap Bakiyesi (€)", value=float(cust["Balance"]), step=1000.0, key="sim_bal")
                new_active = st.selectbox("Müşteriyi Aktif Hale Getir?", [1, 0], index=0 if cust["IsActiveMember"] == 1 else 1, key="sim_act")
            with sim_col2:
                new_crcard = st.selectbox("Kredi Kartı Kampanyası Tanımla?", [1, 0], index=0 if cust["HasCrCard"] == 1 else 1, key="sim_cr")
                new_products = st.slider("Ürün Sayısını Değiştir", 1, 4, value=cust["NumOfProducts"], key="sim_prod")

            if st.button("🔄 Değişim Senaryosunu Simüle Et", type="primary"):
                sim_data = cust.copy()
                sim_data.update({"Balance": new_balance, "IsActiveMember": new_active, "HasCrCard": new_crcard, "NumOfProducts": new_products})

                # API İSTEĞİ (Simülasyon)
                res_sim = make_prediction_api(sim_data)
                if res_sim and "error" not in res_sim:
                    new_risk = res_sim["churn_ihtimali"] * 100
                    diff = new_risk - base_risk
                    if diff < 0: st.metric(label="Yeni Risk", value=f"%{new_risk:.1f}", delta=f"{diff:.1f} Puan", delta_color="normal")
                    else: st.metric(label="Yeni Risk", value=f"%{new_risk:.1f}", delta=f"+{diff:.1f} Puan", delta_color="inverse")
        else:
            st.warning("Lütfen önce 'Tekil Müşteri Analizi' sekmesinden bir analiz yapın.")

    # --- SEKME 3: TOPLU MÜŞTERİ YÜKLEME ---
    with tab3:
        st.subheader("📁 Finansal Odaklı Toplu Analiz")
        uploaded_file = st.file_uploader("Dosya Seçin", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if st.button("🚀 Tüm Listeyi Analiz Et ve Önceliklendir", use_container_width=True):
                with st.spinner("Tüm müşteri portföyü analiz ediliyor..."):
                    # Veriyi liste formatına çevir
                    customers_list = df.to_dict(orient="records")

                    # TEK BİR TOPLU API İSTEĞİ
                    res_batch = make_batch_prediction_api(customers_list)

                    if res_batch and res_batch.get("status") == "success":
                        results_df = pd.DataFrame(res_batch["results"]).sort_values(by="Beklenen Kayıp (€)", ascending=False).reset_index(drop=True)
                        def color_risk(val): return f"color: {'red' if 'Yüksek' in str(val) else 'orange' if 'Orta' in str(val) else 'green'}"
                        styled_df = results_df.style.map(color_risk, subset=['Risk Seviyesi']).format({"Müşteri Değeri (€)": "{:,.2f}", "Beklenen Kayıp (€)": "{:,.2f}"})
                        st.success("✅ Analiz tamamlandı!")
                        st.dataframe(styled_df, use_container_width=True)
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Analiz Raporunu İndir", data=csv, file_name='finansal_churn_raporu.csv', mime='text/csv')
                    else:
                        st.error("Toplu analiz başarısız oldu.")

# --- AKIŞ KONTROLÜ ---
if not st.session_state.logged_in:
    login_screen()
else:
    main_dashboard()
