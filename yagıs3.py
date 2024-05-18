import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Streamlit arayüzü
st.markdown("""
    <div style='text-align: center;'>
        <h1>Yağış Miktarı Tahmini</h1>
    </div>
""", unsafe_allow_html=True)
# Görüntüyü ekle
image_path = "D:\\phytonproject\\anaconda projects\\images\\yagmur.gif"
st.image(image_path, caption='"Predicting rain doesn\'t count, building an ark does." - Warren Buffett', use_column_width=True)

# Kullanıcıdan dosya yüklemesini iste
uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Aylara karşılık gelen sayısal değerler
    months = {
        'OCAK': 1, 'ŞUBAT': 2, 'MART': 3, 'NISAN': 4, 'MAYIS': 5,
        'HAZIRAN': 6, 'TEMMUZ': 7, 'AĞUSTOS': 8, 'EYLÜL': 9, 'EKİM': 10,
        'KASIM': 11, 'ARALIK': 12
    }

    # Aylık veriyi sayısal değerlere dönüştür
    df['AY_NUM'] = df['AY'].map(months)

    # Eksik değerleri işleme
    imputer = SimpleImputer(strategy='mean')
    df[['YAĞIŞ MİKTARI (mm)']] = imputer.fit_transform(df[['YAĞIŞ MİKTARI (mm)']])
    df[['AY_NUM']] = imputer.fit_transform(df[['AY_NUM']])
    df[['YIL']] = imputer.fit_transform(df[['YIL']])

    # 2019 yılına kadar olan verileri eğitim seti olarak kullan
    df_train = df[df['YIL'] <= 2019]
    df_test = df[(df['YIL'] > 2019) & (df['YIL'] <= 2021)]

    # Eğitim verilerini ayır
    X_train = df_train[['YIL', 'AY_NUM']]
    y_train = df_train['YAĞIŞ MİKTARI (mm)']

    # Test verilerini ayır
    X_test = df_test[['YIL', 'AY_NUM']]
    y_test = df_test['YAĞIŞ MİKTARI (mm)']

    # Random Forest modelini eğit
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test verileri üzerinde tahmin yap
    y_pred = model.predict(X_test)

    # Performans ölçümleri
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    std_dev = np.std(y_test - y_pred)
    confidence_interval = 1.96 * std_dev / np.sqrt(len(y_test))

    st.write(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")
    st.write(f"Ortalama Kare Hata (MSE): {mse:.2f}")
    st.write(f"Standart Sapma: {std_dev:.2f}")
    st.write(f"Güven Aralığı (95%): ±{confidence_interval:.2f}")

    # Gerçek ve tahmin edilen değerleri görselleştir
    plt.figure(figsize=(10, 5))
    plt.plot(df_test['YIL'].astype(str) + "-" + df_test['AY_NUM'].astype(str), y_test, label='Gerçek Değerler')
    plt.plot(df_test['YIL'].astype(str) + "-" + df_test['AY_NUM'].astype(str), y_pred, label='Tahmin Edilen Değerler')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title('Gerçek ve Tahmin Edilen Yağış Miktarları (2019-2021)')
    st.pyplot(plt)

    # Kullanıcıdan yıl ve ay seçimi al
    year = st.number_input('Yıl Seçiniz', min_value=2022, max_value=2100, value=2023)
    selected_month = st.selectbox('Ay Seçiniz', df['AY'].unique())

    # Tahmin fonksiyonu
    def predict_precipitation(year, month):
        month_num = months[month]
        prediction = model.predict(np.array([[year, month_num]]))[0]
        return max(0, prediction)  # Negatif değerleri 0 ile değiştir

    # Tahmini hesapla
    if st.button('Tahmin Et'):
        prediction = predict_precipitation(year, selected_month)
        st.write(f'{year} yılı {selected_month} ayı için tahmini yağış miktarı: {prediction:.2f} mm')

        # Geçmiş 10 yılın verilerini al
        past_10_years = range(year - 10, year)
        past_data = df[(df['YIL'].isin(past_10_years)) & (df['AY'] == selected_month)]

        # Eksik yıllar için tahmin yap
        missing_years = set(past_10_years) - set(past_data['YIL'])
        predictions = []
        for missing_year in missing_years:
            predicted_value = predict_precipitation(missing_year, selected_month)
            predictions.append({'YIL': missing_year, 'YAĞIŞ MİKTARI (mm)': predicted_value, 'TÜR': 'Tahmin'})

        past_data = past_data[['YIL', 'YAĞIŞ MİKTARI (mm)']]
        past_data['TÜR'] = 'Gerçek'
        predicted_data = pd.DataFrame(predictions)

        combined_data = pd.concat([past_data, predicted_data]).sort_values('YIL')

        # Grafiği oluştur
        plt.figure(figsize=(10, 5))
        for tür, grp in combined_data.groupby('TÜR'):
            plt.plot(grp['YIL'], grp['YAĞIŞ MİKTARI (mm)'], label=f'{tür} Veriler')

        plt.legend()
        plt.title(f'Son 10 Yılın {selected_month} Ayı Yağış Miktarları ({year - 10} - {year - 1})')
        st.pyplot(plt)
