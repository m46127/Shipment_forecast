import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import base64

# ファイルをアップロードする
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    # アップロードされたファイルを読み込む
    data = pd.read_csv(uploaded_file)

    # 'date'列を日付型に変換する
    data['date'] = pd.to_datetime(data['date'], format='%Y%m')

    # 日付をエポック秒に変換する
    data['date_epoch'] = data['date'].astype(int) // 10**9

    # 欠損値を補完する
    data['shipment'].fillna(0, inplace=True)

    # 線形回帰モデルを作成して訓練する
    model = LinearRegression()
    model.fit(data[['date_epoch']], data['shipment'])

    # 予測対象の日付を指定する
    prediction_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')

    # 予測対象の日付をエポック秒に変換する
    prediction_dates_epoch = (prediction_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    # 予測を実行する
    predictions = model.predict(prediction_dates_epoch.to_frame())

    # 予測結果を小数点以下切り捨てる
    predictions = np.floor(predictions)

    # 結果を表示する
    result = pd.DataFrame({'date': prediction_dates, 'shipment': predictions})
    st.write(result)

    # 結果をCSVファイルとしてダウンロードする
    csv = result.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
