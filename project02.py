"""執行方式：streamlit run project02.py。"""
import pandas as pd
import streamlit as st
from project02_core import load_data, classification_data, regression_data, train_classifier, train_regressor

st.set_page_config(page_title='客戶與訂單預測', layout='wide')
st.title('客戶與訂單預測專案')

@st.cache_data
def get_data():
    return load_data()

try:
    customers, orders, sales, order_data = get_data()
except (OSError, ValueError, KeyError) as exc:
    st.error(f'資料載入失敗：{exc}')
    st.stop()

page = st.sidebar.radio('頁面', ['資料概覽', 'VIP 分類', '訂單金額預測'])
if st.sidebar.button('重新載入資料'):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

if page == '資料概覽':
    st.header('資料狀態與視覺化')
    a, b, c = st.columns(3)
    a.metric('客戶數', len(customers))
    b.metric('訂單數', len(orders))
    c.metric('整合明細數', len(sales))
    st.caption('金額＝數量 × 明細單價 ×（1 − 折扣率）；營收僅統計已完成訂單。')
    st.subheader('缺失值統計')
    st.dataframe(sales.isna().sum().rename('缺失數').to_frame(), use_container_width=True)
    completed = sales[sales.status.eq('completed')].copy()
    st.subheader('每月營收')
    completed['month'] = completed.order_date.dt.to_period('M').astype(str)
    st.line_chart(completed.groupby('month').amount.sum())
    left, right = st.columns(2)
    with left:
        st.subheader('商品類別營收')
        st.bar_chart(completed.groupby('category').amount.sum())
    with right:
        st.subheader('客戶分群人數')
        st.bar_chart(customers.segment.value_counts())
    st.subheader('整合資料預覽（前 500 筆）')
    st.dataframe(sales.head(500), use_container_width=True)
    st.download_button('下載整合資料 CSV', sales.to_csv(index=False).encode('utf-8-sig'), 'integrated_sales.csv', 'text/csv')
else:
    classification = page == 'VIP 分類'
    st.header(page)
    if classification:
        st.warning('目前資料只有靜態 VIP 標籤，無升級日期。模型估計目前 VIP 身分的機率，不能驗證未來升級機率；機率未經校準。')
        frame, features = classification_data(customers, order_data)
        choices = ['羅吉斯迴歸', '隨機森林']
        st.caption('依客戶分層隨機切分：75% 訓練、25% 測試。segment 僅作標籤，不作特徵。')
    else:
        frame, features = regression_data(order_data)
        choices = ['隨機森林', 'XGBoost']
        st.caption('僅使用已完成訂單，較早 80% 日期訓練、較晚日期測試。同日訂單不互相使用金額；測試歷史按日期滾動更新。')
    name = st.selectbox('模型', choices)
    trees = st.slider('樹的數量（樹模型使用）', 50, 300, 150, 50)
    key = f'{page}-{name}-{trees}'
    if st.button('訓練模型', type='primary'):
        try:
            with st.spinner('訓練與評估中…'):
                trainer = train_classifier if classification else train_regressor
                st.session_state[key] = trainer(frame, features, name, trees)
        except (ValueError, ImportError) as exc:
            st.error(f'無法訓練：{exc}')
    if key in st.session_state:
        model, metrics, results, train_count = st.session_state[key]
        st.caption(f'訓練筆數：{train_count:,}；測試筆數：{len(results):,}；固定亂數種子 42。')
        st.dataframe(pd.Series(metrics, name='測試集指標').to_frame())
        if classification:
            st.subheader('混淆矩陣（測試集）')
            st.dataframe(pd.crosstab(results['實際 VIP'], results['預測 VIP']).reindex(index=[0, 1], columns=[0, 1], fill_value=0))
        else:
            st.scatter_chart(results, x='order_amount', y='predicted_amount')
        st.dataframe(results.head(200), use_container_width=True)
        st.download_button('下載測試預測結果', results.to_csv(index=False).encode('utf-8-sig'), 'predictions.csv', 'text/csv')
        st.subheader('輸入特徵進行預測')
        st.caption('預設值取自資料中位數或眾數，可自行修改。歷史消費統計應只包含預測時已知的資料。')
        with st.form('prediction'):
            values = {}
            for feature in features:
                column = frame[feature]
                if pd.api.types.is_numeric_dtype(column):
                    default = float(column.median()) if column.notna().any() else 0.0
                    values[feature] = st.number_input(feature, min_value=0.0, value=max(0.0, default))
                else:
                    values[feature] = st.selectbox(feature, sorted(column.dropna().unique().tolist()))
            submit = st.form_submit_button('預測')
        if submit:
            row = pd.DataFrame([values])[features]
            if classification:
                st.metric('估計 VIP 機率（目前身分）', f'{model.predict_proba(row)[0, 1]:.1%}')
            else:
                st.metric('預測訂單金額', f'{max(0, model.predict(row)[0]):,.2f}')
    else:
        st.info('選擇模型後按「訓練模型」，即可查看評估結果與進行預測。')
