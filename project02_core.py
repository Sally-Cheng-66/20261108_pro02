"""專案二的資料整合與模型訓練；不依賴 Streamlit，可單獨驗證。"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent / 'data' / 'raw'


def load_data():
    """以明細成交價計算折扣後金額，避免誤用產品目錄價格。"""
    customers = pd.read_csv(DATA_DIR / 'customers.csv', parse_dates=['signup_date'])
    orders = pd.read_csv(DATA_DIR / 'orders.csv', parse_dates=['order_date'])
    items = pd.read_csv(DATA_DIR / 'order_items.csv')
    products = pd.read_csv(DATA_DIR / 'products.csv')
    for frame, key in [(customers, 'customer_id'), (orders, 'order_id'), (products, 'product_id')]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f'{key} 有缺值或重複主鍵')
    if not (orders.customer_id.isin(customers.customer_id).all() and items.order_id.isin(orders.order_id).all() and items.product_id.isin(products.product_id).all()):
        raise ValueError('資料表包含無法對應的外鍵')
    if items[['quantity', 'unit_price', 'discount_rate']].isna().any().any() or (items.quantity <= 0).any() or (items.unit_price < 0).any() or not items.discount_rate.between(0, 1).all():
        raise ValueError('商品明細含有無效數量、價格或折扣')
    items['amount'] = items.quantity * items.unit_price * (1 - items.discount_rate)
    sales = (items.merge(products[['product_id', 'category']], on='product_id', validate='many_to_one')
             .merge(orders, on='order_id', validate='many_to_one')
             .merge(customers, on='customer_id', validate='many_to_one'))
    totals = items.groupby('order_id').amount.sum().rename('order_amount')
    order_data = orders.merge(totals, on='order_id', validate='one_to_one').merge(customers, on='customer_id', validate='many_to_one')
    return customers, orders, sales, order_data


def classification_data(customers, order_data):
    completed = order_data[order_data.status.eq('completed')]
    reference = order_data.order_date.max() + pd.Timedelta(days=1)
    summary = completed.groupby('customer_id').agg(order_count=('order_id', 'count'), total_spend=('order_amount', 'sum'), average_spend=('order_amount', 'mean'), last_order=('order_date', 'max'))
    frame = customers.merge(summary, on='customer_id', how='left', validate='one_to_one')
    frame['recency_days'] = (reference - frame.last_order).dt.days
    frame['tenure_days'] = (reference - frame.signup_date).dt.days.clip(lower=0)
    for col in ['order_count', 'total_spend', 'average_spend']:
        frame[col] = frame[col].fillna(0)
    features = ['acquisition_channel', 'city', 'order_count', 'total_spend', 'average_spend', 'recency_days', 'tenure_days']
    return frame, features


def regression_data(order_data):
    """同一天的訂單彼此不可見；歷史特徵僅累計更早日期的已完成訂單。"""
    frame = order_data[order_data.status.eq('completed')].copy()
    daily = frame.groupby(['customer_id', 'order_date']).agg(day_count=('order_id', 'count'), day_spend=('order_amount', 'sum')).reset_index().sort_values(['customer_id', 'order_date'])
    daily['prior_orders'] = daily.groupby('customer_id').day_count.cumsum() - daily.day_count
    daily['prior_spend'] = daily.groupby('customer_id').day_spend.cumsum() - daily.day_spend
    frame = frame.merge(daily[['customer_id', 'order_date', 'prior_orders', 'prior_spend']], on=['customer_id', 'order_date'], validate='many_to_one')
    frame['prior_average'] = frame.prior_spend / frame.prior_orders.replace(0, np.nan)
    frame['tenure_days'] = (frame.order_date - frame.signup_date).dt.days.clip(lower=0)
    frame['month'] = frame.order_date.dt.month
    frame['weekday'] = frame.order_date.dt.dayofweek
    features = ['acquisition_channel', 'city', 'payment_type', 'prior_orders', 'prior_spend', 'prior_average', 'tenure_days', 'month', 'weekday']
    return frame.sort_values(['order_date', 'order_id']), features


def pipeline_for(X, estimator):
    categorical = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
    ])
    return Pipeline([('preprocess', preprocessing), ('model', estimator)])


def train_classifier(frame, features, name, trees=150):
    y = frame.segment.eq('vip').astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError('VIP 與非 VIP 均至少需要兩筆資料')
    X_train, X_test, y_train, y_test = train_test_split(frame[features], y, test_size=.25, random_state=42, stratify=y)
    estimator = LogisticRegression(max_iter=2000, class_weight='balanced') if name == '羅吉斯迴歸' else RandomForestClassifier(n_estimators=trees, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=-1)
    model = pipeline_for(X_train, estimator).fit(X_train, y_train)
    pred = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = {'Accuracy': accuracy_score(y_test, pred), 'Precision': precision_score(y_test, pred, zero_division=0), 'Recall': recall_score(y_test, pred, zero_division=0), 'F1': f1_score(y_test, pred, zero_division=0), 'ROC AUC': roc_auc_score(y_test, probability)}
    return model, metrics, pd.DataFrame({'實際 VIP': y_test, '預測 VIP': pred, 'VIP 機率': probability}), len(X_train)


def train_regressor(frame, features, name, trees=150):
    dates = frame.order_date.drop_duplicates().sort_values().tolist()
    if len(dates) < 3:
        raise ValueError('至少需要三個不同日期才能依時間切分')
    cutoff = dates[min(len(dates) - 1, int(len(dates) * .8))]
    train, test = frame[frame.order_date < cutoff], frame[frame.order_date >= cutoff]
    if len(train) < 2 or len(test) < 2:
        raise ValueError('訓練集與測試集均至少需要兩筆訂單')
    if name == 'XGBoost':
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=trees, max_depth=4, learning_rate=.05, objective='reg:squarederror', random_state=42, n_jobs=2)
    else:
        estimator = RandomForestRegressor(n_estimators=trees, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model = pipeline_for(train[features], estimator).fit(train[features], train.order_amount)
    predicted = model.predict(test[features]).clip(0)
    metrics = {'MAE': mean_absolute_error(test.order_amount, predicted), 'RMSE': np.sqrt(mean_squared_error(test.order_amount, predicted)), 'R²': r2_score(test.order_amount, predicted), '平均值基準 MAE': mean_absolute_error(test.order_amount, np.full(len(test), train.order_amount.mean()))}
    results = test[['order_id', 'order_date', 'order_amount']].copy()
    results['predicted_amount'] = predicted
    return model, metrics, results, len(train)
