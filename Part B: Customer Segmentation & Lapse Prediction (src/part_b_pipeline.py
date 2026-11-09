import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def run_part_b():
    os.makedirs("outputs", exist_ok=True)
    orders = pd.read_csv("data/clean/orders.csv")
    cust = pd.read_csv("data/clean/customers.csv")
    orders['order_ts'] = pd.to_datetime(orders['order_ts'])
    cust['signup_date'] = pd.to_datetime(cust['signup_date'])
    
    # ---------------- 1. RFM Table ----------------
    snapshot = pd.to_datetime("2025-10-01 00:00:00")
    comp = orders[orders['status'] == 'completed']
    rfm = comp.groupby('customer_id').agg(
        recency=('order_ts', lambda s: (snapshot - s.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum')
    ).reset_index()
    
    # ---------------- 2 & 3. K-Means & PCA ----------------
    scaler_rfm = StandardScaler()
    X_rfm_scaled = scaler_rfm.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(X_rfm_scaled)
    
    # Profile clusters
    profile = rfm.groupby('cluster').agg({'monetary': 'mean', 'frequency': 'mean', 'recency': 'mean'})
    prem_cluster = profile['monetary'].idxmax()
    
    # Assign business names
    def map_segment(row):
        c = row['cluster']
        if c == prem_cluster:
            return "High-Value Premium", True
        elif row['frequency'] > 12:
            return "Frequent Regulars", False
        elif row['recency'] > 60:
            return "At-Risk Occasional", False
        else:
            return "Emerging Shoppers", False
            
    names_flags = rfm.apply(map_segment, axis=1)
    rfm['segment_name'] = [nf[0] for nf in names_flags]
    rfm['is_premium'] = [nf[1] for nf in names_flags]
    rfm['segment_id'] = rfm['cluster']
    
    pca = PCA(n_components=2)
    pca.fit(X_rfm_scaled)
    
    rfm[['customer_id', 'recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium']].to_csv(
        "outputs/customer_segments.csv", index=False
    )
    
    # ---------------- 4 & 5. Lapse Prediction Data Prep ----------------
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    pre_orders = orders[orders['order_ts'] < cutoff]
    post_orders = orders[(orders['order_ts'] >= cutoff) & (orders['order_ts'] <= pd.to_datetime("2025-09-30 23:59:59"))]
    
    # Cohort: at least one completed order pre-cutoff
    pre_comp = pre_orders[pre_orders['status'] == 'completed']
    cohort_ids = pre_comp['customer_id'].unique()
    
    post_comp_cids = set(post_orders[post_orders['status'] == 'completed']['customer_id'].unique())
    
    prod = pd.read_csv("data/clean/products.csv")
    pre_orders_full = pre_orders.merge(prod[['product_id', 'category']], on='product_id', how='left')
    pre_comp_full = pre_orders_full[pre_orders_full['status'] == 'completed']
    
    # Build feature aggregations
    f_rec = pre_comp.groupby('customer_id')['order_ts'].max().apply(lambda s: (cutoff - s).days).rename('recency')
    f_freq = pre_comp.groupby('customer_id')['order_id'].nunique().rename('frequency')
    f_mon = pre_comp.groupby('customer_id')['order_amount'].sum().rename('monetary')
    f_disc = pre_comp.groupby('customer_id')['discount_pct'].mean().rename('avg_discount_pct')
    f_deliv = pre_comp.groupby('customer_id')['delivery_minutes'].mean().rename('avg_delivery_minutes')
    
    # Cancellation rate
    pre_total = pre_orders[pre_orders['status'].isin(['completed', 'cancelled'])].groupby('customer_id')['order_id'].count()
    pre_canc = pre_orders[pre_orders['status'] == 'cancelled'].groupby('customer_id')['order_id'].count()
    f_canc_rate = (pre_canc / pre_total).fillna(0).rename('cancellation_rate')
    
    # Favourite category
    fav_cat = pre_comp_full.groupby(['customer_id', 'category']).size().reset_index(name='cnt')
    fav_cat = fav_cat.sort_values(['customer_id', 'cnt'], ascending=[True, False]).groupby('customer_id').first()['category'].rename('favourite_category')
    
    feat_df = pd.DataFrame({'customer_id': cohort_ids})
    feat_df = feat_df.join(f_rec, on='customer_id')
    feat_df = feat_df.join(f_freq, on='customer_id')
    feat_df = feat_df.join(f_mon, on='customer_id')
    feat_df = feat_df.join(f_disc, on='customer_id')
    feat_df = feat_df.join(f_canc_rate, on='customer_id')
    feat_df = feat_df.join(f_deliv, on='customer_id')
    feat_df = feat_df.join(fav_cat, on='customer_id')
    feat_df = feat_df.merge(cust[['customer_id', 'city', 'age', 'gender', 'signup_date']], on='customer_id', how='left')
    feat_df['tenure'] = (cutoff - feat_df['signup_date']).dt.days
    feat_df = feat_df.drop(columns=['signup_date'])
    
    feat_df['lapsed'] = feat_df['customer_id'].apply(lambda cid: 1 if cid not in post_comp_cids else 0)
    
    # Split
    X = feat_df.drop(columns=['customer_id', 'lapsed'])
    y = feat_df['lapsed']
    cids = feat_df['customer_id']
    
    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
        X, y, cids, test_size=0.2, stratify=y, random_state=42
    )
    
    num_cols = ['recency', 'frequency', 'monetary', 'avg_discount_pct', 'cancellation_rate', 'avg_delivery_minutes', 'age', 'tenure']
    cat_cols = ['city', 'gender', 'favourite_category']
    
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
    
    X_tr = np.hstack([X_train_num, X_train_cat])
    X_te = np.hstack([X_test_num, X_test_cat])
    
    # ---------------- 6. Train, Tune, Evaluate ----------------
    # GridSearch LR
    param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
    gs_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1')
    gs_lr.fit(X_tr, y_train)
    
    # GridSearch RF
    param_grid_rf = {'max_depth': [3, 5, 8], 'min_samples_leaf': [5, 10, 20]}
    gs_rf = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42), param_grid_rf, cv=5, scoring='f1')
    gs_rf.fit(X_tr, y_train)
    
    # Best model selection: Random Forest
    best_rf = gs_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_te)
    
    # Save Pipeline bundle
    model_bundle = {
        'model': best_rf,
        'scaler': scaler,
        'ohe': ohe,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
    joblib.dump(model_bundle, "outputs/lapse_model.joblib")
    
    # ---------------- 8. Retention List ----------------
    test_probs = best_rf.predict_proba(X_te)[:, 1]
    res_test = pd.DataFrame({
        'customer_id': cid_test,
        'lapse_probability': test_probs,
        'actually_lapsed': y_test
    })
    
    segments = pd.read_csv("outputs/customer_segments.csv")
    res_test = res_test.merge(segments[['customer_id', 'is_premium']], on='customer_id', how='left')
    prem_test = res_test[res_test['is_premium'] == True].sort_values('lapse_probability', ascending=False)
    
    retention_list = prem_test.head(10)[['customer_id', 'lapse_probability', 'actually_lapsed']]
    retention_list.to_csv("outputs/retention_list.csv", index=False)

if __name__ == "__main__":
    run_part_b()
