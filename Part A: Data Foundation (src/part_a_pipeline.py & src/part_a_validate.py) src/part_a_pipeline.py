import os
import sqlite3
import pandas as pd
import numpy as np

def run_part_a():
    os.makedirs("data/clean", exist_ok=True)
    
    # 1. Load Raw Data
    cust_raw = pd.read_csv("data/raw/customers.csv")
    prod_raw = pd.read_csv("data/raw/products.csv")
    orders_raw = pd.read_csv("data/raw/orders.csv")
    
    # 2. Clean Customers
    cust = cust_raw.copy()
    # (a) City standardize
    cust['city'] = cust['city'].astype(str).str.strip().str.title()
    # (b) Gender standardize
    def clean_gender(g):
        if pd.isna(g) or str(g).strip() == '':
            return 'unknown'
        g_str = str(g).strip().lower()
        if g_str in ['male', 'm']:
            return 'male'
        elif g_str in ['female', 'f']:
            return 'female'
        return 'unknown'
    cust['gender'] = cust['gender'].apply(clean_gender)
    # (c) Age outside 10-100 treated as missing, imputed with city median
    cust['age'] = pd.to_numeric(cust['age'], errors='coerce')
    cust.loc[(cust['age'] < 10) | (cust['age'] > 100), 'age'] = np.nan
    cust['age'] = cust.groupby('city')['age'].transform(lambda s: s.fillna(s.median()))
    # (d) Signup date to datetime
    cust['signup_date'] = pd.to_datetime(cust['signup_date'])
    
    # 3. Clean Orders
    orders = orders_raw.copy()
    # (1) order_ts to datetime
    orders['order_ts'] = pd.to_datetime(orders['order_ts'])
    # (2) Remove exact duplicate rows
    orders = orders.drop_duplicates()
    # (3) Keep latest record per order_id
    orders = orders.sort_values('order_ts').groupby('order_id', as_index=False).last()
    # (4) Numeric amount and fill missing
    orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
    orders = orders.merge(prod_raw[['product_id', 'price_inr', 'category']], on='product_id', how='left')
    calc_amount = (orders['price_inr'] * orders['quantity'] * (1 - orders['discount_pct'] / 100.0)).round(2)
    orders['order_amount'] = orders['order_amount'].fillna(calc_amount)
    # (5) Delivery minutes
    orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
    orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
    # Median delivery time among completed orders by product category
    completed_mask = orders['status'] == 'completed'
    median_deliv = orders[completed_mask].groupby('category')['delivery_minutes'].transform('median')
    orders.loc[completed_mask, 'delivery_minutes'] = orders.loc[completed_mask, 'delivery_minutes'].fillna(median_deliv)
    orders.loc[orders['status'] == 'cancelled', 'delivery_minutes'] = np.nan
    
    # Save Clean CSVs
    cust.to_csv("data/clean/customers.csv", index=False)
    prod_raw.to_csv("data/clean/products.csv", index=False)
    orders_clean_export = orders[['order_id', 'customer_id', 'product_id', 'quantity', 'order_ts', 
                                  'status', 'discount_pct', 'order_amount', 'delivery_minutes']]
    orders_clean_export.to_csv("data/clean/orders.csv", index=False)
    
    # Save to SQLite
    db_path = "data/instamart.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cust_db = cust.copy()
    cust_db['signup_date'] = cust_db['signup_date'].dt.strftime('%Y-%m-%d')
    cust_db.to_sql("customers", conn, if_exists="replace", index=False)
    prod_raw.to_sql("products", conn, if_exists="replace", index=False)
    
    ord_db = orders_clean_export.copy()
    ord_db['order_ts'] = ord_db['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    ord_db.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    run_part_a()
