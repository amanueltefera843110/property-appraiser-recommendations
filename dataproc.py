import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv("combined_appraisals.csv")


mask = df1['subject_year_built'].astype(str).str.fullmatch(r'\d{4}')
df_years = df1[mask].copy()

# 2. Convert year to int
df_years['subject_year_built'] = df_years['subject_year_built'].astype(int)

# 3. Subtract from 2025 to get the age
df_years['age'] = 2025 - df_years['subject_year_built']
df2 = df_years.copy()

keep_columns = [
    # Location information
    'property_city', 'property_province', 'property_postal_code',
    'property_latitude', 'property_longitude',
    
    # Property characteristics 
    'property_bedrooms', 'property_gla', 'property_full_baths', 'property_half_baths',
    'property_room_count', 'property_lot_size_sf', 'property_year_built',
    'property_property_sub_type', 'property_structure_type', 'property_levels',
    'property_basement', 'property_cooling', 'property_heating','subject_subject_age','comp_stories','age',
    
    # Target variable
    'property_close_price', 'property_close_date'
]

# Filter to keep only the columns we want
df2 = df2[keep_columns]

always_keep_and_fill_zero = [
    'property_half_baths',
    'property_lot_size_sf',
    'property_basement',
    'property_full_baths'
]

total_rows = len(df2)
null_percentage = df2.isnull().sum() / total_rows * 100

columns_to_keep = set(null_percentage[null_percentage <= 10].index) | set(always_keep_and_fill_zero)
df3 = df2[list(columns_to_keep)].copy()

for col in always_keep_and_fill_zero:
    if col in df3.columns:
        df3[col] = df3[col].fillna(0)

removed_columns = set(null_percentage[null_percentage > 10].index) - set(always_keep_and_fill_zero)

    # i replaced property_year_built with age so its okay that it is removed

df4 = df3.copy()
df4['price_per_sqft'] = df4['property_close_price']/df3['property_gla']
df5= df4[df4['property_structure_type'].str.contains('house', case=False, na=False)]
df5.dropna(inplace=True)
from fractions import Fraction
import re
import pandas as pd

def parse_storeys(s):

    if not isinstance(s, str):
        return pd.NA

    # look for "2 1/2" or "1.5" or "2"
    m = re.search(r'(\d+\s+\d+/\d+|\d+(\.\d+)?)', s)
    if not m:
        return pd.NA

    token = m.group(1)
    # if it’s “2 1/2” style, split and build via Fraction
    if ' ' in token and '/' in token:
        whole, frac = token.split()
        return float(int(whole) + Fraction(frac))
    return float(token)

# apply and cast to pandas’ nullable FloatDtype
df5['num_storeys'] = df5['comp_stories'].apply(parse_storeys).astype('Float64')

# now drop non-numeric rows
df5 = df5.dropna(subset=['num_storeys'])

numeric_cols = df5.select_dtypes(include='number').columns

features = [
'property_longitude', 'property_latitude', 'property_gla',
       'property_bedrooms', 'property_full_baths', 'property_lot_size_sf',
       'property_room_count', 'property_half_baths', 'property_close_price',
       'age', 'price_per_sqft', 'num_storeys'
]
from sklearn.preprocessing import StandardScaler

X = df5[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
df5['cluster'] = kmeans.fit_predict(X_scaled)
from sklearn.metrics import pairwise_distances

def recommend_similar(df5, scaler, features, kmeans, query_idx, n=2):
    # 1) Get the query point’s cluster
    label = df5.loc[query_idx, 'cluster']

    # 2) Subset to that cluster
    same_cluster = df5[df5['cluster'] == label].copy()

    # 3) Compute scaled distances to the query property
    q_scaled = scaler.transform(df5.loc[[query_idx], features])
    S_scaled = scaler.transform(same_cluster[features])
    dists = pairwise_distances(q_scaled, S_scaled)[0]

    # 4) Attach distances and pick the closest n (excluding itself)
    same_cluster['dist'] = dists
    recs = (
        same_cluster
        .sort_values('dist')
        .drop(query_idx, errors='ignore')
        .head(n)
    )
    return recs

input_idx =43
# example: find 2 houses like the one at index=2 from df6 
recs = recommend_similar(df5, scaler, features, kmeans, query_idx=input_idx, n=4)
print(recs[['property_close_price', 'price_per_sqft', 'num_storeys', 'property_bedrooms','property_latitude',
    'property_longitude']])
df5.head(10)