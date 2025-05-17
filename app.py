from flask import Flask, request, jsonify, render_template_string
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

from dataproc import df5, recommend_similar, scaler, features, kmeans

# Create the Flask app before you use @app.route
app = Flask(__name__)

@app.route('/List')

def List():
    return df5.to_html()
  
@app.route('/recommend')
def recommend():
    idx = request.args.get('idx', default=2, type=int)
    if idx not in df5.index:
        return f"Index {idx} not found in data.", 400
    recs = recommend_similar(df5, scaler, features, kmeans, query_idx=idx, n=4)
    table_html = recs.to_html(classes="styled-table", border=0)
    return render_template_string('''
        <html>
        <head>
        <style>
            body { font-family: Arial, sans-serif; background: #f8f9fa; margin: 40px;}
            h2 { color: #333; }
            .styled-table {
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 1em;
                min-width: 600px;
                box-shadow: 0 0 10px rgba(0,0,0,0.08);
                background: #fff;
            }
            .styled-table th, .styled-table td {
                padding: 12px 15px;
                border: 1px solid #ddd;
            }
            .styled-table th {
                background-color: #007bff;
                color: #fff;
                text-align: left;
            }
            .styled-table tr:nth-child(even) {
                background-color: #f3f3f3;
            }
            .styled-table tr:hover {
                background-color: #e9ecef;
            }
            a { color: #007bff; text-decoration: none; }
        </style>
        </head>
        <body>
            <h2>Recommended Similar Properties</h2>
            {{ table_html|safe }}
            <a href="/choose">&larr; Back to Choose</a>
        </body>
        </html>
    ''', table_html=table_html)

@app.route('/choose')
def choose():
    # Standardize city names for reliable filtering
    df5['property_city_clean'] = df5['property_city'].str.strip().str.lower()
    # Get unique cities for dropdown (original casing)
    cities = sorted(df5['property_city'].dropna().unique())
    city = request.args.get('property_city', default='', type=str)
    if city:
        filtered = df5[df5['property_city_clean'] == city.strip().lower()]
    else:
        filtered = df5
    table_html = filtered.head(200).to_html(classes="styled-table", border=0)
    return render_template_string('''
        <html>
        <head>
        <style>
            body { font-family: Arial, sans-serif; background: #f8f9fa; margin: 40px;}
            h2 { color: #333; }
            .styled-table {
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 1em;
                min-width: 600px;
                box-shadow: 0 0 10px rgba(0,0,0,0.08);
                background: #fff;
            }
            .styled-table th, .styled-table td {
                padding: 12px 15px;
                border: 1px solid #ddd;
            }
            .styled-table th {
                background-color: #007bff;
                color: #fff;
                text-align: left;
            }
            .styled-table tr:nth-child(even) {
                background-color: #f3f3f3;
            }
            .styled-table tr:hover {
                background-color: #e9ecef;
            }
            form { margin-top: 30px; }
            input[type="number"], input[type="text"] { padding: 6px; width: 120px; }
            input[type="submit"], .list-btn { padding: 6px 16px; background: #007bff; color: #fff; border: none; border-radius: 3px; cursor: pointer;}
            .list-btn { margin-left: 10px; text-decoration: none; }
        </style>
        </head>
        <body>
            <h2>Available Properties (first 20 rows)</h2>
            {{ table_html|safe }}
            <form method="get" action="/choose">
                <label>Filter by City: <input name="property_city" type="text" value="{{ request.args.get('property_city', '') }}"></label>
                <input type="submit" value="Filter">
                <a href="/List" class="list-btn">View Full List</a>
            </form>
            <form action="/recommend" style="margin-top:20px;">
                <label>Enter index: <input name="idx" type="number" min="0" required></label>
                <input type="submit" value="Get Recommendations">
            </form>
        </body>
        </html>
    ''', table_html=table_html)

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    app.run(debug=True)