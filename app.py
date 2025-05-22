from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 10)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from dataproc import df5, recommend_similar, scaler, features, kmeans

app = Flask(__name__)

# --- Improved HTML header and CSS ---
HTML_HEADER = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Property Recommendation App</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: 'Arial', sans-serif; 
            background: linear-gradient(135deg, #74b9ff, #0984e3); 
            margin: 0; 
            padding: 0;
            min-height: 100vh;
        }
        
        .header { 
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            color: #fff; 
            padding: 30px 0; 
            text-align: center; 
            font-size: 2.5rem; 
            font-weight: 700; 
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .container { 
            max-width: 1000px; 
            margin: 40px auto; 
            padding: 40px; 
            background: #fff; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: relative;
        }
        
        h2 { 
            color: #2d3436; 
            margin-bottom: 25px;
            font-size: 1.8rem;
            border-bottom: 3px solid #74b9ff;
            padding-bottom: 10px;
        }
        
        .styled-table { 
            border-collapse: collapse; 
            margin: 30px 0; 
            font-size: 0.95em; 
            width: 100%;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
            background: #fff;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .styled-table th, .styled-table td { 
            padding: 15px 12px; 
            text-align: left;
        }
        
        .styled-table th { 
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: #fff; 
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        
        .styled-table td {
            border-bottom: 1px solid #eee;
        }
        
        .styled-table tr:nth-child(even) td { 
            background-color: #f8f9fa; 
        }
        
        .styled-table tr:hover td { 
            background-color: #e3f2fd;
            transform: scale(1.01);
            transition: all 0.2s ease;
        }
        
        .form-container {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
            border: 1px solid #dee2e6;
        }
        
        form { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 15px; 
            align-items: center;
        }
        
        label { 
            font-weight: 600;
            color: #495057;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        input[type="number"], input[type="text"], select { 
            padding: 12px 15px; 
            width: 160px; 
            border-radius: 8px; 
            border: 2px solid #dee2e6;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        input[type="number"]:focus, input[type="text"]:focus, select:focus { 
            outline: none;
            border-color: #74b9ff;
            box-shadow: 0 0 0 3px rgba(116, 185, 255, 0.1);
        }
        
        .btn {
            padding: 12px 25px; 
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: #fff; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 14px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(116, 185, 255, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #6c5ce7, #5f3dc4);
        }
        
        .btn-secondary:hover {
            box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4);
        }
        
        .back-btn {
            background: linear-gradient(135deg, #fd79a8, #e84393);
            margin-top: 20px;
        }
        
        .back-btn:hover {
            box-shadow: 0 5px 15px rgba(253, 121, 168, 0.4);
        }
        
        .footer { 
            text-align: center; 
            margin-top: 50px; 
            color: rgba(255,255,255,0.8); 
            font-size: 1rem;
            padding: 20px;
        }
        
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #a8e6cf, #7fcdcd);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #2d3436;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        @media (max-width: 768px) {
            .container { 
                margin: 20px 10px; 
                padding: 20px; 
            }
            .header { 
                font-size: 1.8rem; 
                padding: 20px 0;
            }
            .styled-table {
                font-size: 0.8em;
            }
            .styled-table th, .styled-table td {
                padding: 10px 8px;
            }
            form {
                flex-direction: column;
                align-items: stretch;
            }
            input[type="number"], input[type="text"], select {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        🏠 Property Recommendation System
    </div>
    <div class="container">
'''

HTML_FOOTER = '''
    </div>
    <div class="footer">
        © 2025 Property Recommendation System | Made with Flask & Python
    </div>
</body>
</html>
'''

# --- Flask routes ---

@app.route('/')
def home():
    """Redirects to choose."""
    return redirect(url_for('choose'))

@app.route('/List')
def List():
    """Shows the entire dataframe."""
    total_properties = len(df5)
    table_html = df5.to_html(classes="styled-table", border=0, index=False)
    
    content = f'''
        <h2>📋 Complete Property Database</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-number">{total_properties}</span>
                <span class="stat-label">Total Properties</span>
            </div>
        </div>
        {table_html}
        <a class='btn back-btn' href='/choose'>← Back to Search</a>
    '''
    return HTML_HEADER + content + HTML_FOOTER

@app.route('/recommend')
def recommend():
    """Recommends similar properties by index."""
    idx = request.args.get('idx', default=2, type=int)
    if idx not in df5.index:
        content = f'''
            <div class="alert">
                <strong>Error:</strong> Property index {idx} not found in our database.
            </div>
            <a class='btn back-btn' href='/choose'>← Back to Search</a>
        '''
        return HTML_HEADER + content + HTML_FOOTER, 404
    
    recs = recommend_similar(df5, scaler, features, kmeans, query_idx=idx, n=4)
    table_html = recs.to_html(classes="styled-table", border=0, index=False)
    
    content = f'''
        <h2>🎯 Recommended Similar Properties</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-number">{len(recs)}</span>
                <span class="stat-label">Recommendations</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{idx}</span>
                <span class="stat-label">Base Property Index</span>
            </div>
        </div>
        {table_html}
        <a class="btn back-btn" href="/choose">← Back to Search</a>
    '''
    return HTML_HEADER + content + HTML_FOOTER

@app.route('/choose')
def choose():
    """Filter and select properties. Enter index for recommendations."""
    # City names cleaned for reliable filtering
    df5['property_city_clean'] = df5['property_city'].astype(str).str.strip().str.lower()

    city = request.args.get('property_city', default='', type=str)
    postal_code = request.args.get('property_postal_code', default='', type=str)
    bedrooms_filter = request.args.get('property_bedrooms', default='', type=str)

    # Filtering logic
    filtered = df5.copy()
    if city:
        filtered = filtered[filtered['property_city_clean'] == city.strip().lower()]
    if postal_code:
        filtered = filtered[filtered['property_postal_code'].astype(str).str.strip() == postal_code.strip()]
    if bedrooms_filter:
        try:
            bedrooms_val = int(bedrooms_filter)
            filtered = filtered[filtered['property_bedrooms'] == bedrooms_val]
        except ValueError:
            pass  # Ignore invalid

    # Reorder columns: city, bedrooms, postal code, then the rest
    display_data = filtered.head(20)
    cols = display_data.columns.tolist()
    first_cols = ['property_city', 'property_bedrooms', 'property_postal_code']
    first_cols = [col for col in first_cols if col in cols]
    rest_cols = [col for col in cols if col not in first_cols]
    ordered_cols = first_cols + rest_cols
    display_data = display_data[ordered_cols]

    # Reset index so it is sequential for the filtered results
    display_data = display_data.reset_index(drop=False)
    # Move the original index to the first column, rename for clarity
    display_data.rename(columns={'index': 'Property Index'}, inplace=True)

    table_html = display_data.to_html(classes="styled-table", border=0, index=False)

    # Stats
    total_filtered = len(filtered)
    total_properties = len(df5)

    content = f'''
        <h2>🔍 Property Search & Recommendations</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-number">{total_properties}</span>
                <span class="stat-label">Total Properties</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{total_filtered}</span>
                <span class="stat-label">Filtered Results</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{len(display_data)}</span>
                <span class="stat-label">Showing</span>
            </div>
        </div>
        
        <div class="form-container">
            <h3 style="margin-bottom: 20px; color: #2d3436;">🔧 Filter Properties</h3>
            <form method="get" action="/choose">
                <label>
                    City
                    <input name="property_city" type="text" value="{city}" placeholder="Enter city name">
                </label>
                <label>
                    Postal Code
                    <input name="property_postal_code" type="text" value="{postal_code}" placeholder="Enter postal code">
                </label>
                <label>
                    Bedrooms
                    <input name="property_bedrooms" type="number" min="0" value="{bedrooms_filter}" placeholder="Number of bedrooms">
                </label>
                <input type="submit" value="🔍 Filter" class="btn">
                <a href="/List" class="btn btn-secondary">📋 View All</a>
            </form>
            <h3 style="margin: 30px 0 10px 0; color: #2d3436;">🎯 Get Recommendations</h3>
            <form action="/recommend">
                <label>
                    Property Index
                    <input name="idx" type="number" min="0" required placeholder="Enter property index (see table)">
                </label>
                <input type="submit" value="🚀 Get Recommendations" class="btn">
            </form>
        </div>
        
        <h3 style="color: #2d3436; margin-top: 30px;">Properties Found (showing first 20)</h3>
        <div style="overflow-x:auto;">{table_html}</div>
    '''
    return HTML_HEADER + content + HTML_FOOTER

if __name__ == "__main__":
    print("Starting Python Flask Server For Property Recommendations...")
  
    app.run(debug=True)