# Property Recommendation System

This project is a web-based Property Recommendation App built with Flask and Python. It allows users to filter properties by city, postal code, and bedrooms, view property details, and get recommendations for similar properties using machine learning.

## Features

- **Interactive Web App**: Built with Flask for easy property search and recommendations.
- **Filtering**: Filter properties by city, postal code, and number of bedrooms.
- **Recommendations**: Get similar property recommendations using clustering and distance metrics.
- **Data Visualization**: Uses pandas, matplotlib, and seaborn for data processing and visualization.
- **Machine Learning**: Utilizes scikit-learn for clustering and scaling features.

## Setup

1. **Clone or Download the Repository**

2. **Install Requirements**

   Open your terminal and run:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare Data**

   - Place your `combined_appraisals.csv` file in the project directory.
   - If you have a JSON dataset, use `import.py` to convert it to CSV.

4. **Run the App**

   ```
   python app.py
   ```

   The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Files

- `app.py` - Main Flask application.
- `dataproc.py` - Data processing and ML functions.
- `import.py` - Script to convert JSON data to CSV.
- `requirements.txt` - Python dependencies.
- `README.md` - Project documentation.

## Requirements

- Python 3.7+
- See `requirements.txt` for required packages.

by Amanuel Tefera
