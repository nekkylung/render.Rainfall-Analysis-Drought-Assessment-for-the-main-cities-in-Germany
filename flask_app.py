from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import requests

app = Flask(__name__)

# Global variable to cache data
cached_data = None
data_last_loaded = None

def load_and_process_data():
    """Load and process SPI data"""
    global cached_data, data_last_loaded
    
    try:
        # Check if we need to reload data (cache for 1 hour)
        if (cached_data is None or 
            data_last_loaded is None or 
            (datetime.now() - data_last_loaded).seconds > 3600):
            
            print("Loading SPI data...")
            df = pd.read_csv('spi_tableau_data.csv')
            df['Date'] = pd.to_datetime(df['Date_String'])
            
            # Process data for frontend
            cities = sorted(df['City'].unique().tolist())
            years = sorted([int(year) for year in df['Year'].unique()])
            dates = sorted(df['Date'].unique())
            
            # Prepare time series data
            time_series_data = {}
            for scale in ['1M', '3M', '6M', '12M']:
                time_series_data[scale] = []
                for date in dates:
                    date_pd = pd.to_datetime(date)
                    date_data = df[df['Date'] == date].copy()
                    daily_data = []
                    for _, row in date_data.iterrows():
                        spi_value = row[f'SPI_{scale}']
                        if not pd.isna(spi_value):
                            daily_data.append({
                                'lat': float(row['Latitude']),
                                'lng': float(row['Longitude']),
                                'city': str(row['City']),
                                'spi': float(spi_value),
                                'rainfall': float(row['Rainfall_mm']) if not pd.isna(row['Rainfall_mm']) else None,
                                'temperature': float(row['Temperature_C']) if not pd.isna(row['Temperature_C']) else None,
                                'humidity': float(row['Humidity_pct']) if not pd.isna(row['Humidity_pct']) else None,
                                'elevation': int(row['Elevation_m']),
                                'climate': str(row['Climate_Type'])
                            })
                    time_series_data[scale].append({
                        'date': date_pd.strftime('%Y-%m'),
                        'data': daily_data
                    })
            
            # Prepare summary data
            summary_data = {}
            for scale in ['1M', '3M', '6M', '12M']:
                # Overall summary
                overall = df.groupby('City').agg({
                    'Latitude': 'first',
                    'Longitude': 'first',
                    f'SPI_{scale}': 'mean',
                    'Climate_Type': 'first',
                    'Elevation_m': 'first',
                    'Rainfall_mm': 'mean',
                    'Temperature_C': 'mean',
                    'Humidity_pct': 'mean'
                }).reset_index()
                
                summary_data[f'{scale}_overall'] = []
                for _, row in overall.iterrows():
                    summary_data[f'{scale}_overall'].append({
                        'city': str(row['City']),
                        'lat': float(row['Latitude']),
                        'lng': float(row['Longitude']),
                        'spi': float(row[f'SPI_{scale}']) if not pd.isna(row[f'SPI_{scale}']) else None,
                        'rainfall': float(row['Rainfall_mm']) if not pd.isna(row['Rainfall_mm']) else None,
                        'temperature': float(row['Temperature_C']) if not pd.isna(row['Temperature_C']) else None,
                        'humidity': float(row['Humidity_pct']) if not pd.isna(row['Humidity_pct']) else None,
                        'elevation': int(row['Elevation_m']),
                        'climate': str(row['Climate_Type'])
                    })
                
                # Yearly summaries
                for year in years:
                    yearly = df[df['Year'] == year].groupby('City').agg({
                        'Latitude': 'first',
                        'Longitude': 'first',
                        f'SPI_{scale}': 'mean',
                        'Climate_Type': 'first',
                        'Elevation_m': 'first',
                        'Rainfall_mm': 'mean',
                        'Temperature_C': 'mean',
                        'Humidity_pct': 'mean'
                    }).reset_index()
                    
                    summary_data[f'{scale}_{year}'] = []
                    for _, row in yearly.iterrows():
                        summary_data[f'{scale}_{year}'].append({
                            'city': str(row['City']),
                            'lat': float(row['Latitude']),
                            'lng': float(row['Longitude']),
                            'spi': float(row[f'SPI_{scale}']) if not pd.isna(row[f'SPI_{scale}']) else None,
                            'rainfall': float(row['Rainfall_mm']) if not pd.isna(row['Rainfall_mm']) else None,
                            'temperature': float(row['Temperature_C']) if not pd.isna(row['Temperature_C']) else None,
                            'humidity': float(row['Humidity_pct']) if not pd.isna(row['Humidity_pct']) else None,
                            'elevation': int(row['Elevation_m']),
                            'climate': str(row['Climate_Type'])
                        })
            
            # Prepare heatmap data
            heatmap_data = {}
            for scale in ['1M', '3M', '6M', '12M']:
                heat_data = []
                for date in dates:
                    date_data = df[df['Date'] == date].copy()
                    daily_data = []
                    for _, row in date_data.iterrows():
                        spi_value = row[f'SPI_{scale}']
                        if not pd.isna(spi_value):
                            intensity = min(abs(float(spi_value)) * 15, 100)
                            daily_data.append([float(row['Latitude']), float(row['Longitude']), float(intensity)])
                    heat_data.append(daily_data)
                
                heatmap_data[scale] = {
                    'data': heat_data,
                    'index': [pd.to_datetime(date).strftime('%Y-%m') for date in dates]
                }
            
            cached_data = {
                'timeSeriesData': time_series_data,
                'summaryData': summary_data,
                'heatmapData': heatmap_data,
                'cities': cities,
                'years': years,
                'dateRange': {
                    'min': df['Date'].min().strftime('%Y-%m'),
                    'max': df['Date'].max().strftime('%Y-%m')
                },
                'totalRecords': len(df)
            }
            data_last_loaded = datetime.now()
            print(f"Data loaded successfully: {len(df)} records for {len(cities)} cities")
        
        return cached_data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_germany_boundaries():
    """Get Germany administrative boundaries as GeoJSON"""
    try:
        url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/germany.geojson"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        try:
            url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
    return None

@app.route('/')
def index():
    """Main dashboard page"""
    data = load_and_process_data()
    if data is None:
        return "Error: Could not load data. Please check if 'spi_tableau_data.csv' exists.", 500
    return render_template('index.html', 
                         date_range=data['dateRange'],
                         total_records=data['totalRecords'],
                         cities_count=len(data['cities']))

@app.route('/api/data')
def get_data():
    """API endpoint to serve all processed SPI data"""
    data = load_and_process_data()
    if data is None:
        return jsonify({'error': 'Data file not found or could not be processed'}), 404
    return jsonify(data)

@app.route('/api/germany-boundaries')
def get_boundaries():
    """API endpoint to serve Germany boundaries"""
    boundaries = get_germany_boundaries()
    if boundaries:
        return jsonify(boundaries)
    else:
        return jsonify({'error': 'Could not load Germany boundaries'}), 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': cached_data is not None
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize data on startup
@app.before_first_request
def initialize():
    """Initialize data when the app starts"""
    print("Initializing SPI Dashboard...")
    load_and_process_data()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)