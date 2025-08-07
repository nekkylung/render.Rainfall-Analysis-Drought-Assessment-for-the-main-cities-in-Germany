from flask import Flask, render_template, jsonify, send_file, send_from_directory
import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# Create Flask app
app = Flask(__name__)

# Create static directory for serving files
STATIC_DIR = 'static'
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def process_rainfall_data():
    """Process the rainfall data and perform initial analysis"""
    print("Starting data processing...")
    
    # Load and process data
    df_rainfall = pd.read_csv('Rainfall_Data_Germany_Complete.csv')
    
    # Basic info
    print(f"Dataset shape: {df_rainfall.shape}")
    df_rainfall.info()
    
    # Check missing values
    missing = df_rainfall.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing Values:")
        print(missing[missing > 0])
    else:
        print(f"\nNo missing values found")
    
    # Check duplicates
    if all(col in df_rainfall.columns for col in ['City', 'Year', 'Month','Rainfall (mm)','Climate_Type','Temperature (°C)', 'Humidity (%)']):
        duplicates = df_rainfall.duplicated(subset=['City', 'Year', 'Month','Rainfall (mm)','Climate_Type','Temperature (°C)', 'Humidity (%)']).sum()
        print(f"\nDuplicate records: {duplicates}")
    
    # Validity checks
    print("\nValidity Checks:")
    if 'Rainfall (mm)' in df_rainfall.columns:
        print(f"  Invalid rainfall: {(df_rainfall['Rainfall (mm)'] < 0).sum()}")
    if 'Year' in df_rainfall.columns:
        print(f"  Invalid Year: {((df_rainfall['Year'] <2000 ) | (df_rainfall['Year'] > 2025)).sum()}")
    if 'Month' in df_rainfall.columns:
        print(f"  Invalid months: {((df_rainfall['Month'] < 1) | (df_rainfall['Month'] > 12)).sum()}")
    
    # Save cleaned data
    df_rainfall.to_csv(os.path.join(STATIC_DIR, "Rainfall_Cleaned.csv"), index=False)
    print("Cleaned data saved")
    
    return df_rainfall

def create_stacked_bar_chart():
    """Create and save the stacked bar chart"""
    print("Creating stacked bar chart...")
    
    # Read cleaned data
    df_cleaned = pd.read_csv(os.path.join(STATIC_DIR, 'Rainfall_Cleaned.csv'))
    
    # Calculate annual rainfall per city
    city_annual = df_cleaned.groupby(['City', 'Year', 'Climate_Type'])['Rainfall (mm)'].sum().reset_index()
    
    # Calculate total rainfall per climate type per year
    climate_total = city_annual.groupby(['Climate_Type', 'Year'])['Rainfall (mm)'].sum().reset_index()
    
    # Pivot the data for stacked bar chart
    climate_pivot = climate_total.pivot(index='Year', columns='Climate_Type', values='Rainfall (mm)').fillna(0)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    climate_pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8)
    
    # Add value labels on bars
    for i, climate in enumerate(climate_pivot.columns):
        for j, value in enumerate(climate_pivot[climate]):
            if value > 1000:  # Only label if rainfall > 1000mm
                y_pos = sum(climate_pivot.iloc[j, :i]) + value/2
                ax.text(j, y_pos, f'{value:,.0f}',
                        ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add trend line
    total_by_year = climate_pivot.sum(axis=1).values
    x_pos = np.arange(len(climate_pivot.index))
    z = np.polyfit(x_pos, total_by_year, 1)
    p = np.poly1d(z)
    ax.plot(x_pos, p(x_pos), "k--", alpha=0.7, linewidth=2, label='Trend')
    
    # Customize plot
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Annual Rainfall (mm)')
    ax.set_title('Total Annual Rainfall by Climate Type')
    ax.legend(title='Climate Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save outputs
    plt.savefig(os.path.join(STATIC_DIR, 'rainfall_stacked_bar_simplified.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export data
    climate_pivot.to_csv(os.path.join(STATIC_DIR, 'rainfall_stacked_data.csv'))
    print("Stacked bar chart saved")

def create_seasonal_charts():
    """Create and save seasonal rainfall charts"""
    print("Creating seasonal charts...")
    
    # Load data
    df = pd.read_csv(os.path.join(STATIC_DIR, 'Rainfall_Cleaned.csv'))
    
    # Map months to seasons
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'
    }
    df['Season'] = df['Month'].map(season_map)
    
    # Calculate total rainfall by city, climate type, year, and season
    city_totals = df.groupby(['Climate_Type', 'City', 'Year', 'Season'])['Rainfall (mm)'].sum().reset_index()
    
    # Calculate average rainfall per climate type
    climate_avg = []
    for climate in df['Climate_Type'].unique():
        climate_data = city_totals[city_totals['Climate_Type'] == climate]
        n_cities = climate_data['City'].nunique()
        
        avg = climate_data.groupby(['Year', 'Season'])['Rainfall (mm)'].sum().reset_index()
        avg['Avg_Rainfall (mm)'] = avg['Rainfall (mm)'] / n_cities
        avg['Climate_Type'] = climate
        climate_avg.append(avg)
    
    # Combine into one DataFrame
    final_data = pd.concat(climate_avg, ignore_index=True)
    
    # Define season order
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    climate_types = df['Climate_Type'].unique()[:2]  # First 2 climate types
    
    # Calculate overall statistics
    pivot = final_data.pivot_table(index='Season', columns='Year', values='Avg_Rainfall (mm)', aggfunc='mean').reindex(season_order)
    mean_rainfall = pivot.mean(axis=1)
    std_rainfall = pivot.std(axis=1)
    
    # Plot for each climate type
    for i, climate in enumerate(climate_types):
        plt.figure(figsize=(12, 8))
        climate_data = final_data[final_data['Climate_Type'] == climate]
        
        # Add shaded region for ±1 Std Dev
        plt.fill_between(season_order,
                         mean_rainfall - std_rainfall,
                         mean_rainfall + std_rainfall,
                         color='gray', alpha=0.2, label='±1 Std Dev (Overall)')
        
        # Add mean line
        plt.plot(season_order, mean_rainfall, color='black', linestyle='--', linewidth=2, label='Mean (Overall)')
        
        # Plot each year as a separate line
        for year in sorted(climate_data['Year'].unique()):
            year_data = climate_data[climate_data['Year'] == year]
            
            # Ensure all seasons are present
            plot_data = []
            for season in season_order:
                season_val = year_data[year_data['Season'] == season]['Avg_Rainfall (mm)'].values
                plot_data.append(season_val[0] if len(season_val) > 0 else 0)
            
            plt.plot(season_order, plot_data, marker='o', label=f'{year}', linewidth=1.5)
        
        plt.title(f'Average Rainfall by Season - {climate} Climate', fontsize=16, fontweight='bold')
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Rainfall (mm)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.legend(title='Year', fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1), title_fontsize='large')
        plt.tight_layout()
        
        # Save as PNG
        plt.savefig(os.path.join(STATIC_DIR, f'{climate}_climate_rainfall.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Seasonal charts saved")

def calculate_spi():
    """Calculate SPI values for all cities and time scales"""
    print("Calculating SPI values...")
    
    # Load data
    df = pd.read_csv(os.path.join(STATIC_DIR, 'Rainfall_Cleaned.csv'))
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df = df.sort_values(['City', 'Date'])
    
    # Initialize results list
    all_results = []
    
    # Time scales to calculate (in months)
    time_scales = [1, 3, 6, 12]
    
    # Get all cities
    cities = df['City'].unique()
    
    # Calculate SPI for each city
    for city in cities:
        print(f"Processing SPI for {city}...")
        city_data = df[df['City'] == city].copy()
        
        # Ensure continuous monthly sequence
        city_data = city_data.set_index('Date')
        
        # Get city metadata
        lat = city_data['Latitude'].iloc[0]
        lon = city_data['Longitude'].iloc[0]
        climate_type = city_data['Climate_Type'].iloc[0]
        elevation = city_data['Elevation (m)'].iloc[0]
        
        # Calculate SPI for each time scale
        spi_results = {}
        
        for scale_months in time_scales:
            # Calculate rolling sum for the specified scale
            rainfall_sum = city_data['Rainfall (mm)'].rolling(window=scale_months, min_periods=scale_months).sum()
            
            # Remove NaN values for fitting
            valid_data = rainfall_sum.dropna()
            
            if len(valid_data) < 30:  # Need at least 30 data points
                spi_values = pd.Series(np.nan, index=city_data.index)
            else:
                # Remove zeros for gamma fitting
                non_zero_data = valid_data[valid_data > 0]
                
                if len(non_zero_data) < 10:
                    spi_values = pd.Series(np.nan, index=city_data.index)
                else:
                    # Fit gamma distribution
                    alpha, loc, beta = stats.gamma.fit(non_zero_data, floc=0)
                    
                    # Calculate probability of zero precipitation
                    prob_zero = len(valid_data[valid_data == 0]) / len(valid_data)
                    
                    # Calculate SPI values
                    spi_list = []
                    
                    for value in rainfall_sum:
                        if pd.isna(value):
                            spi_list.append(np.nan)
                        elif value == 0:
                            prob = prob_zero
                            spi_list.append(stats.norm.ppf(prob))
                        else:
                            prob = prob_zero + (1 - prob_zero) * stats.gamma.cdf(value, alpha, loc=loc, scale=beta)
                            prob = np.clip(prob, 0.0001, 0.9999)
                            spi_list.append(stats.norm.ppf(prob))
                    
                    spi_values = pd.Series(spi_list, index=city_data.index)
            
            spi_results[f'SPI_{scale_months}M'] = spi_values
        
        # Create rows for each date
        for idx in range(len(city_data)):
            row_data = {
                'City': city,
                'Latitude': lat,
                'Longitude': lon,
                'Date': city_data.index[idx],
                'Year': city_data.index[idx].year,
                'Month': city_data.index[idx].month,
                'Month_Name': city_data.index[idx].strftime('%B'),
                'Climate_Type': climate_type,
                'Elevation_m': elevation,
                'Rainfall_mm': city_data['Rainfall (mm)'].iloc[idx],
                'Temperature_C': city_data['Temperature (°C)'].iloc[idx],
                'Humidity_pct': city_data['Humidity (%)'].iloc[idx],
                'SPI_1M': spi_results['SPI_1M'].iloc[idx],
                'SPI_3M': spi_results['SPI_3M'].iloc[idx],
                'SPI_6M': spi_results['SPI_6M'].iloc[idx],
                'SPI_12M': spi_results['SPI_12M'].iloc[idx]
            }
            all_results.append(row_data)
    
    # Create final dataframe
    result_df = pd.DataFrame(all_results)
    
    # Add SPI classification function
    def classify_spi(spi_value):
        if pd.isna(spi_value):
            return 'No Data'
        elif spi_value >= 2.0:
            return 'Extremely Wet'
        elif spi_value >= 1.5:
            return 'Very Wet'
        elif spi_value >= 1.0:
            return 'Moderately Wet'
        elif spi_value >= 0.5:
            return 'Slightly Wet'
        elif spi_value >= -0.5:
            return 'Near Normal'
        elif spi_value >= -1.0:
            return 'Slightly Dry'
        elif spi_value >= -1.5:
            return 'Moderately Dry'
        elif spi_value >= -2.0:
            return 'Severely Dry'
        else:
            return 'Extremely Dry'
    
    # Add classification columns
    for scale in [1, 3, 6, 12]:
        class_col = f'SPI_{scale}M_Class'
        result_df[class_col] = result_df[f'SPI_{scale}M'].apply(classify_spi)
    
    # Add additional fields
    result_df['Year_Month'] = result_df['Year'].astype(str) + '-' + result_df['Month'].astype(str).str.zfill(2)
    result_df['Date_String'] = result_df['Date'].dt.strftime('%Y-%m-%d')
    result_df['Drought_Intensity'] = result_df['SPI_1M'].apply(lambda x: max(-x, 0) if not pd.isna(x) else 0)
    result_df['Wetness_Intensity'] = result_df['SPI_1M'].apply(lambda x: max(x, 0) if not pd.isna(x) else 0)
    
    # Save main file
    result_df.to_csv(os.path.join(STATIC_DIR, 'spi_tableau_data.csv'), index=False)
    print("SPI data saved")
    
    # Create and save summary statistics
    try:
        summary = result_df.groupby(['City', 'Year']).agg({
            'Rainfall_mm': ['sum', 'mean', 'std'],
            'SPI_1M': ['mean', 'min', 'max'],
            'SPI_3M': ['mean', 'min', 'max'],
            'SPI_6M': ['mean', 'min', 'max'],
            'SPI_12M': ['mean', 'min', 'max']
        }).round(2)
        
        summary.to_csv(os.path.join(STATIC_DIR, 'spi_annual_summary.csv'))
        print("Summary statistics saved")
    except Exception as e:
        print(f"Could not create summary file: {e}")
    
    return result_df

def get_germany_boundaries():
    """Get Germany administrative boundaries as GeoJSON"""
    try:
        url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/germany.geojson"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("Successfully loaded Germany boundaries")
            return response.json()
    except:
        try:
            url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("Successfully loaded Germany boundaries from alternative source")
                return response.json()
        except:
            pass
    
    print("Could not download Germany boundaries. Map will show without country borders.")
    return None

def create_comprehensive_dashboard():
    """Create a comprehensive interactive dashboard in a single HTML file"""
    
    # Check if SPI data exists
    spi_file = os.path.join(STATIC_DIR, 'spi_tableau_data.csv')
    if not os.path.exists(spi_file):
        return "<html><body><h1>Dashboard not available - SPI data not generated yet</h1></body></html>"
    
    # Load data
    df = pd.read_csv(spi_file)
    df['Date'] = pd.to_datetime(df['Date_String'])
    
    print("Creating comprehensive SPI dashboard...")
    print(f"Dataset: {len(df)} records for {len(df['City'].unique())} cities")
    print(f"Time period: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    
    # Get basic info for the dashboard
    cities = sorted(df['City'].unique().tolist())
    years = sorted([int(year) for year in df['Year'].unique()])
    dates = sorted(df['Date'].unique())
    
    # Get Germany boundaries
    germany_geojson = get_germany_boundaries()
    
    # Prepare data for different visualizations
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
    
    # Summary data for different aggregations
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
    
    # Heatmap data
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
    
    # Create comprehensive JavaScript for the interactive dashboard
    dashboard_js = f"""
    <script>
    // Data for the dashboard
    var timeSeriesData = {json.dumps(time_series_data)};
    var summaryData = {json.dumps(summary_data)};
    var heatmapData = {json.dumps(heatmap_data)};
    var cities = {json.dumps(cities)};
    var years = {json.dumps(years)};
    
    // Map instance
    var map;
    var currentMarkers = [];
    var currentHeatmap = null;
    var currentFeatureGroups = [];
    
    // Initialize map
    function initializeMap() {{
        map = L.map('map').setView([51.1657, 10.4515], 6);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add Germany boundaries if available
        {f'L.geoJSON({json.dumps(germany_geojson)}).addTo(map);' if germany_geojson else ''}
    }}
    
    // Color functions
    function getSPIColor(spiValue) {{
        if (spiValue === null || spiValue === undefined) return '#808080';
        if (spiValue >= 2.0) return '#000080';
        if (spiValue >= 1.5) return '#0000CD';
        if (spiValue >= 1.0) return '#4169E1';
        if (spiValue >= 0.5) return '#87CEEB';
        if (spiValue >= -0.5) return '#98FB98';
        if (spiValue >= -1.0) return '#FFE4B5';
        if (spiValue >= -1.5) return '#F4A460';
        if (spiValue >= -2.0) return '#CD5C5C';
        return '#8B0000';
    }}
    
    function classifySPI(spiValue) {{
        if (spiValue === null || spiValue === undefined) return 'No Data';
        if (spiValue >= 2.0) return 'Extremely Wet';
        if (spiValue >= 1.5) return 'Very Wet';
        if (spiValue >= 1.0) return 'Moderately Wet';
        if (spiValue >= 0.5) return 'Slightly Wet';
        if (spiValue >= -0.5) return 'Near Normal';
        if (spiValue >= -1.0) return 'Slightly Dry';
        if (spiValue >= -1.5) return 'Moderately Dry';
        if (spiValue >= -2.0) return 'Severely Dry';
        return 'Extremely Dry';
    }}
    
    // Clear current visualization
    function clearVisualization() {{
        currentMarkers.forEach(marker => map.removeLayer(marker));
        currentMarkers = [];
        
        if (currentHeatmap) {{
            map.removeLayer(currentHeatmap);
            currentHeatmap = null;
        }}
        
        currentFeatureGroups.forEach(fg => map.removeLayer(fg));
        currentFeatureGroups = [];
    }}
    
    // Update visualization based on controls
    function updateVisualization() {{
        clearVisualization();
        
        var vizType = document.getElementById('vizType').value;
        var spiScale = document.getElementById('spiScale').value;
        var timeType = document.getElementById('timeType').value;
        var yearFilter = document.getElementById('yearFilter').value;
        
        if (vizType === 'markers') {{
            if (timeType === 'timeseries') {{
                showTimeSeries(spiScale);
            }} else {{
                showSummary(spiScale, yearFilter);
            }}
        }} else if (vizType === 'heatmap') {{
            showHeatmap(spiScale);
        }}
    }}
    
    // Show time series with layer control
    function showTimeSeries(spiScale) {{
        var data = timeSeriesData[spiScale];
        var layerControl = L.control.layers(null, null, {{collapsed: false}}).addTo(map);
        
        data.forEach((timePoint, index) => {{
            var layerGroup = L.layerGroup();
            var isLastLayer = index === data.length - 1;
            
            timePoint.data.forEach(point => {{
                var color = getSPIColor(point.spi);
                var classification = classifySPI(point.spi);
                var radius = point.spi ? Math.min(Math.max(Math.abs(point.spi) * 2 + 6, 6), 15) : 6;
                
                var popupContent = `
                    <div style="font-family: Arial; font-size: 12px;">
                        <b style="font-size: 14px;">${{point.city}}</b><br>
                        <b>Date:</b> ${{timePoint.date}}<br>
                        <b>SPI-${{spiScale}}:</b> ${{point.spi ? point.spi.toFixed(2) : 'No Data'}}<br>
                        <b>Classification:</b> ${{classification}}<br>
                        <b>Rainfall:</b> ${{point.rainfall ? point.rainfall.toFixed(1) + ' mm' : 'No Data'}}<br>
                        <b>Temperature:</b> ${{point.temperature ? point.temperature.toFixed(1) + '°C' : 'No Data'}}<br>
                        <b>Humidity:</b> ${{point.humidity ? point.humidity.toFixed(1) + '%' : 'No Data'}}<br>
                        <b>Elevation:</b> ${{point.elevation}} m<br>
                        <b>Climate:</b> ${{point.climate}}
                    </div>
                `;
                
                var marker = L.circleMarker([point.lat, point.lng], {{
                    radius: radius,
                    fillColor: color,
                    color: 'black',
                    weight: 1,
                    fillOpacity: 0.8
                }}).bindPopup(popupContent).bindTooltip(`${{point.city}}: ${{classification}}`);
                
                layerGroup.addLayer(marker);
            }});
            
            layerControl.addOverlay(layerGroup, timePoint.date);
            if (isLastLayer) {{
                layerGroup.addTo(map);
            }}
            currentFeatureGroups.push(layerGroup);
        }});
        
        currentFeatureGroups.push(layerControl);
    }}
    
    // Show summary data
    function showSummary(spiScale, yearFilter) {{
        var dataKey = yearFilter === 'all' ? `${{spiScale}}_overall` : `${{spiScale}}_${{yearFilter}}`;
        var data = summaryData[dataKey];
        
        if (!data) return;
        
        data.forEach(point => {{
            var color = getSPIColor(point.spi);
            var classification = classifySPI(point.spi);
            var radius = point.spi ? Math.min(Math.max(Math.abs(point.spi) * 3 + 8, 8), 20) : 8;
            
            var popupContent = `
                <div style="font-family: Arial; font-size: 12px;">
                    <b style="font-size: 14px;">${{point.city}}</b><br>
                    <b>Average SPI-${{spiScale}}:</b> ${{point.spi ? point.spi.toFixed(2) : 'No Data'}}<br>
                    <b>Classification:</b> ${{classification}}<br>
                    <b>Avg Rainfall:</b> ${{point.rainfall ? point.rainfall.toFixed(1) + ' mm' : 'No Data'}}<br>
                    <b>Avg Temperature:</b> ${{point.temperature ? point.temperature.toFixed(1) + '°C' : 'No Data'}}<br>
                    <b>Avg Humidity:</b> ${{point.humidity ? point.humidity.toFixed(1) + '%' : 'No Data'}}<br>
                    <b>Climate:</b> ${{point.climate}}<br>
                    <b>Elevation:</b> ${{point.elevation}} m
                </div>
            `;
            
            var marker = L.circleMarker([point.lat, point.lng], {{
                radius: radius,
                fillColor: color,
                color: 'black',
                weight: 1,
                fillOpacity: 0.8
            }}).bindPopup(popupContent).bindTooltip(`${{point.city}}: ${{classification}}`);
            
            marker.addTo(map);
            currentMarkers.push(marker);
        }});
    }}
    
    // Show heatmap
    function showHeatmap(spiScale) {{
        var data = heatmapData[spiScale];
        
        if (window.L && window.L.heatLayer && window.L.heatLayer.heatmapWithTime) {{
            currentHeatmap = L.heatLayer.heatmapWithTime(data.data, {{
                index: data.index,
                radius: 30,
                blur: 20,
                maxOpacity: 0.8,
                minOpacity: 0.1,
                gradient: {{0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1.0: 'red'}}
            }}).addTo(map);
        }} else {{
            // Fallback: show latest heatmap data as regular heatmap
            if (data.data.length > 0) {{
                var latestData = data.data[data.data.length - 1];
                currentHeatmap = L.heatLayer(latestData, {{
                    radius: 30,
                    blur: 20,
                    maxOpacity: 0.8,
                    minOpacity: 0.1
                }}).addTo(map);
            }}
        }}
    }}
    
    // Initialize everything when page loads
    document.addEventListener('DOMContentLoaded', function() {{
        initializeMap();
        updateVisualization();
        
        // Add event listeners to controls
        document.getElementById('vizType').addEventListener('change', updateVisualization);
        document.getElementById('spiScale').addEventListener('change', updateVisualization);
        document.getElementById('timeType').addEventListener('change', function() {{
            var yearControl = document.getElementById('yearFilter');
            yearControl.style.display = this.value === 'summary' ? 'inline-block' : 'none';
            updateVisualization();
        }});
        document.getElementById('yearFilter').addEventListener('change', updateVisualization);
    }});
    </script>
    """
    
    # Create the complete HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Germany SPI Drought Analysis Dashboard</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 15px;
                text-align: center;
            }}
            .controls {{
                background-color: #ecf0f1;
                padding: 15px;
                border-bottom: 2px solid #bdc3c7;
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: center;
                justify-content: center;
            }}
            .control-group {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 5px;
            }}
            .control-group label {{
                font-weight: bold;
                font-size: 12px;
                color: #2c3e50;
            }}
            .control-group select, .control-group input {{
                padding: 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }}
            #map {{
                height: calc(100vh - 160px);
                width: 100%;
            }}
            .legend {{
                position: fixed;
                top: 160px;
                right: 10px;
                width: 200px;
                background-color: white;
                border: 2px solid grey;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
                z-index: 1000;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 3px 0;
            }}
            .legend-color {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                margin-right: 8px;
                border: 1px solid #333;
            }}
            .info-panel {{
                position: fixed;
                bottom: 10px;
                left: 10px;
                background-color: white;
                border: 2px solid grey;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
                z-index: 1000;
                max-width: 300px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Germany SPI Drought Analysis Dashboard</h1>
            <p>Interactive visualization of Standardized Precipitation Index across German cities ({df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')})</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="vizType">Visualization Type:</label>
                <select id="vizType">
                    <option value="markers">City Markers</option>
                    <option value="heatmap">Heatmap</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="spiScale">SPI Time Scale:</label>
                <select id="spiScale">
                    <option value="1M">SPI-1M (Monthly)</option>
                    <option value="3M">SPI-3M (Seasonal)</option>
                    <option value="6M">SPI-6M (Half-yearly)</option>
                    <option value="12M" selected>SPI-12M (Annual)</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="timeType">Time Analysis:</label>
                <select id="timeType">
                    <option value="timeseries">Time Series (All Months)</option>
                    <option value="summary" selected>Summary (Average)</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="yearFilter">Year Filter:</label>
                <select id="yearFilter" style="display: inline-block;">
                    <option value="all" selected>All Years Average</option>
                    {chr(10).join([f'<option value="{year}">{year}</option>' for year in years])}
                </select>
            </div>
        </div>
        
        <div id="map"></div>
        
        <div class="legend">
            <p style="margin-top:0; font-weight: bold;">SPI Classification</p>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #000080;"></div>
                <span>Extremely Wet (≥ 2.0)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #0000CD;"></div>
                <span>Very Wet (1.5 to 2.0)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #4169E1;"></div>
                <span>Moderately Wet (1.0 to 1.5)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #87CEEB;"></div>
                <span>Slightly Wet (0.5 to 1.0)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #98FB98;"></div>
                <span>Near Normal (-0.5 to 0.5)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FFE4B5;"></div>
                <span>Slightly Dry (-1.0 to -0.5)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #F4A460;"></div>
                <span>Moderately Dry (-1.5 to -1.0)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #CD5C5C;"></div>
                <span>Severely Dry (-2.0 to -1.5)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #8B0000;"></div>
                <span>Extremely Dry (< -2.0)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #808080;"></div>
                <span>No Data</span>
            </div>
            <p style="margin-top: 10px; font-size: 10px; font-style: italic;">
                Marker size indicates SPI magnitude
            </p>
        </div>
        
        <div class="info-panel">
            <p style="margin-top: 0; font-weight: bold;">Dataset Information</p>
            <p><strong>Cities:</strong> {len(cities)} German cities</p>
            <p><strong>Time Period:</strong> {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}</p>
            <p><strong>Total Records:</strong> {len(df):,}</p>
            <p><strong>SPI Scales:</strong> 1M, 3M, 6M, 12M</p>
            <p style="font-size: 10px; font-style: italic; margin-bottom: 0;">
                Click markers for detailed information. Use controls above to change visualization.
            </p>
        </div>
        
        {dashboard_js}
    </body>
    </html>
    """
    
    return html_content

def generate_all_visualizations():
    """Generate all visualizations when the Flask app starts"""
    print("=" * 50)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Step 1: Process rainfall data
        df_rainfall = process_rainfall_data()
        
        # Step 2: Create stacked bar chart
        create_stacked_bar_chart()
        
        # Step 3: Create seasonal charts
        create_seasonal_charts()
        
        # Step 4: Calculate SPI values
        result_df = calculate_spi()
        
        # Step 5: Create interactive dashboard
        print("Creating interactive dashboard...")
        html_content = create_comprehensive_dashboard()
        
        # Save dashboard
        dashboard_file = os.path.join(STATIC_DIR, 'germany_spi_comprehensive_dashboard.html')
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to {dashboard_file}")
        
        print("=" * 50)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 50)
        
        # List generated files
        files = os.listdir(STATIC_DIR)
        print(f"Generated files in {STATIC_DIR}:")
        for file in sorted(files):
            print(f"  - {file}")
        
        return True
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

# Generate visualizations when the Flask app starts
generation_success = generate_all_visualizations()

# Flask routes
@app.route('/')
def home():
    # Get list of available files
    image_files = []
    csv_files = []
    html_files = []
    
    if os.path.exists(STATIC_DIR):
        files = os.listdir(STATIC_DIR)
        image_files = [f for f in files if f.endswith('.png')]
        csv_files = [f for f in files if f.endswith('.csv')]
        html_files = [f for f in files if f.endswith('.html')]
    
    status_msg = "✅ All visualizations generated successfully!" if generation_success else "❌ Error generating some visualizations"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rainfall Analysis & Drought Assessment</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .status {{ padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .success {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .error {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
            .section {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }}
            .btn {{ display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
            .btn:hover {{ background-color: #0056b3; }}
            .btn-success {{ background-color: #28a745; }}
            .btn-success:hover {{ background-color: #218838; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin: 8px 0; }}
            li a {{ text-decoration: none; color: #007bff; }}
            li a:hover {{ color: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌧️ Rainfall Analysis & Drought Assessment</h1>
            <h2>German Cities Climate Data Analysis</h2>
            
            <div class="status {'success' if generation_success else 'error'}">
                <strong>Status:</strong> {status_msg}
            </div>
            
            <div class="section">
                <h3>🌍 Interactive Dashboard</h3>
                <p>Explore the comprehensive SPI drought analysis with our interactive map:</p>
                <a href="/dashboard" target="_blank" class="btn btn-success">🚀 Open Interactive Dashboard</a>
            </div>
            
            <div class="section">
                <h3>📊 Generated Visualizations ({len(image_files)} charts)</h3>
                <ul>
                    {''.join([f'<li>🖼️ <a href="/images/{img}" target="_blank">{img}</a></li>' for img in sorted(image_files)])}
                </ul>
            </div>
            
            <div class="section">
                <h3>📋 Data Files ({len(csv_files)} datasets)</h3>
                <ul>
                    {''.join([f'<li>📄 <a href="/download/{csv}" target="_blank">{csv}</a></li>' for csv in sorted(csv_files)])}
                </ul>
            </div>
            
            <div class="section">
                <h3>🔗 API Endpoints</h3>
                <ul>
                    <li>📊 <a href="/data">Processed SPI Data Summary</a></li>
                    <li>📈 <a href="/summary">Statistical Summary</a></li>
                    <li>🏙️ <a href="/cities">Cities Analyzed</a></li>
                    <li>📁 <a href="/files">All Generated Files</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h3>📖 About This Analysis</h3>
                <p><strong>Dataset:</strong> Rainfall data for major German cities from 2000-2024</p>
                <p><strong>Analysis:</strong> Standardized Precipitation Index (SPI) calculation for drought assessment</p>
                <p><strong>Time Scales:</strong> 1-month, 3-month, 6-month, and 12-month SPI values</p>
                <p><strong>Cities:</strong> Berlin, Cologne, Dresden, Düsseldorf, Frankfurt, Hamburg, Hanover, Leipzig, Munich, Stuttgart</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/dashboard')
def dashboard():
    """Serve the interactive dashboard"""
    try:
        return send_from_directory(STATIC_DIR, 'germany_spi_comprehensive_dashboard.html')
    except FileNotFoundError:
        return """
        <h1>Dashboard not found</h1>
        <p>The dashboard is being generated. Please wait a moment and refresh the page.</p>
        <p><a href="/">← Back to Home</a></p>
        """, 404

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve generated images"""
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        return f"Image {filename} not found.", 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download CSV or other data files"""
    try:
        return send_from_directory(STATIC_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return f"File {filename} not found.", 404

@app.route('/files')
def list_files():
    """List all generated files"""
    files = []
    if os.path.exists(STATIC_DIR):
        files = os.listdir(STATIC_DIR)
    
    file_info = []
    total_size = 0
    
    for file in sorted(files):
        file_path = os.path.join(STATIC_DIR, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            size_mb = size / (1024 * 1024)
            
            if file.endswith('.png'):
                file_info.append(f'<li>🖼️ <a href="/images/{file}" target="_blank">{file}</a> <small>({size_mb:.2f} MB)</small></li>')
            elif file.endswith('.html'):
                file_info.append(f'<li>🌍 <a href="/dashboard" target="_blank">{file}</a> <small>({size_mb:.2f} MB)</small></li>')
            elif file.endswith('.csv'):
                file_info.append(f'<li>📊 <a href="/download/{file}">{file}</a> <small>({size_mb:.2f} MB)</small></li>')
            else:
                file_info.append(f'<li>📄 {file} <small>({size_mb:.2f} MB)</small></li>')
    
    total_size_mb = total_size / (1024 * 1024)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Generated Files</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-radius: 4px; }}
            small {{ color: #666; }}
            .summary {{ background-color: #e9ecef; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📁 Generated Files</h1>
            
            <div class="summary">
                <strong>Summary:</strong> {len(files)} files generated, {total_size_mb:.2f} MB total
            </div>
            
            <ul>
                {''.join(file_info) if file_info else '<li>No files generated yet.</li>'}
            </ul>
            
            <p><a href="/">← Back to Home</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/data')
def get_data():
    """Return processed SPI data summary"""
    spi_file = os.path.join(STATIC_DIR, 'spi_tableau_data.csv')
    
    if os.path.exists(spi_file):
        df = pd.read_csv(spi_file)
        
        return jsonify({
            "message": "SPI Data Analysis Complete",
            "cities": sorted(df['City'].unique().tolist()),
            "total_records": len(df),
            "time_period": {
                "start": df['Year'].min(),
                "end": df['Year'].max()
            },
            "spi_scales": ["1M", "3M", "6M", "12M"],
            "analysis_complete": True,
            "available_visualizations": [f for f in os.listdir(STATIC_DIR) if f.endswith('.png')] if os.path.exists(STATIC_DIR) else [],
            "dashboard_available": os.path.exists(os.path.join(STATIC_DIR, 'germany_spi_comprehensive_dashboard.html'))
        })
    else:
        return jsonify({
            "message": "SPI data not yet generated",
            "analysis_complete": False
        })

@app.route('/summary')
def get_summary():
    """Return drought statistics summary"""
    spi_file = os.path.join(STATIC_DIR, 'spi_tableau_data.csv')
    
    if os.path.exists(spi_file):
        df = pd.read_csv(spi_file)
        
        # Calculate drought conditions (SPI-12M < -1.0)
        drought_summary = df[df['SPI_12M'] < -1.0].groupby('City').size().sort_values(ascending=False)
        
        # Calculate SPI statistics
        spi_stats = {}
        for scale in ['1M', '3M', '6M', '12M']:
            col = f'SPI_{scale}'
            if col in df.columns:
                spi_stats[f'SPI_{scale}'] = {
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'drought_months': int((df[col] < -1.0).sum()) if not df[col].isna().all() else 0,
                    'wet_months': int((df[col] > 1.0).sum()) if not df[col].isna().all() else 0
                }
        
        return jsonify({
            "drought_conditions_by_city": drought_summary.to_dict(),
            "spi_statistics": spi_stats,
            "total_cities": len(df['City'].unique()),
            "total_records": len(df),
            "analysis_period": f"{df['Year'].min()}-{df['Year'].max()}"
        })
    else:
        return jsonify({
            "message": "Summary data not available - analysis not complete",
            "drought_conditions_by_city": {}
        })

@app.route('/cities')
def get_cities():
    """Return list of cities analyzed"""
    spi_file = os.path.join(STATIC_DIR, 'spi_tableau_data.csv')
    
    if os.path.exists(spi_file):
        df = pd.read_csv(spi_file)
        cities_info = []
        
        for city in sorted(df['City'].unique()):
            city_data = df[df['City'] == city].iloc[0]
            cities_info.append({
                "name": city,
                "latitude": float(city_data['Latitude']),
                "longitude": float(city_data['Longitude']),
                "climate_type": city_data['Climate_Type'],
                "elevation": int(city_data['Elevation_m']),
                "records": len(df[df['City'] == city])
            })
        
        return jsonify({
            "cities": cities_info,
            "count": len(cities_info)
        })
    else:
        cities = ["Berlin", "Cologne", "Dresden", "Dusseldorf", "Frankfurt", 
                  "Hamburg", "Hanover", "Leipzig", "Munich", "Stuttgart"]
        return jsonify({
            "cities": cities, 
            "count": len(cities),
            "note": "Default city list - detailed analysis not yet complete"
        })

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)