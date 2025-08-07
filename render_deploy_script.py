#!/usr/bin/env python3
"""
Deployment preparation script for SPI Dashboard on Render.com
This script processes the raw rainfall data and prepares it for deployment
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
import os
import sys

warnings.filterwarnings('ignore')

def process_spi_data():
    """Process SPI data from raw rainfall data"""
    print("🔄 Processing SPI data for deployment...")
    
    # Check if input file exists
    if not os.path.exists('Rainfall_Cleaned.csv'):
        print("❌ Error: 'Rainfall_Cleaned.csv' not found!")
        print("Please ensure your cleaned rainfall data file is in the same directory.")
        return False
    
    try:
        # Load data
        df = pd.read_csv('Rainfall_Cleaned.csv')
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
        df = df.sort_values(['City', 'Date'])
        
        print(f"📊 Loaded data: {len(df)} records for {len(df['City'].unique())} cities")
        
        # Initialize results list to store all data
        all_results = []
        
        # Time scales to calculate (in months)
        time_scales = [1, 3, 6, 12]
        
        # Get all cities
        cities = df['City'].unique()
        
        # Calculate SPI for each city
        for i, city in enumerate(cities, 1):
            print(f"📍 Processing {city}... ({i}/{len(cities)})")
            city_data = df[df['City'] == city].copy()
            
            # Ensure continuous monthly sequence
            city_data = city_data.set_index('Date')
            
            # Get city metadata (constant values)
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
                
                if len(valid_data) < 30:  # Need at least 30 data points for reliable fitting
                    spi_values = pd.Series(np.nan, index=city_data.index)
                else:
                    # Fit gamma distribution to the precipitation data
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
                                # For zero precipitation, use the probability of zero
                                prob = prob_zero
                                if prob >= 0.9999:
                                    prob = 0.9999
                                if prob <= 0.0001:
                                    prob = 0.0001
                                spi_list.append(stats.norm.ppf(prob))
                            else:
                                # For positive precipitation, use gamma CDF adjusted for probability of zero
                                prob = prob_zero + (1 - prob_zero) * stats.gamma.cdf(value, alpha, loc=loc, scale=beta)
                                # Ensure probability is within valid range
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
                    'Temperature_C': city_data['Temperature (??C)'].iloc[idx],
                    'Humidity_pct': city_data['Humidity (%)'].iloc[idx],
                    'SPI_1M': spi_results['SPI_1M'].iloc[idx],
                    'SPI_3M': spi_results['SPI_3M'].iloc[idx],
                    'SPI_6M': spi_results['SPI_6M'].iloc[idx],
                    'SPI_12M': spi_results['SPI_12M'].iloc[idx]
                }
                all_results.append(row_data)
        
        # Create final dataframe
        result_df = pd.DataFrame(all_results)
        
        # Add SPI classification columns
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
        
        for scale in [1, 3, 6, 12]:
            class_col = f'SPI_{scale}M_Class'
            result_df[class_col] = result_df[f'SPI_{scale}M'].apply(classify_spi)
        
        # Add additional fields
        result_df['Year_Month'] = result_df['Year'].astype(str) + '-' + result_df['Month'].astype(str).str.zfill(2)
        result_df['Date_String'] = result_df['Date'].dt.strftime('%Y-%m-%d')
        result_df['Drought_Intensity'] = result_df['SPI_1M'].apply(lambda x: max(-x, 0) if not pd.isna(x) else 0)
        result_df['Wetness_Intensity'] = result_df['SPI_1M'].apply(lambda x: max(x, 0) if not pd.isna(x) else 0)
        
        # Save processed data
        result_df.to_csv('spi_tableau_data.csv', index=False)
        
        print(f"✅ SPI data processing complete!")
        print(f"📁 Output saved as 'spi_tableau_data.csv'")
        print(f"📊 Final dataset: {len(result_df):,} records")
        print(f"🏙️  Cities: {len(result_df['City'].unique())} cities")
        print(f"📅 Date range: {result_df['Date'].min().strftime('%Y-%m')} to {result_df['Date'].max().strftime('%Y-%m')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        return False

def check_deployment_files():
    """Check if all required deployment files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'spi_tableau_data.csv',
        'templates/index.html'
    ]
    
    print("\n🔍 Checking deployment files...")
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Please create these files before deploying to Render.com")
        return False
    else:
        print("\n✅ All deployment files present!")
        return True

def create_project_structure():
    """Create the required project structure"""
    print("\n📁 Creating project structure...")
    
    # Create directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("✅ Project directories created")

def main():
    """Main deployment preparation function"""
    print("🚀 SPI Dashboard - Render.com Deployment Preparation")
    print("=" * 60)
    
    # Create project structure
    create_project_structure()
    
    # Process SPI data
    if not process_spi_data():
        print("❌ Data processing failed. Cannot proceed with deployment.")
        return False
    
    # Check deployment files
    if not check_deployment_files():
        print("\n❌ Deployment files check failed.")
        print("Please ensure all required files are present before deploying.")
        return False
    
    print("\n🎉 Deployment preparation complete!")
    print("\n📋 Next steps for Render.com deployment:")
    print("1. Create a GitHub repository and upload these files")
    print("2. Connect your GitHub repository to Render.com")
    print("3. Configure the web service with these settings:")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: gunicorn app:app")
    print("   - Environment: Python 3")
    print("\n🌐 Your dashboard will be available at: https://your-app-name.onrender.com")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)