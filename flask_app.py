from flask import Flask, render_template, jsonify, send_file, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from scipy import stats
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

def generate_stacked_bar_chart():
    """Generate the stacked bar chart you want"""
    print("🔄 Generating stacked bar chart...")
    
    try:
        # Read and process data
        df_cleaned = pd.read_csv('Rainfall_Data_Germany_Complete.csv')
        
        # Calculate annual rainfall per city
        city_annual = df_cleaned.groupby(['City', 'Year', 'Climate_Type'])['Rainfall (mm)'].sum().reset_index()
        
        # Calculate total rainfall per climate type per year (for stacked bar)
        climate_total = city_annual.groupby(['Climate_Type', 'Year'])['Rainfall (mm)'].sum().reset_index()
        
        # Pivot the data for stacked bar chart
        climate_pivot = climate_total.pivot(index='Year', columns='Climate_Type', values='Rainfall (mm)').fillna(0)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        climate_pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        
        # Add value labels on bars (only for bars with significant height)
        for i, climate in enumerate(climate_pivot.columns):
            for j, value in enumerate(climate_pivot[climate]):
                if value > 1000:  # Only label if rainfall > 1000mm
                    # Calculate position
                    y_pos = sum(climate_pivot.iloc[j, :i]) + value/2
                    ax.text(j, y_pos, f'{value:,.0f}',
                            ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Prepare data for the trend line (total rainfall per year)
        total_by_year = climate_pivot.sum(axis=1).values  # Sum of all climate types for each year
        x = climate_pivot.index.values  # Years (x-axis)
        
        # Total rainfall per year
        total_by_year = climate_pivot.sum(axis=1).values
        
        # Use integer x positions for bar chart
        x_pos = np.arange(len(climate_pivot.index))
        
        # Fit and plot trend line
        z = np.polyfit(x_pos, total_by_year, 1)
        p = np.poly1d(z)
        ax.plot(x_pos, p(x_pos), "k--", alpha=0.7, linewidth=2, label='Trend')
        
        # Customize plot
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Annual Rainfall (mm)')
        ax.set_title('Total Annual Rainfall by Climate Type')
        ax.legend(title='Climate Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save outputs to static directory
        plt.savefig(os.path.join(STATIC_DIR, 'rainfall_stacked_bar_simplified.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        # Export data
        climate_pivot.to_csv(os.path.join(STATIC_DIR, 'rainfall_stacked_data.csv'))
        
        print("✅ Stacked bar chart generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating stacked bar chart: {e}")
        return False

def generate_seasonal_charts():
    """Generate the seasonal rainfall charts you want"""
    print("🔄 Generating seasonal charts...")
    
    try:
        # Load data
        df = pd.read_csv('Rainfall_Data_Germany_Complete.csv')
        
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
        
        # Calculate overall mean and std across all climate types for comparison
        pivot = final_data.pivot_table(index='Season', columns='Year', values='Avg_Rainfall (mm)', aggfunc='mean').reindex(season_order)
        mean_rainfall = pivot.mean(axis=1)
        std_rainfall = pivot.std(axis=1)
        
        # Plot for each climate type
        for i, climate in enumerate(climate_types):
            plt.figure(figsize=(12, 8))
            climate_data = final_data[final_data['Climate_Type'] == climate]
            
            # Add shaded region for ±1 Std Dev (calculated from overall data)
            plt.fill_between(season_order,
                             mean_rainfall - std_rainfall,
                             mean_rainfall + std_rainfall,
                             color='gray', alpha=0.2, label='±1 Std Dev (Overall)')
            
            # Add mean line (calculated from overall data)
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
            
            # Adjust the legend
            plt.legend(
                title='Year',
                fontsize='small',
                loc='upper left',
                bbox_to_anchor=(1.05, 1),
                title_fontsize='large'
            )
            
            plt.tight_layout()
            
            # Save as PNG to static directory
            plt.savefig(os.path.join(STATIC_DIR, f'{climate}_climate_rainfall.png'), dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
        
        print("✅ Seasonal charts generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating seasonal charts: {e}")
        return False

def create_simple_dashboard():
    """Create a simple interactive dashboard"""
    print("🔄 Creating interactive dashboard...")
    
    try:
        # Simple HTML dashboard
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Germany Rainfall Analysis Dashboard</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    color: #2c3e50;
                    margin-bottom: 30px;
                }
                .chart-section {
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }
                .chart-section h3 {
                    color: #34495e;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                .chart-section img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .description {
                    margin-top: 15px;
                    color: #666;
                    line-height: 1.6;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌧️ Germany Rainfall Analysis Dashboard</h1>
                    <p>Interactive visualization of rainfall patterns and drought analysis across German cities (2000-2024)</p>
                </div>
                
                <div class="chart-section">
                    <h3>📊 Annual Rainfall by Climate Type (Stacked Bar Chart)</h3>
                    <img src="/images/rainfall_stacked_bar_simplified.png" alt="Stacked Bar Chart">
                    <div class="description">
                        <p>This chart shows the total annual rainfall across different climate types in Germany. 
                        The stacked bars represent the contribution of each climate type to the total rainfall, 
                        with a trend line showing the overall pattern over time.</p>
                    </div>
                </div>
                
                <div class="chart-section">
                    <h3>🌍 Seasonal Rainfall Patterns by Climate Type</h3>
                    <p>Below are the seasonal rainfall patterns for different climate types:</p>
                    
                    <h4>Continental Climate</h4>
                    <img src="/images/Continental_climate_rainfall.png" alt="Continental Climate Chart">
                    <div class="description">
                        <p>Seasonal rainfall patterns for continental climate regions, showing variations across 
                        winter, spring, summer, and autumn seasons with year-by-year comparisons.</p>
                    </div>
                    
                    <h4>Oceanic Climate</h4>
                    <img src="/images/Oceanic_climate_rainfall.png" alt="Oceanic Climate Chart">
                    <div class="description">
                        <p>Seasonal rainfall patterns for oceanic climate regions, demonstrating the influence 
                        of maritime conditions on precipitation patterns throughout the year.</p>
                    </div>
                </div>
                
                <div class="chart-section">
                    <h3>📈 Dataset Information</h3>
                    <ul>
                        <li><strong>Cities Analyzed:</strong> Berlin, Cologne, Dresden, Düsseldorf, Frankfurt, Hamburg, Hanover, Leipzig, Munich, Stuttgart</li>
                        <li><strong>Time Period:</strong> 2000-2024</li>
                        <li><strong>Data Points:</strong> Monthly rainfall, temperature, and humidity measurements</li>
                        <li><strong>Climate Types:</strong> Continental and Oceanic</li>
                    </ul>
                </div>
                
                <div class="chart-section">
                    <h3>🔗 Navigation</h3>
                    <p><a href="/">← Back to Home</a> | <a href="/files">View All Files</a> | <a href="/data">API Data</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard to static directory
        dashboard_file = os.path.join(STATIC_DIR, 'germany_spi_comprehensive_dashboard.html')
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ Dashboard created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        return False

def generate_all_visualizations():
    """Generate all three main visualizations"""
    print("=" * 60)
    print("🚀 STARTING VISUALIZATION GENERATION")
    print("=" * 60)
    
    success_count = 0
    
    # Generate stacked bar chart
    if generate_stacked_bar_chart():
        success_count += 1
    
    # Generate seasonal charts
    if generate_seasonal_charts():
        success_count += 1
    
    # Create dashboard
    if create_simple_dashboard():
        success_count += 1
    
    print("=" * 60)
    if success_count == 3:
        print("🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("✅ Stacked Bar Chart: rainfall_stacked_bar_simplified.png")
        print("✅ Seasonal Charts: [Climate]_climate_rainfall.png")
        print("✅ Interactive Dashboard: germany_spi_comprehensive_dashboard.html")
    else:
        print(f"⚠️  Only {success_count}/3 visualizations generated successfully")
    
    print("=" * 60)
    
    # List generated files
    if os.path.exists(STATIC_DIR):
        files = os.listdir(STATIC_DIR)
        print(f"📁 Generated files in {STATIC_DIR}:")
        for file in sorted(files):
            print(f"   📄 {file}")
    
    return success_count == 3

# Generate visualizations when the Flask app starts
print("🔄 Flask app starting - generating visualizations...")
generation_success = generate_all_visualizations()

# Flask routes
@app.route('/')
def home():
    """Enhanced homepage showing all generated visualizations"""
    # Get list of available files
    image_files = []
    csv_files = []
    
    if os.path.exists(STATIC_DIR):
        files = os.listdir(STATIC_DIR)
        image_files = [f for f in files if f.endswith('.png')]
        csv_files = [f for f in files if f.endswith('.csv')]
    
    status_msg = "✅ All 3 key visualizations generated successfully!" if generation_success else "❌ Some visualizations failed to generate"
    status_class = "success" if generation_success else "error"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌧️ Rainfall Analysis & Drought Assessment</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{ 
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white; 
                padding: 40px 30px; 
                text-align: center; 
            }}
            .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
            .header p {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 1.2em; }}
            .content {{ padding: 30px; }}
            .status {{ 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
                font-weight: bold;
                text-align: center;
            }}
            .success {{ 
                background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
                color: white;
            }}
            .error {{ 
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                color: white;
            }}
            .section {{ 
                background: #f8f9fa; 
                padding: 25px; 
                margin: 25px 0; 
                border-radius: 12px; 
                border-left: 5px solid #3498db;
            }}
            .section h3 {{ 
                color: #2c3e50; 
                margin-top: 0; 
                font-size: 1.3em;
            }}
            .btn {{ 
                display: inline-block; 
                padding: 12px 25px; 
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white; 
                text-decoration: none; 
                border-radius: 25px; 
                margin: 8px; 
                transition: transform 0.3s;
                font-weight: 600;
            }}
            .btn:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
            }}
            .btn-dashboard {{ 
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                font-size: 1.1em;
                padding: 15px 30px;
            }}
            .file-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 15px; 
                margin: 20px 0;
            }}
            .file-item {{ 
                background: white; 
                padding: 15px; 
                border-radius: 8px; 
                border: 1px solid #ddd;
                transition: box-shadow 0.3s;
            }}
            .file-item:hover {{ 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .stats {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin: 20px 0;
            }}
            .stat-card {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center;
                border: 1px solid #ddd;
            }}
            .stat-number {{ 
                font-size: 2em; 
                font-weight: bold; 
                color: #3498db;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌧️ Rainfall Analysis & Drought Assessment</h1>
                <p>German Cities Climate Data Analysis Dashboard</p>
            </div>
            
            <div class="content">
                <div class="status {status_class}">
                    🎯 Status: {status_msg}
                </div>
                
                <div class="section">
                    <h3>🚀 Interactive Dashboard</h3>
                    <p>Explore your generated visualizations in our comprehensive dashboard:</p>
                    <a href="/dashboard" target="_blank" class="btn btn-dashboard">🌍 Open Interactive Dashboard</a>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{len(image_files)}</div>
                        <div>Charts Generated</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(csv_files)}</div>
                        <div>Data Files</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">10</div>
                        <div>German Cities</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">24</div>
                        <div>Years of Data</div>
                    </div>
                </div>
                
                <div class="section">
                    <h3>📊 Your Generated Visualizations</h3>
                    <div class="file-grid">
                        {''.join([f'<div class="file-item">🖼️ <a href="/images/{img}" target="_blank">{img}</a></div>' for img in sorted(image_files)])}
                    </div>
                    {f'<p style="color: #e74c3c;">⚠️ No visualizations found. Check if data file exists.</p>' if not image_files else ''}
                </div>
                
                <div class="section">
                    <h3>📋 Data Downloads</h3>
                    <div class="file-grid">
                        {''.join([f'<div class="file-item">📄 <a href="/download/{csv}" target="_blank">{csv}</a></div>' for csv in sorted(csv_files)])}
                    </div>
                </div>
                
                <div class="section">
                    <h3>🔗 API Endpoints & Tools</h3>
                    <a href="/data" class="btn">📊 Data Summary</a>
                    <a href="/summary" class="btn">📈 Statistics</a>
                    <a href="/cities" class="btn">🏙️ Cities Info</a>
                    <a href="/files" class="btn">📁 All Files</a>
                </div>
                
                <div class="section">
                    <h3>📖 About This Analysis</h3>
                    <p><strong>🎯 Focus:</strong> The 3 key visualizations you requested:</p>
                    <ul>
                        <li><strong>Stacked Bar Chart:</strong> Annual rainfall by climate type with trend analysis</li>
                        <li><strong>Seasonal Charts:</strong> Climate-specific seasonal rainfall patterns</li>
                        <li><strong>Interactive Dashboard:</strong> Comprehensive visualization interface</li>
                    </ul>
                    <p><strong>📍 Coverage:</strong> Berlin, Cologne, Dresden, Düsseldorf, Frankfurt, Hamburg, Hanover, Leipzig, Munich, Stuttgart</p>
                    <p><strong>⏰ Period:</strong> 2000-2024 (24 years of climate data)</p>
                </div>
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
        <div style="text-align: center; padding: 50px; font-family: Arial;">
            <h1>🔄 Dashboard Loading</h1>
            <p>The dashboard is being generated. Please wait a moment and refresh the page.</p>
            <p><a href="/" style="color: #3498db;">← Back to Home</a></p>
        </div>
        """, 404

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve generated images"""
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        return f"Image {filename} not found. Please check if visualization generation completed successfully.", 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download CSV or other data files"""
    try:
        return send_from_directory(STATIC_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return f"File {filename} not found.", 404

@app.route('/files')
def list_files():
    """List all generated files with details"""
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
                file_info.append(f'<tr><td>🖼️</td><td><a href="/images/{file}" target="_blank">{file}</a></td><td>{size_mb:.2f} MB</td><td>Image</td></tr>')
            elif file.endswith('.html'):
                file_info.append(f'<tr><td>🌍</td><td><a href="/dashboard" target="_blank">{file}</a></td><td>{size_mb:.2f} MB</td><td>Dashboard</td></tr>')
            elif file.endswith('.csv'):
                file_info.append(f'<tr><td>📊</td><td><a href="/download/{file}">{file}</a></td><td>{size_mb:.2f} MB</td><td>Data</td></tr>')
            else:
                file_info.append(f'<tr><td>📄</td><td>{file}</td><td>{size_mb:.2f} MB</td><td>Other</td></tr>')
    
    total_size_mb = total_size / (1024 * 1024)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>📁 Generated Files</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            .summary {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .btn {{ display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📁 Generated Files</h1>
            
            <div class="summary">
                <strong>📊 Summary:</strong> {len(files)} files generated | 💾 Total size: {total_size_mb:.2f} MB
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>File Name</th>
                        <th>Size</th>
                        <th>Category</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(file_info) if file_info else '<tr><td colspan="4">No files generated yet.</td></tr>'}
                </tbody>
            </table>
            
            <a href="/" class="btn">← Back to Home</a>
        </div>
    </body>
    </html>
    """

@app.route('/data')
def get_data():
    """Return comprehensive data summary"""
    if os.path.exists(STATIC_DIR):
        files = os.listdir(STATIC_DIR)
        image_files = [f for f in files if f.endswith('.png')]
        csv_files = [f for f in files if f.endswith('.csv')]
        html_files = [f for f in files if f.endswith('.html')]
    else:
        image_files = []
        csv_files = []
        html_files = []
    
    return jsonify({
        "message": "Rainfall Analysis Complete - Your 3 Key Visualizations",
        "generation_status": generation_success,
        "visualizations_generated": {
            "stacked_bar_chart": "rainfall_stacked_bar_simplified.png" in image_files,
            "seasonal_charts": any("climate_rainfall.png" in f for f in image_files),
            "dashboard": "germany_spi_comprehensive_dashboard.html" in html_files
        },
        "cities": ["Berlin", "Cologne", "Dresden", "Dusseldorf", "Frankfurt", 
                   "Hamburg", "Hanover", "Leipzig", "Munich", "Stuttgart"],
        "total_files_generated": len(image_files) + len(csv_files) + len(html_files),
        "available_charts": image_files,
        "available_data": csv_files,
        "dashboard_available": len(html_files) > 0,
        "urls": {
            "dashboard": "/dashboard",
            "stacked_chart": "/images/rainfall_stacked_bar_simplified.png",
            "all_files": "/files"
        },
        "analysis_complete": True
    })

@app.route('/summary')
def get_summary():
    """Return analysis summary"""
    return jsonify({
        "analysis_focus": "3 Key Visualizations for German Rainfall Analysis",
        "visualizations": {
            "1": "Stacked Bar Chart - Annual rainfall by climate type",
            "2": "Seasonal Charts - Climate-specific patterns",
            "3": "Interactive Dashboard - Comprehensive view"
        },
        "dataset_info": {
            "cities": 10,
            "years": "2000-2024",
            "data_points": "Monthly rainfall, temperature, humidity",
            "climate_types": ["Continental", "Oceanic"]
        },
        "files_generated": len(os.listdir(STATIC_DIR)) if os.path.exists(STATIC_DIR) else 0,
        "status": "✅ Ready for deployment" if generation_success else "⚠️ Generation issues detected"
    })

@app.route('/cities')
def get_cities():
    """Return cities information"""
    cities = [
        {"name": "Berlin", "climate": "Continental"},
        {"name": "Cologne", "climate": "Oceanic"},
        {"name": "Dresden", "climate": "Continental"},
        {"name": "Dusseldorf", "climate": "Oceanic"},
        {"name": "Frankfurt", "climate": "Continental"},
        {"name": "Hamburg", "climate": "Oceanic"},
        {"name": "Hanover", "climate": "Continental"},
        {"name": "Leipzig", "climate": "Continental"},
        {"name": "Munich", "climate": "Continental"},
        {"name": "Stuttgart", "climate": "Continental"}
    ]
    
    return jsonify({
        "cities": cities,
        "count": len(cities),
        "climate_distribution": {
            "Continental": 7,
            "Oceanic": 3
        },
        "analysis_period": "2000-2024"
    })

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    print(f"📊 Generation success: {generation_success}")
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)