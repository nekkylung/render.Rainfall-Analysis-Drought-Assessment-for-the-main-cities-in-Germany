from flask import Flask, send_from_directory, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # CRITICAL for server deployment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Create Flask app
app = Flask(__name__)

print("🚀 Flask app starting...")

def create_static_dir():
    """Create static directory for charts"""
    if not os.path.exists('static'):
        os.makedirs('static')
        print("📁 Created static directory")
    return 'static'

def generate_chart_1():
    """Generate Chart 1: rainfall_stacked_bar_simplified.png"""
    print("📊 Generating Chart 1: Stacked Bar Chart")
    
    try:
        # Load data
        df = pd.read_csv('Rainfall_Data_Germany_Complete.csv')
        print(f"✅ Data loaded: {len(df)} rows")
        
        # Create annual data grouped by climate type
        annual_data = df.groupby(['Year', 'Climate_Type'])['Rainfall (mm)'].sum().unstack(fill_value=0)
        
        # Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        annual_data.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        
        # Add labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Annual Rainfall (mm)')
        ax.set_title('Total Annual Rainfall by Climate Type')
        ax.legend(title='Climate Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add trend line
        total_by_year = annual_data.sum(axis=1).values
        x_pos = np.arange(len(annual_data.index))
        z = np.polyfit(x_pos, total_by_year, 1)
        p = np.poly1d(z)
        ax.plot(x_pos, p(x_pos), "k--", alpha=0.7, linewidth=2, label='Trend')
        
        # Save the chart
        static_dir = create_static_dir()
        chart_path = os.path.join(static_dir, 'rainfall_stacked_bar_simplified.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close to free memory
        
        # Export data
        annual_data.to_csv(os.path.join(static_dir, 'rainfall_stacked_data.csv'))
        
        print(f"✅ Chart 1 saved: {chart_path}")
        return True
        
    except Exception as e:
        print(f"❌ Chart 1 failed: {e}")
        return False

def generate_chart_2():
    """Generate Chart 2: Climate rainfall charts"""
    print("📊 Generating Chart 2: Climate Rainfall Charts")
    
    try:
        # Load data
        df = pd.read_csv('Rainfall_Data_Germany_Complete.csv')
        
        # Map months to seasons
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'
        }
        df['Season'] = df['Month'].map(season_map)
        
        # Calculate seasonal averages by climate type
        climate_types = df['Climate_Type'].unique()
        static_dir = create_static_dir()
        
        for climate in climate_types:
            print(f"📈 Creating chart for {climate} climate")
            
            # Filter data for this climate
            climate_data = df[df['Climate_Type'] == climate]
            
            # Calculate seasonal averages by year
            seasonal_avg = climate_data.groupby(['Year', 'Season'])['Rainfall (mm)'].mean().unstack()
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each year as a line
            seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
            for year in seasonal_avg.index:
                year_data = []
                for season in seasons:
                    if season in seasonal_avg.columns:
                        value = seasonal_avg.loc[year, season]
                        year_data.append(value if not pd.isna(value) else 0)
                    else:
                        year_data.append(0)
                
                ax.plot(seasons, year_data, marker='o', label=f'{year}', linewidth=1.5)
            
            # Add overall mean line
            overall_mean = []
            for season in seasons:
                if season in seasonal_avg.columns:
                    overall_mean.append(seasonal_avg[season].mean())
                else:
                    overall_mean.append(0)
            
            ax.plot(seasons, overall_mean, color='black', linestyle='--', linewidth=2, label='Overall Mean')
            
            # Customize plot
            ax.set_title(f'Average Rainfall by Season - {climate} Climate', fontsize=16, fontweight='bold')
            ax.set_xlabel('Season', fontsize=12)
            ax.set_ylabel('Average Rainfall (mm)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(title='Year', fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
            
            plt.tight_layout()
            
            # Save the chart
            chart_path = os.path.join(static_dir, f'{climate}_climate_rainfall.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {climate} chart saved: {chart_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chart 2 failed: {e}")
        return False

def generate_chart_3():
    """Generate Chart 3: Interactive Dashboard HTML"""
    print("📊 Generating Chart 3: Interactive Dashboard")
    
    try:
        static_dir = create_static_dir()
        
        # Check what image files we have
        image_files = []
        if os.path.exists(static_dir):
            image_files = [f for f in os.listdir(static_dir) if f.endswith('.png')]
        
        # Create simple HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🌧️ Germany Rainfall Analysis Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .content {{
            padding: 40px;
        }}
        .chart-section {{
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }}
        .chart-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .success {{
            background: #d4edda;
            color: #155724;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #c3e6cb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌧️ Germany Rainfall Analysis</h1>
            <p>Your 3 Key Visualizations - Successfully Generated!</p>
        </div>
        
        <div class="content">
            <div class="success">
                ✅ <strong>Success!</strong> Generated {len(image_files)} visualization(s)
            </div>
            
            <div class="chart-section">
                <h3>📊 Chart 1: Annual Rainfall by Climate Type (Stacked Bar)</h3>
                <img src="/images/rainfall_stacked_bar_simplified.png" alt="Stacked Bar Chart" class="chart-image" onerror="this.style.display='none'">
                <p>This stacked bar chart shows the total annual rainfall across different climate types with a trend line.</p>
            </div>
            
            <div class="chart-section">
                <h3>🌍 Chart 2: Seasonal Rainfall Patterns by Climate</h3>
                
                <h4>Continental Climate</h4>
                <img src="/images/Continental_climate_rainfall.png" alt="Continental Climate" class="chart-image" onerror="this.style.display='none'">
                
                <h4>Oceanic Climate</h4>
                <img src="/images/Oceanic_climate_rainfall.png" alt="Oceanic Climate" class="chart-image" onerror="this.style.display='none'">
                
                <p>These charts show seasonal rainfall patterns for different climate types across German cities.</p>
            </div>
            
            <div class="chart-section">
                <h3>📋 Generated Files</h3>
                <ul>
                    {''.join([f'<li>🖼️ <a href="/images/{img}" target="_blank">{img}</a></li>' for img in image_files])}
                </ul>
            </div>
            
            <div class="chart-section">
                <h3>🔗 Navigation</h3>
                <p>
                    <a href="/" style="color: #3498db; text-decoration: none;">🏠 Home</a> | 
                    <a href="/data" style="color: #3498db; text-decoration: none;">📊 Data API</a> | 
                    <a href="/files" style="color: #3498db; text-decoration: none;">📁 All Files</a>
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        # Save dashboard
        dashboard_path = os.path.join(static_dir, 'germany_spi_comprehensive_dashboard.html')
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard saved: {dashboard_path}")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        return False

def generate_all_charts():
    """Generate all 3 charts you want"""
    print("=" * 60)
    print("🎯 GENERATING YOUR 3 SPECIFIC CHARTS")
    print("=" * 60)
    
    success_count = 0
    
    # Generate Chart 1: Stacked Bar
    if generate_chart_1():
        success_count += 1
    
    # Generate Chart 2: Climate Charts
    if generate_chart_2():
        success_count += 1
    
    # Generate Chart 3: Dashboard
    if generate_chart_3():
        success_count += 1
    
    print("=" * 60)
    print(f"🎉 COMPLETED: {success_count}/3 charts generated successfully!")
    
    # List generated files
    static_dir = 'static'
    if os.path.exists(static_dir):
        files = os.listdir(static_dir)
        print(f"📁 Files in {static_dir}:")
        for file in files:
            print(f"   ✅ {file}")
    
    return success_count == 3

# Generate charts when app starts
try:
    generation_success = generate_all_charts()
    print(f"🎯 Final result: {generation_success}")
except Exception as e:
    print(f"❌ Generation error: {e}")
    generation_success = False

# Flask Routes
@app.route('/')
def home():
    """Simple homepage"""
    static_dir = 'static'
    image_files = []
    
    if os.path.exists(static_dir):
        image_files = [f for f in os.listdir(static_dir) if f.endswith('.png')]
    
    status = "✅ SUCCESS" if generation_success else "❌ FAILED"
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>🌧️ Rainfall Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }}
        .status {{ padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .success {{ background: #d4edda; color: #155724; }}
        .failed {{ background: #f8d7da; color: #721c24; }}
        .btn {{ display: inline-block; padding: 15px 25px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌧️ Rainfall Analysis & Drought Assessment</h1>
        <h2>German Cities - Your 3 Key Visualizations</h2>
        
        <div class="status {'success' if generation_success else 'failed'}">
            <strong>Generation Status:</strong> {status}<br>
            <strong>Charts Created:</strong> {len(image_files)}/3
        </div>
        
        <h3>🚀 Quick Access</h3>
        <a href="/dashboard" class="btn">🌍 Open Dashboard</a>
        <a href="/images/rainfall_stacked_bar_simplified.png" class="btn">📊 Stacked Chart</a>
        <a href="/data" class="btn">📋 API Data</a>
        <a href="/files" class="btn">📁 All Files</a>
        
        <h3>📊 Your Charts ({len(image_files)})</h3>
        <ul>
            {''.join([f'<li><a href="/images/{img}" target="_blank">🖼️ {img}</a></li>' for img in image_files]) if image_files else '<li>No charts generated</li>'}
        </ul>
        
        <h3>📖 About</h3>
        <p><strong>Objective:</strong> Generate exactly 3 visualizations:</p>
        <ol>
            <li><code>rainfall_stacked_bar_simplified.png</code> - Stacked bar chart</li>
            <li><code>[Climate]_climate_rainfall.png</code> - Seasonal patterns</li>
            <li><code>germany_spi_comprehensive_dashboard.html</code> - Interactive dashboard</li>
        </ol>
    </div>
</body>
</html>
    """

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard"""
    try:
        return send_from_directory('static', 'germany_spi_comprehensive_dashboard.html')
    except:
        return "<h1>Dashboard not available</h1><p><a href='/'>Back to Home</a></p>", 404

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images"""
    try:
        return send_from_directory('static', filename)
    except:
        return f"Image {filename} not found", 404

@app.route('/files')
def list_files():
    """List all files"""
    files = []
    if os.path.exists('static'):
        files = os.listdir('static')
    
    return f"""
    <h1>📁 Generated Files</h1>
    <p><strong>Total:</strong> {len(files)} files</p>
    <ul>
        {''.join([f'<li>{file}</li>' for file in files]) if files else '<li>No files</li>'}
    </ul>
    <p><a href="/">← Back</a></p>
    """

@app.route('/data')
def get_data():
    """API endpoint"""
    files = []
    if os.path.exists('static'):
        files = os.listdir('static')
    
    return jsonify({
        "status": "Your 3 Charts Generator",
        "generation_success": generation_success,
        "files_generated": len(files),
        "target_charts": [
            "rainfall_stacked_bar_simplified.png",
            "[Climate]_climate_rainfall.png", 
            "germany_spi_comprehensive_dashboard.html"
        ],
        "actual_files": files,
        "urls": {
            "dashboard": "/dashboard",
            "stacked_chart": "/images/rainfall_stacked_bar_simplified.png"
        }
    })

if __name__ == '__main__':
    print("🚀 Starting your 3-chart Flask app...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)