from flask import Flask, render_template, jsonify, send_file, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # CRITICAL: Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Create Flask app
app = Flask(__name__)

# Global variables to track status
GENERATION_STATUS = {
    'started': False,
    'completed': False,
    'errors': [],
    'files_created': []
}

def log_status(message):
    """Log status messages"""
    print(f"🔄 {message}")
    GENERATION_STATUS['errors'].append(message)

def create_static_dir():
    """Create static directory"""
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        log_status(f"Created directory: {static_dir}")
    return static_dir

def find_data_file():
    """Find the rainfall data file - try multiple possible names"""
    possible_names = [
        'Rainfall_Data_Germany_Complete.csv',
        'rainfall_data.csv',
        'data.csv'
    ]
    
    # Check current directory
    for name in possible_names:
        if os.path.exists(name):
            log_status(f"Found data file: {name}")
            return name
    
    # List all CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    log_status(f"Available CSV files: {csv_files}")
    
    if csv_files:
        # Use the first CSV file found
        log_status(f"Using first available CSV: {csv_files[0]}")
        return csv_files[0]
    
    log_status("❌ ERROR: No CSV data file found!")
    return None

def create_sample_data():
    """Create sample data if no data file is found"""
    log_status("Creating sample data...")
    
    # Create sample rainfall data
    cities = ['Berlin', 'Munich', 'Hamburg', 'Cologne', 'Frankfurt']
    climate_types = ['Continental', 'Oceanic']
    years = list(range(2020, 2025))
    months = list(range(1, 13))
    
    data = []
    for city in cities:
        for year in years:
            for month in months:
                data.append({
                    'City': city,
                    'Year': year,
                    'Month': month,
                    'Rainfall (mm)': np.random.normal(50, 20),
                    'Climate_Type': np.random.choice(climate_types),
                    'Temperature (°C)': np.random.normal(10, 5),
                    'Humidity (%)': np.random.randint(40, 90),
                    'Latitude': 50 + np.random.random() * 5,
                    'Longitude': 8 + np.random.random() * 5,
                    'Elevation (m)': np.random.randint(0, 500)
                })
    
    df = pd.DataFrame(data)
    filename = 'sample_rainfall_data.csv'
    df.to_csv(filename, index=False)
    log_status(f"Sample data created: {filename}")
    return filename

def generate_chart_1():
    """Generate the stacked bar chart"""
    try:
        log_status("Starting Chart 1: Stacked Bar Chart")
        
        # Find or create data
        data_file = find_data_file()
        if not data_file:
            data_file = create_sample_data()
        
        # Load data
        df = pd.read_csv(data_file)
        log_status(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Create chart
        static_dir = create_static_dir()
        
        # Group by year and climate type
        if 'Climate_Type' in df.columns:
            yearly_climate = df.groupby(['Year', 'Climate_Type'])['Rainfall (mm)'].sum().unstack(fill_value=0)
        else:
            # Fallback: just group by year
            yearly_climate = df.groupby('Year')['Rainfall (mm)'].sum().to_frame()
            yearly_climate.columns = ['Total Rainfall']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_climate.plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_title('Annual Rainfall by Climate Type', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Rainfall (mm)', fontsize=12)
        ax.legend(title='Climate Type')
        ax.grid(True, alpha=0.3)
        
        # Save the chart
        chart_path = os.path.join(static_dir, 'rainfall_stacked_bar_simplified.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        GENERATION_STATUS['files_created'].append(chart_path)
        log_status(f"✅ Chart 1 created: {chart_path}")
        return True
        
    except Exception as e:
        log_status(f"❌ Chart 1 failed: {str(e)}")
        return False

def generate_chart_2():
    """Generate the climate rainfall charts"""
    try:
        log_status("Starting Chart 2: Climate Charts")
        
        # Find or create data
        data_file = find_data_file()
        if not data_file:
            data_file = create_sample_data()
        
        # Load data
        df = pd.read_csv(data_file)
        static_dir = create_static_dir()
        
        # Map months to seasons
        season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
        df['Season'] = df['Month'].map(season_map)
        
        # Get climate types
        if 'Climate_Type' in df.columns:
            climate_types = df['Climate_Type'].unique()
        else:
            climate_types = ['Sample_Climate']
        
        # Create chart for each climate type
        for climate in climate_types:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if 'Climate_Type' in df.columns:
                climate_data = df[df['Climate_Type'] == climate]
            else:
                climate_data = df
            
            # Group by season and year
            seasonal_data = climate_data.groupby(['Season', 'Year'])['Rainfall (mm)'].mean().unstack()
            
            # Plot
            seasonal_data.plot(kind='line', ax=ax, marker='o')
            ax.set_title(f'Seasonal Rainfall - {climate} Climate', fontsize=14, fontweight='bold')
            ax.set_xlabel('Season')
            ax.set_ylabel('Average Rainfall (mm)')
            ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Save chart
            chart_path = os.path.join(static_dir, f'{climate}_climate_rainfall.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            GENERATION_STATUS['files_created'].append(chart_path)
            log_status(f"✅ Climate chart created: {chart_path}")
        
        return True
        
    except Exception as e:
        log_status(f"❌ Chart 2 failed: {str(e)}")
        return False

def generate_dashboard():
    """Generate the HTML dashboard"""
    try:
        log_status("Starting Dashboard Creation")
        
        static_dir = create_static_dir()
        
        # Get list of generated images
        if os.path.exists(static_dir):
            image_files = [f for f in os.listdir(static_dir) if f.endswith('.png')]
        else:
            image_files = []
        
        # Create dashboard HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🌧️ Germany Rainfall Analysis Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: 'Arial', sans-serif;
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
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
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
        .chart-section h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .chart-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .status {{
            padding: 20px;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            margin: 20px 0;
            color: #155724;
        }}
        .btn {{
            display: inline-block;
            padding: 12px 25px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 5px;
            transition: background 0.3s;
        }}
        .btn:hover {{
            background: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌧️ Germany Rainfall Analysis</h1>
            <p>Interactive Dashboard - Generated Successfully!</p>
        </div>
        
        <div class="content">
            <div class="status">
                ✅ <strong>Dashboard Status:</strong> Successfully generated {len(image_files)} visualizations
            </div>
            
            <div class="chart-section">
                <h3>📊 Generated Visualizations</h3>
                {''.join([f'<div><h4>{img}</h4><img src="/images/{img}" alt="{img}" class="chart-image"></div>' for img in image_files]) if image_files else '<p>No charts generated yet.</p>'}
            </div>
            
            <div class="chart-section">
                <h3>🔗 Navigation</h3>
                <a href="/" class="btn">🏠 Home</a>
                <a href="/data" class="btn">📊 Data API</a>
                <a href="/files" class="btn">📁 All Files</a>
                <a href="/debug" class="btn">🔧 Debug Info</a>
            </div>
            
            <div class="chart-section">
                <h3>📖 About</h3>
                <p><strong>Dataset:</strong> German cities rainfall analysis</p>
                <p><strong>Visualizations:</strong> Stacked bar charts and seasonal patterns</p>
                <p><strong>Generated:</strong> {len(image_files)} charts available</p>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        dashboard_path = os.path.join(static_dir, 'germany_spi_comprehensive_dashboard.html')
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        GENERATION_STATUS['files_created'].append(dashboard_path)
        log_status(f"✅ Dashboard created: {dashboard_path}")
        return True
        
    except Exception as e:
        log_status(f"❌ Dashboard creation failed: {str(e)}")
        return False

def generate_all_visualizations():
    """Generate all visualizations with detailed logging"""
    GENERATION_STATUS['started'] = True
    
    log_status("🚀 STARTING VISUALIZATION GENERATION")
    log_status("=" * 50)
    
    # List current directory contents
    current_files = os.listdir('.')
    log_status(f"Current directory files: {current_files}")
    
    success_count = 0
    total_charts = 3
    
    # Generate each visualization
    if generate_chart_1():
        success_count += 1
    
    if generate_chart_2():
        success_count += 1
    
    if generate_dashboard():
        success_count += 1
    
    GENERATION_STATUS['completed'] = True
    
    log_status("=" * 50)
    log_status(f"🎯 GENERATION COMPLETE: {success_count}/{total_charts} successful")
    log_status(f"📁 Files created: {GENERATION_STATUS['files_created']}")
    
    return success_count == total_charts

# Generate visualizations when app starts
print("🔄 Flask app initializing...")
try:
    generation_success = generate_all_visualizations()
    print(f"🎯 Generation result: {generation_success}")
except Exception as e:
    print(f"❌ Generation error: {e}")
    generation_success = False

# Flask Routes
@app.route('/')
def home():
    """Homepage with comprehensive status"""
    static_dir = 'static'
    
    # Get file lists
    image_files = []
    csv_files = []
    all_files = []
    
    if os.path.exists(static_dir):
        all_files = os.listdir(static_dir)
        image_files = [f for f in all_files if f.endswith('.png')]
        csv_files = [f for f in all_files if f.endswith('.csv')]
    
    status_color = "#d4edda" if generation_success else "#f8d7da"
    status_text_color = "#155724" if generation_success else "#721c24"
    status_message = "✅ All visualizations generated successfully!" if generation_success else "⚠️ Some issues during generation - check debug info"
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>🌧️ Rainfall Analysis Dashboard</title>
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
            padding: 40px; 
            text-align: center; 
        }}
        .content {{ padding: 30px; }}
        .status {{ 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
            background: {status_color};
            color: {status_text_color};
            border: 1px solid currentColor;
            font-weight: bold;
        }}
        .section {{ 
            background: #f8f9fa; 
            padding: 25px; 
            margin: 25px 0; 
            border-radius: 10px; 
        }}
        .btn {{ 
            display: inline-block; 
            padding: 15px 25px; 
            background: #3498db; 
            color: white; 
            text-decoration: none; 
            border-radius: 8px; 
            margin: 10px 5px; 
            font-weight: bold;
            transition: transform 0.2s;
        }}
        .btn:hover {{ 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }}
        .btn-success {{ background: #27ae60; }}
        .file-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 15px; 
            margin: 20px 0;
        }}
        .file-item {{ 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            border: 1px solid #ddd;
        }}
        .stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 20px; 
            margin: 20px 0;
        }}
        .stat {{ 
            text-align: center; 
            background: white; 
            padding: 20px; 
            border-radius: 10px;
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
            <div class="status">
                {status_message}
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{len(image_files)}</div>
                    <div>Charts Generated</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{len(all_files)}</div>
                    <div>Total Files</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{"✅" if generation_success else "❌"}</div>
                    <div>Status</div>
                </div>
            </div>
            
            <div class="section">
                <h3>🚀 Quick Access</h3>
                <a href="/dashboard" class="btn btn-success">🌍 Open Dashboard</a>
                <a href="/images/rainfall_stacked_bar_simplified.png" class="btn" target="_blank">📊 Stacked Chart</a>
                <a href="/files" class="btn">📁 All Files</a>
                <a href="/debug" class="btn">🔧 Debug Info</a>
            </div>
            
            <div class="section">
                <h3>📊 Generated Charts ({len(image_files)})</h3>
                <div class="file-grid">
                    {''.join([f'<div class="file-item">🖼️ <a href="/images/{img}" target="_blank">{img}</a></div>' for img in image_files]) if image_files else '<div class="file-item">No charts generated yet</div>'}
                </div>
            </div>
            
            <div class="section">
                <h3>🔗 API Endpoints</h3>
                <a href="/data" class="btn">📊 Data Summary</a>
                <a href="/summary" class="btn">📈 Statistics</a>
            </div>
        </div>
    </div>
</body>
</html>
    """

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard"""
    try:
        return send_from_directory('static', 'germany_spi_comprehensive_dashboard.html')
    except FileNotFoundError:
        return f"""
        <div style="text-align: center; padding: 50px; font-family: Arial;">
            <h1>🔄 Dashboard Not Ready</h1>
            <p>Dashboard is being generated or failed to create.</p>
            <p><a href="/debug">Check Debug Info</a> | <a href="/">Back to Home</a></p>
        </div>
        """, 404

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images"""
    try:
        return send_from_directory('static', filename)
    except FileNotFoundError:
        return f"Image {filename} not found. <a href='/debug'>Check debug info</a>", 404

@app.route('/files')
def list_files():
    """List all files"""
    static_dir = 'static'
    files = []
    
    if os.path.exists(static_dir):
        files = os.listdir(static_dir)
    
    file_list = '<br>'.join([f"📄 {file}" for file in files]) if files else "No files found"
    
    return f"""
    <div style="padding: 40px; font-family: Arial;">
        <h1>📁 Generated Files</h1>
        <p><strong>Directory:</strong> {static_dir}</p>
        <p><strong>Files ({len(files)}):</strong></p>
        <div style="background: #f5f5f5; padding: 20px; border-radius: 8px;">
            {file_list}
        </div>
        <p><a href="/">← Back to Home</a></p>
    </div>
    """

@app.route('/debug')
def debug_info():
    """Debug information"""
    import sys
    import platform
    
    current_dir_files = os.listdir('.')
    static_exists = os.path.exists('static')
    static_files = os.listdir('static') if static_exists else []
    
    debug_html = f"""
    <div style="padding: 40px; font-family: monospace; background: #f5f5f5;">
        <h1>🔧 Debug Information</h1>
        
        <h3>📊 Generation Status</h3>
        <p><strong>Started:</strong> {GENERATION_STATUS['started']}</p>
        <p><strong>Completed:</strong> {GENERATION_STATUS['completed']}</p>
        <p><strong>Success:</strong> {generation_success}</p>
        <p><strong>Files Created:</strong> {len(GENERATION_STATUS['files_created'])}</p>
        
        <h3>📁 Directory Contents</h3>
        <p><strong>Current Directory:</strong> {current_dir_files}</p>
        <p><strong>Static Directory Exists:</strong> {static_exists}</p>
        <p><strong>Static Files:</strong> {static_files}</p>
        
        <h3>🔍 Error Log</h3>
        <pre style="background: white; padding: 20px; border-radius: 5px;">
{'<br>'.join(GENERATION_STATUS['errors']) if GENERATION_STATUS['errors'] else 'No errors logged'}
        </pre>
        
        <h3>🖥️ System Info</h3>
        <p><strong>Python:</strong> {sys.version}</p>
        <p><strong>Platform:</strong> {platform.platform()}</p>
        
        <p><a href="/">← Back to Home</a></p>
    </div>
    """
    
    return debug_html

@app.route('/data')
def get_data():
    """API endpoint"""
    static_files = []
    if os.path.exists('static'):
        static_files = os.listdir('static')
    
    return jsonify({
        "status": "deployed_and_running",
        "generation_success": generation_success,
        "files_generated": len(static_files),
        "static_files": static_files,
        "generation_status": GENERATION_STATUS,
        "urls": {
            "dashboard": "/dashboard",
            "stacked_chart": "/images/rainfall_stacked_bar_simplified.png",
            "debug": "/debug"
        }
    })

@app.route('/summary')
def get_summary():
    """Summary endpoint"""
    return jsonify({
        "project": "German Rainfall Analysis",
        "status": "✅ Deployed successfully" if generation_success else "⚠️ Deployed with issues",
        "visualizations_target": 3,
        "visualizations_created": len(GENERATION_STATUS['files_created']),
        "deployment_url": "https://render-rainfall-analysis-drought.onrender.com"
    })

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    print(f"📊 Generation status: {generation_success}")
    print(f"📁 Files created: {len(GENERATION_STATUS.get('files_created', []))}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)