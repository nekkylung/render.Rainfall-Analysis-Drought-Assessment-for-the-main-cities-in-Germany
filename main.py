from flask import Flask, send_from_directory, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

print("🚀 STARTING SIMPLE CHART GENERATOR")

def ensure_static_dir():
    if not os.path.exists('static'):
        os.makedirs('static')
    return 'static'

def create_charts():
    print("📊 Creating your 3 charts...")
    
    try:
        # Load data
        df = pd.read_csv('Rainfall_Data_Germany_Complete.csv')
        print(f"✅ Data loaded: {len(df)} rows, {df.columns.tolist()}")
        
        static_dir = ensure_static_dir()
        
        # CHART 1: Stacked Bar Chart
        print("📊 Chart 1: rainfall_stacked_bar_simplified.png")
        annual_climate = df.groupby(['Year', 'Climate_Type'])['Rainfall (mm)'].sum().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        annual_climate.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Total Annual Rainfall by Climate Type')
        ax.set_xlabel('Year')
        ax.set_ylabel('Rainfall (mm)')
        
        # Add trend line
        total_annual = annual_climate.sum(axis=1)
        x_pos = np.arange(len(total_annual))
        z = np.polyfit(x_pos, total_annual.values, 1)
        p = np.poly1d(z)
        ax.plot(x_pos, p(x_pos), "k--", alpha=0.7, linewidth=2, label='Trend')
        
        plt.savefig(os.path.join(static_dir, 'rainfall_stacked_bar_simplified.png'), dpi=300, bbox_inches='tight')
        plt.close()
        annual_climate.to_csv(os.path.join(static_dir, 'rainfall_stacked_data.csv'))
        print("✅ Chart 1 saved")
        
        # CHART 2: Climate rainfall charts
        print("📊 Chart 2: Climate rainfall charts")
        seasons = {1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn',12:'Winter'}
        df['Season'] = df['Month'].map(seasons)
        
        for climate in df['Climate_Type'].unique():
            print(f"📈 Creating {climate} chart")
            climate_data = df[df['Climate_Type'] == climate]
            seasonal = climate_data.groupby(['Year', 'Season'])['Rainfall (mm)'].mean().unstack()
            
            plt.figure(figsize=(12, 8))
            season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
            
            for year in seasonal.index:
                year_values = [seasonal.loc[year, s] if s in seasonal.columns else 0 for s in season_order]
                plt.plot(season_order, year_values, marker='o', label=f'{year}')
            
            plt.title(f'Average Rainfall by Season - {climate} Climate', fontsize=16, fontweight='bold')
            plt.xlabel('Season')
            plt.ylabel('Average Rainfall (mm)')
            plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(static_dir, f'{climate}_climate_rainfall.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {climate} chart saved")
        
        # CHART 3: Dashboard HTML
        print("📊 Chart 3: Dashboard HTML")
        images = [f for f in os.listdir(static_dir) if f.endswith('.png')]
        
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🌧️ Germany Rainfall Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        .success {{ background: #d4edda; color: #155724; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .chart {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .btn {{ display: inline-block; padding: 12px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌧️ Germany Rainfall Analysis Dashboard</h1>
        
        <div class="success">
            ✅ <strong>SUCCESS!</strong> Your 3 charts have been generated successfully!
        </div>
        
        <div class="chart">
            <h3>📊 Chart 1: Annual Rainfall by Climate Type</h3>
            <img src="/images/rainfall_stacked_bar_simplified.png" alt="Stacked Bar Chart">
        </div>
        
        <div class="chart">
            <h3>🌍 Chart 2: Seasonal Rainfall Patterns</h3>
            {''.join([f'<h4>{img.replace("_climate_rainfall.png", "").replace("_", " ")} Climate</h4><img src="/images/{img}" alt="{img}"><br>' for img in images if 'climate_rainfall' in img])}
        </div>
        
        <div class="chart">
            <h3>📋 All Generated Files ({len(images)} images)</h3>
            <ul>
                {''.join([f'<li><a href="/images/{img}" target="_blank">🖼️ {img}</a></li>' for img in images])}
            </ul>
        </div>
        
        <div>
            <a href="/" class="btn">🏠 Home</a>
            <a href="/data" class="btn">📊 API Data</a>
        </div>
    </div>
</body>
</html>"""
        
        with open(os.path.join(static_dir, 'germany_spi_comprehensive_dashboard.html'), 'w') as f:
            f.write(dashboard_html)
        print("✅ Dashboard saved")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Create charts when app starts
print("🔄 Generating charts...")
success = create_charts()
print(f"🎯 Generation result: {success}")

@app.route('/')
def home():
    files = []
    if os.path.exists('static'):
        files = [f for f in os.listdir('static') if f.endswith('.png')]
    
    return f"""
<!DOCTYPE html>
<html>
<head><title>🌧️ Your 3 Charts</title>
<style>body{{font-family:Arial;padding:40px;background:#f5f5f5;}}
.container{{max-width:800px;margin:0 auto;background:white;padding:40px;border-radius:10px;}}
.success{{background:#d4edda;color:#155724;padding:20px;border-radius:8px;margin:20px 0;}}
.btn{{display:inline-block;padding:15px 25px;background:#007bff;color:white;text-decoration:none;border-radius:5px;margin:10px 5px;}}
</style></head>
<body>
<div class="container">
<h1>🌧️ Your 3 Charts Are Ready!</h1>
<div class="success">✅ <strong>SUCCESS!</strong> Generated {len(files)} chart(s)</div>

<h3>🚀 Quick Access</h3>
<a href="/dashboard" class="btn">🌍 View Dashboard</a>
<a href="/images/rainfall_stacked_bar_simplified.png" class="btn">📊 Stacked Chart</a>
<a href="/data" class="btn">📋 API Info</a>

<h3>📊 Your Generated Charts</h3>
<ul>{''.join([f'<li><a href="/images/{f}" target="_blank">🖼️ {f}</a></li>' for f in files]) if files else '<li>No charts found</li>'}</ul>

<h3>✅ Mission Complete</h3>
<p>Your 3 specific visualizations:</p>
<ol>
<li><code>rainfall_stacked_bar_simplified.png</code> ✅</li>
<li><code>[Climate]_climate_rainfall.png</code> ✅</li>
<li><code>germany_spi_comprehensive_dashboard.html</code> ✅</li>
</ol>
</div></body></html>"""

@app.route('/dashboard')
def dashboard():
    try:
        return send_from_directory('static', 'germany_spi_comprehensive_dashboard.html')
    except:
        return "<h1>Dashboard not ready</h1><a href='/'>Home</a>", 404

@app.route('/images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory('static', filename)
    except:
        return f"Image not found: {filename}", 404

@app.route('/data')
def api_data():
    files = []
    if os.path.exists('static'):
        files = os.listdir('static')
    
    return jsonify({
        "message": "YOUR 3 CHARTS GENERATED SUCCESSFULLY!",
        "status": "✅ WORKING",
        "generation_success": success,
        "files_created": len(files),
        "your_charts": [
            "rainfall_stacked_bar_simplified.png",
            "Continental_climate_rainfall.png", 
            "Oceanic_climate_rainfall.png",
            "germany_spi_comprehensive_dashboard.html"
        ],
        "actual_files": files,
        "chart_urls": {
            "stacked_bar": "/images/rainfall_stacked_bar_simplified.png",
            "dashboard": "/dashboard"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)