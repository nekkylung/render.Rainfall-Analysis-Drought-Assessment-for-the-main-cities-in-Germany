# Fix Render.com Deployment Issues

## Problem Analysis
Your deployment failed with: `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'`

This means the `requirements.txt` file is either:
1. Missing from your repository
2. Located in the wrong directory
3. Named incorrectly

## Step-by-Step Fix

### Step 1: Check Your Repository Structure
Your repository should look like this:
```
your-repo/
├── app.py
├── requirements.txt          ← This file MUST be in root directory
├── Rainfall_Cleaned.csv
├── templates/
│   └── index.html
└── spi_tableau_data.csv     ← Generated after data processing
```

### Step 2: Create/Fix requirements.txt
Create a file named exactly `requirements.txt` (lowercase, with .txt extension) in the **root directory** of your repository:

```txt
Flask==2.3.3
pandas==2.1.1
numpy==1.24.3
scipy==1.11.3
requests==2.31.0
gunicorn==21.2.0
Werkzeug==2.3.7
```

### Step 3: Update Your Repository
1. **Add the requirements.txt file** to the root of your repository
2. **Commit and push** the changes to GitHub
3. **Redeploy** on Render.com

### Step 4: Fix Render.com Configuration
In your Render.com service settings:

**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
gunicorn app:app
```

**Environment:**
- Select `Python 3`
- Python Version: `3.11.9` (recommended for better compatibility)

### Step 5: Alternative Build Commands (if needed)
If the simple build command doesn't work, try these alternatives:

**Option A - With data processing:**
```
pip install -r requirements.txt && python deploy.py
```

**Option B - Upgrade pip first:**
```
pip install --upgrade pip && pip install -r requirements.txt
```

**Option C - Force reinstall:**
```
pip install --force-reinstall -r requirements.txt
```

## Common Issues and Solutions

### Issue 1: File Not in Root Directory
- **Problem:** requirements.txt is in a subdirectory
- **Solution:** Move it to the root directory (same level as app.py)

### Issue 2: Wrong File Name
- **Problem:** File named `Requirements.txt` (capital R) or `requirement.txt` (singular)
- **Solution:** Rename to exactly `requirements.txt`

### Issue 3: File Encoding Issues
- **Problem:** File has wrong encoding or invisible characters
- **Solution:** Recreate the file using plain text editor

### Issue 4: Poetry/Conda Dependencies
- **Problem:** Using Poetry or Conda instead of pip
- **Solution:** Use pip-compatible requirements.txt format

## Test Locally First
Before deploying to Render, test locally:

```bash
# Navigate to your project directory
cd your-project-directory

# Install dependencies
pip install -r requirements.txt

# Run your app
python app.py

# Visit http://localhost:5000
```

## Render.com Service Settings Checklist

✅ **Repository:** Connected to correct GitHub repository
✅ **Branch:** `main` or `master` (whichever contains your code)
✅ **Runtime:** `Python 3`
✅ **Build Command:** `pip install -r requirements.txt`
✅ **Start Command:** `gunicorn app:app`
✅ **Python Version:** `3.11.9` (in Environment Variables)

## Advanced Troubleshooting

### Check Your Repository Files
Visit your GitHub repository directly and verify:
1. `requirements.txt` is visible in the root directory
2. Click on the file to make sure it contains the correct dependencies
3. Check that there are no typos in the filename

### Manual Deploy Option
If GitHub integration isn't working:
1. Download your repository as ZIP
2. Extract and verify `requirements.txt` is in the root
3. Use Render's manual deploy option

### Environment Variables (if needed)
Add these environment variables in Render dashboard:
- `PYTHON_VERSION`: `3.11.9`
- `PORT`: `5000`
- `FLASK_ENV`: `production`

## Quick Fix Commands

**Create requirements.txt file:**
```bash
echo "Flask==2.3.3
pandas==2.1.1
numpy==1.24.3
scipy==1.11.3
requests==2.31.0
gunicorn==21.2.0
Werkzeug==2.3.7" > requirements.txt
```

**Verify file exists:**
```bash
ls -la requirements.txt
cat requirements.txt
```

## Next Steps After Fix
1. Push the corrected `requirements.txt` to GitHub
2. In Render dashboard, click "Manual Deploy" → "Deploy latest commit"
3. Monitor the deployment logs
4. Once successful, your app will be available at your Render URL

Your dashboard should be accessible at: `https://your-app-name.onrender.com`