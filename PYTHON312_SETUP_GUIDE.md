# Python 3.12 Migration Guide

## Current Issue
- ❌ You have: Python 3.14.2
- ✅ You need: Python 3.12.x (for TensorFlow support)

## Solution: 4 Easy Steps

### STEP 1: Uninstall Python 3.14 ⚙️
1. Press `Windows Key + I` to open Settings
2. Click **Apps**
3. Click **Apps & features**
4. Search for **"Python 3.14"**
5. Click it and select **Uninstall**
6. Follow the prompts and click **Yes** to confirm
7. Close Settings

### STEP 2: Download Python 3.12 ⬇️
1. Open your browser
2. Go to: https://www.python.org/downloads/
3. Click the **Download Python 3.12** button
4. Save the installer to your Downloads folder

### STEP 3: Install Python 3.12 ✅
1. Double-click the Python 3.12 installer
2. **IMPORTANT:** Check the box "Add Python to PATH"
3. Click **Install Now** and wait for installation to complete
4. Click **Disable path length limit** (optional but recommended)
5. Click **Close**

### STEP 4: Verify & Install Packages 📦
Open PowerShell and run these commands one by one:

```powershell
# Verify Python 3.12 is installed
python --version
# Should see: Python 3.12.x

# Navigate to your project
cd c:\Users\muluw\Desktop\capstone

# Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# Install all packages including TensorFlow
pip install -r requirements.txt
# This will take 5-10 minutes

# Verify TensorFlow works
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__} installed!')"
# Should see: TensorFlow 2.14.x installed!
```

### STEP 5: Test Your App 🚀
```powershell
streamlit run app.py
```

Your app should now open in a browser with full Deep Learning support!

---

## Troubleshooting

**If `python --version` still shows 3.14:**
- Close PowerShell completely
- Reopen PowerShell
- Try `python --version` again

**If pip install fails:**
- Make sure you used `python -m pip install --upgrade pip` first
- Run `pip install -r requirements.txt` again

**If TensorFlow import fails:**
- Double-check Python version: `python --version` should be 3.12.x
- Run `pip install tensorflow --upgrade` separately

---

## Expected Results After Setup
✅ `python --version` → Python 3.12.x  
✅ `import tensorflow` → No errors  
✅ Page 2 Deep Learning → CNN, RNN, LSTM available  
✅ Streamlit app → Launches in browser with all features  

Good luck! 🎉
