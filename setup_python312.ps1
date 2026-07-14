# Python 3.12 Setup Script for Windows
# This script helps uninstall Python 3.14 and prepare for Python 3.12 installation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python 3.12 Installation Helper" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check current Python version
Write-Host "Current Python version:" -ForegroundColor Yellow
python --version
Write-Host ""

Write-Host "NEXT STEPS:" -ForegroundColor Green
Write-Host ""
Write-Host "1. Uninstall Python 3.14:" -ForegroundColor Yellow
Write-Host "   - Open Settings"
Write-Host "   - Go to Apps > Apps & features"
Write-Host "   - Search for 'Python 3.14'"
Write-Host "   - Click Uninstall and follow prompts"
Write-Host ""

Write-Host "2. Download Python 3.12:" -ForegroundColor Yellow
Write-Host "   - Visit: https://www.python.org/downloads/"
Write-Host "   - Click 'Download Python 3.12'"
Write-Host "   - Run the installer"
Write-Host ""

Write-Host "3. During Python 3.12 Installation:" -ForegroundColor Yellow
Write-Host "   ✓ CHECK: 'Add Python 3.12 to PATH'"
Write-Host "   ✓ CHOOSE: 'Install Now' (recommended)"
Write-Host ""

Write-Host "4. After Installation, run these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   cd c:\Users\muluw\Desktop\capstone" -ForegroundColor Cyan
Write-Host "   python --version" -ForegroundColor Cyan
Write-Host "   python -m pip install --upgrade pip setuptools wheel" -ForegroundColor Cyan
Write-Host "   pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host "   python -c `"import tensorflow; print(f'TensorFlow {tensorflow.__version__} installed!')`"" -ForegroundColor Cyan
Write-Host ""

Write-Host "5. Then run your Streamlit app:" -ForegroundColor Yellow
Write-Host "   streamlit run app.py" -ForegroundColor Cyan
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ready to proceed? Follow the steps above." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
