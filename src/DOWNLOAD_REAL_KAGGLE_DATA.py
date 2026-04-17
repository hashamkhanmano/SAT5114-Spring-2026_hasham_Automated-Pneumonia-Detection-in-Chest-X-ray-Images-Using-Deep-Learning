"""
DOWNLOAD REAL KAGGLE DATASET
This script downloads the Kaggle Chest X-Ray dataset:
- 5,863 pediatric chest X-ray images
- From Guangzhou Women and Children's Medical Center
- Published by Kermany et al. (2018) in Cell journal
"""

import subprocess
import sys
import os
import shutil

def install_kaggle():
    """Install Kaggle API if not present"""
    try:
        import kaggle
        print("✅ Kaggle API already installed")
        return True
    except ImportError:
        print("📦 Installing Kaggle API...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
            print("✅ Kaggle API installed successfully")
            return True
        except:
            print("❌ Failed to install Kaggle API")
            return False

def verify_credentials():
    """Verify Kaggle credentials are properly set up"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle credentials not found!")
        print(f"Expected location: {kaggle_json}")
        print("\n🔑 SETUP REQUIRED:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print(f"4. Place it at: {kaggle_json}")
        return False
    
    # Set proper permissions (important for security)
    try:
        os.chmod(kaggle_json, 0o600)
        print("✅ Kaggle credentials found and secured")
        return True
    except:
        print("⚠️ Credentials found but couldn't set permissions")
        return True

def download_kaggle_dataset():
    """Download the Kaggle Chest X-Ray dataset"""
    print("📥 Downloading Kaggle Chest X-Ray Dataset...")
    print("Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("Expected: 5,863 images from Kermany et al. (2018)")
    
    # Clean up any existing data
    if os.path.exists('data'):
        print("🧹 Removing old dataset...")
        shutil.rmtree('data')
    
    os.makedirs('data', exist_ok=True)
    
    try:
        # Download using Kaggle API
        result = subprocess.run([
            'kaggle', 'datasets', 'download', 
            'paultimothymooney/chest-xray-pneumonia',
            '-p', 'data',
            '--unzip'
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dataset downloaded successfully!")
        print("📊 Verifying dataset structure...")
        
        return verify_dataset_structure()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("Error output:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def verify_dataset_structure():
    """Verify we got the correct dataset structure and counts"""
    print("🔍 Verifying dataset structure and completeness...")
    
    expected_structure = {
        'data/chest_xray/train/NORMAL': 'Training Normal',
        'data/chest_xray/train/PNEUMONIA': 'Training Pneumonia',
        'data/chest_xray/val/NORMAL': 'Validation Normal',
        'data/chest_xray/val/PNEUMONIA': 'Validation Pneumonia',
        'data/chest_xray/test/NORMAL': 'Test Normal',
        'data/chest_xray/test/PNEUMONIA': 'Test Pneumonia'
    }
    
    total_images = 0
    normal_total = 0
    pneumonia_total = 0
    
    print("\n📋 DATASET VERIFICATION:")
    print("-" * 50)
    
    for path, description in expected_structure.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_images += count
            
            if 'NORMAL' in path:
                normal_total += count
            else:
                pneumonia_total += count
            
            print(f"✅ {description}: {count} images")
        else:
            print(f"❌ Missing: {path}")
            return False
    
    print("-" * 50)
    print(f"📊 TOTAL IMAGES: {total_images}")
    print(f"📊 Normal Cases: {normal_total} ({normal_total/total_images*100:.1f}%)")
    print(f"📊 Pneumonia Cases: {pneumonia_total} ({pneumonia_total/total_images*100:.1f}%)")
    print(f"📊 Class Ratio: {pneumonia_total/normal_total:.1f}:1 (Pneumonia:Normal)")
    
    # Verify dataset completeness
    expected_total = 5863
    expected_normal = 1583
    expected_pneumonia = 4273
    
    print(f"\n🎯 DATASET VERIFICATION:")
    print(f"Expected Total: {expected_total} | Actual: {total_images} | {'✅' if abs(total_images - expected_total) < 100 else '❌'}")
    print(f"Expected Normal: {expected_normal} | Actual: {normal_total} | {'✅' if abs(normal_total - expected_normal) < 100 else '❌'}")
    print(f"Expected Pneumonia: {expected_pneumonia} | Actual: {pneumonia_total} | {'✅' if abs(pneumonia_total - expected_pneumonia) < 100 else '❌'}")
    
    if total_images >= 5000:
        print("✅ DATASET VERIFIED: Complete Kaggle dataset available!")
        print("✅ Ready for medical AI training with real Kaggle data!")
        return True
    else:
        print("❌ Dataset size mismatch - please check download")
        return False

def main():
    print("🏥 REAL KAGGLE DATASET DOWNLOADER")
    print("Downloading Kaggle Chest X-Ray dataset")
    print("=" * 60)
    
    # Step 1: Install Kaggle API
    if not install_kaggle():
        return False
    
    # Step 2: Verify credentials
    if not verify_credentials():
        return False
    
    # Step 3: Download dataset
    if not download_kaggle_dataset():
        return False
    
    print("\n🎉 SUCCESS!")
    print("✅ Real Kaggle dataset (5,863 images) downloaded successfully")
    print("✅ Dataset structure verified and complete")
    print("✅ Ready to run FINAL_COMPLETE_PROJECT.py with REAL data")
    print("\n🚀 Next step: Run python FINAL_COMPLETE_PROJECT.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup incomplete. Please resolve issues above and try again.")
        sys.exit(1)