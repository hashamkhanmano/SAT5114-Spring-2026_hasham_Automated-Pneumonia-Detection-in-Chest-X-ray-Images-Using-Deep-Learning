"""
KAGGLE CREDENTIALS SETUP HELPER
This script helps you set up Kaggle API credentials properly.
"""

import os
import json

def setup_kaggle_credentials():
    """Guide user through Kaggle credentials setup"""
    print("🔑 KAGGLE API CREDENTIALS SETUP")
    print("=" * 50)
    
    # Check if credentials already exist
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_json):
        print("✅ Kaggle credentials already found!")
        print(f"📁 Location: {kaggle_json}")
        
        # Verify credentials format
        try:
            with open(kaggle_json, 'r') as f:
                creds = json.load(f)
            if 'username' in creds and 'key' in creds:
                print("✅ Credentials format is valid")
                print(f"👤 Username: {creds['username']}")
                return True
            else:
                print("❌ Invalid credentials format")
        except:
            print("❌ Error reading credentials file")
    
    print("\n🚨 KAGGLE CREDENTIALS SETUP REQUIRED:")
    print("1. Go to: https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print(f"5. Create directory: {kaggle_dir}")
    print(f"6. Move kaggle.json to: {kaggle_json}")
    print("7. Run this script again to verify")
    
    # Create directory if it doesn't exist
    os.makedirs(kaggle_dir, exist_ok=True)
    print(f"\n📁 Created directory: {kaggle_dir}")
    
    return False

def test_kaggle_connection():
    """Test if Kaggle API works"""
    try:
        import kaggle
        print("\n🧪 Testing Kaggle API connection...")
        
        # Try to list datasets (this will fail if credentials are wrong)
        kaggle.api.authenticate()
        print("✅ Kaggle API authentication successful!")
        
        # Try to access the specific dataset we need
        dataset_name = "paultimothymooney/chest-xray-pneumonia"
        print(f"🔍 Checking access to dataset: {dataset_name}")
        
        dataset_info = kaggle.api.dataset_list_files(dataset_name)
        print("✅ Dataset access confirmed!")
        print(f"📊 Dataset contains {len(dataset_info)} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Kaggle API test failed: {e}")
        print("Please check your credentials and try again.")
        return False

if __name__ == "__main__":
    print("🏥 PNEUMONIA DETECTION PROJECT - KAGGLE SETUP")
    print("Setting up access to real medical dataset (5,863 images)")
    print("=" * 60)
    
    # Step 1: Setup credentials
    if setup_kaggle_credentials():
        # Step 2: Test connection
        if test_kaggle_connection():
            print("\n🎉 SUCCESS! Kaggle is ready!")
            print("You can now run FINAL_COMPLETE_PROJECT.py to download the real dataset")
        else:
            print("\n⚠️ Credentials found but connection failed")
            print("Please verify your kaggle.json file is correct")
    else:
        print("\n⏳ Please complete the setup steps above and run this script again")