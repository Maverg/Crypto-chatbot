"""
Setup script for mocking environment.
"""

import os
import sys
from create_mock_data import create_mock_database

def clean_previous_mock_data():
    """Clean up any previous mock data."""
    print("🧹 Cleaning up previous mock data...")
    
    files_to_remove = [
        "mocking_articles.db",
        "mock_vector_store"
    ]
    
    for file_or_dir in files_to_remove:
        if os.path.exists(file_or_dir):
            if os.path.isdir(file_or_dir):
                import shutil
                shutil.rmtree(file_or_dir)
                print(f"  Removed directory: {file_or_dir}")
            else:
                os.remove(file_or_dir)
                print(f"  Removed file: {file_or_dir}")

def setup_mocking_environment():
    """Set up the complete mocking environment."""
    print("🧪 Setting up Mocking Environment")
    print("=" * 50)
    
    # Change to mocking directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Clean previous data
    clean_previous_mock_data()
    
    # Create mock database
    db = create_mock_database()
    
    # Test database functionality
    print("\n🔧 Testing database functionality...")
    
    # Test queries for different time ranges
    for days in [1, 3, 7, 14]:
        articles = db.get_articles(days_back=days)
        print(f"📊 Articles (last {days} days): {len(articles)}")
    
    print("\n✅ Mocking environment ready!")
    print("🚀 You can now run:")
    print("   python chat.py")
    print("   python test_database.py")
    print("   python fetcher_service.py (mock mode)")

if __name__ == "__main__":
    setup_mocking_environment()