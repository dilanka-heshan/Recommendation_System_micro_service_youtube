#!/usr/bin/env python3
"""
Test script to verify that all backend modules can be imported correctly.
This helps diagnose module import issues before deploying to Google Cloud Run.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test all critical backend imports"""
    print("Testing backend module imports...")
    print("-" * 50)
    
    tests = [
        ("backend", "Backend root package"),
        ("backend.services", "Services package"),
        ("backend.services.rerank", "Rerank service module"),
        ("backend.pipelines", "Pipelines package"),
        ("backend.pipelines.rerank_videos_node", "Rerank videos node"),
        ("backend.database", "Database package"),
        ("backend.database.mongodb_client", "MongoDB client"),
        ("backend.database.qdrant_client", "Qdrant client"),
        ("backend.models", "Models package"),
        ("backend.api", "API package"),
        ("backend.api.routes", "API routes package"),
        ("backend.utils", "Utils package"),
    ]
    
    success_count = 0
    fail_count = 0
    
    for module_name, description in tests:
        try:
            __import__(module_name)
            print(f"✓ {description:40s} - OK")
            success_count += 1
        except ImportError as e:
            print(f"✗ {description:40s} - FAILED: {e}")
            fail_count += 1
        except Exception as e:
            print(f"✗ {description:40s} - ERROR: {e}")
            fail_count += 1
    
    print("-" * 50)
    print(f"\nResults: {success_count} passed, {fail_count} failed")
    
    if fail_count == 0:
        print("✓ All imports successful! Ready for deployment.")
        return True
    else:
        print("✗ Some imports failed. Please fix before deploying.")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
