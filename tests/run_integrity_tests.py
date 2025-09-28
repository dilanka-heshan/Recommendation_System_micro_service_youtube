#!/usr/bin/env python3
"""
Database Integrity Test Runner
Simple script to run database integrity tests with better error handling
"""
import subprocess
import sys
import os

def run_database_tests():
    """Run database integrity tests with improved error handling"""
    
    print("🔍 Running Database Integrity Tests")
    print("=" * 50)
    
    # Test categories to run individually  
    test_categories = [
        ("Constraint Validation", "tests/database_integrity/test_constraint_validation.py"),
        ("Cross-Database Integrity", "tests/database_integrity/test_cross_database_integrity.py"), 
        ("Data Consistency", "tests/database_integrity/test_data_consistency.py"),
        ("Orphaned Records", "tests/database_integrity/test_orphaned_records.py"),
        ("Data Freshness", "tests/database_integrity/test_data_freshness.py"),
        ("Health Monitoring", "tests/database_integrity/test_health_monitoring.py")
    ]
    
    results = {}
    
    for category_name, test_path in test_categories:
        print(f"\n🧪 Running {category_name} Tests...")
        print("-" * 40)
        
        try:
            # Run the test with minimal output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_path, 
                "-v", 
                "--tb=short",
                "--disable-warnings"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                # Count passed tests
                passed_count = result.stdout.count(" PASSED")
                results[category_name] = {"status": "✅ PASSED", "count": passed_count, "details": None}
                print(f"✅ {category_name}: {passed_count} tests PASSED")
            else:
                # Count failed tests
                failed_count = result.stdout.count(" FAILED")
                passed_count = result.stdout.count(" PASSED")
                results[category_name] = {
                    "status": "⚠️ MIXED" if passed_count > 0 else "❌ FAILED", 
                    "count": f"{passed_count} passed, {failed_count} failed",
                    "details": result.stdout.split('\n')[-10:]  # Last 10 lines
                }
                if passed_count > 0:
                    print(f"⚠️ {category_name}: {passed_count} passed, {failed_count} failed")
                else:
                    print(f"❌ {category_name}: {failed_count} tests failed")
                    
        except Exception as e:
            results[category_name] = {"status": "🔥 ERROR", "count": 0, "details": str(e)}
            print(f"🔥 {category_name}: Error - {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    total_passed = 0
    total_categories = len(test_categories)
    
    for category, result in results.items():
        status_icon = result["status"].split()[0]
        print(f"{status_icon} {category}: {result['count']}")
        if result["status"].startswith("✅"):
            total_passed += 1
    
    print(f"\n🎯 Overall: {total_passed}/{total_categories} test categories fully passed")
    
    if total_passed == total_categories:
        print("🎉 All database integrity tests are working perfectly!")
        return 0
    else:
        print(f"⚠️ {total_categories - total_passed} categories need attention")
        print("\n💡 Note: Some 'failures' may be data quality insights, not actual test bugs")
        return 1

if __name__ == "__main__":
    exit_code = run_database_tests()
    sys.exit(exit_code)