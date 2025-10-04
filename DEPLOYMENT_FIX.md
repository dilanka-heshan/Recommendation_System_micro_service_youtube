# Google Cloud Run Deployment Fix - Rerank Module Issue

## Problem
When deploying to Google Cloud Run, the application was failing with a "no rerank module" error. The rerank module exists in `backend/services/rerank.py` but Python couldn't find it during import.

## Root Cause
**Missing `__init__.py` Files**: Several backend directories were missing `__init__.py` files, which are required for Python to recognize directories as packages. Without these files, Python's module system cannot resolve imports like:
- `from backend.services.rerank import video_reranker`
- `from backend.database.mongodb_client import mongodb_client`
- `from backend.pipelines.rerank_videos_node import rerank_videos_node`

## Files Created
The following `__init__.py` files have been created to fix the module recognition issue:

1. **`backend/__init__.py`** - Root backend package
2. **`backend/services/__init__.py`** - Services package (contains rerank.py)
3. **`backend/pipelines/__init__.py`** - Pipelines package
4. **`backend/database/__init__.py`** - Database package
5. **`backend/utils/__init__.py`** - Utils package
6. **`backend/api/routes/__init__.py`** - API routes package

## Why This Works in Development but Failed in Cloud Run
- **Local Development**: Python sometimes uses the current working directory in sys.path, which can allow imports to work even without proper `__init__.py` files
- **Docker/Cloud Run**: Stricter Python path resolution requires proper package structure with `__init__.py` files

## Verification Steps
After deploying with these changes:

1. The import chain should now work:
   ```python
   from backend.services.rerank import video_reranker
   ```

2. The rerank model should load properly in the VideoReranker class

3. All other backend imports should also work correctly

## Additional Notes
- The `requirements.txt` already includes `sentence-transformers>=2.2.0,<3.0.0` which is correct
- The Dockerfile properly bundles the BAAI/bge-reranker-base model
- The rerank.py has proper error handling for when dependencies are not available

## Next Steps
1. Commit these new `__init__.py` files to your repository
2. Rebuild your Docker image
3. Redeploy to Google Cloud Run
4. The rerank module should now be recognized and imported successfully
