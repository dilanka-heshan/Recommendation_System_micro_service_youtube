# Rerank Model Loading Fix - use_auth_token Deprecation

## Error in Google Cloud Run Logs

```
Failed to load from HuggingFace: CrossEncoder.__init__() got an unexpected keyword argument 'use_auth_token'
Failed to load reranking models from both local and HuggingFace
Reranking models not available, using fallback
```

## Root Cause

The `use_auth_token` parameter has been **deprecated** in newer versions of `sentence-transformers` library (version 2.2.0+). The library now uses `token` parameter instead, or the parameter can be omitted entirely when not using private models.

## Files Fixed

### 1. `backend/services/rerank.py`

**Changed:**

- ❌ Before: `SentenceTransformer(..., use_auth_token=False)`
- ✅ After: `SentenceTransformer(..., trust_remote_code=True)`
- ❌ Before: `CrossEncoder(..., use_auth_token=False)`
- ✅ After: `CrossEncoder(..., trust_remote_code=True)`

### 2. `Dockerfile`

**Changed:**

- ❌ Before: `SentenceTransformer('BAAI/bge-base-en', trust_remote_code=True, use_auth_token=False)`
- ✅ After: `SentenceTransformer('BAAI/bge-base-en', trust_remote_code=True)`
- ❌ Before: `CrossEncoder('BAAI/bge-reranker-base', trust_remote_code=True, use_auth_token=False)`
- ✅ After: `CrossEncoder('BAAI/bge-reranker-base', trust_remote_code=True)`

## Why This Happened

- The `sentence-transformers` library updated its API
- The `use_auth_token` parameter was replaced with `token` parameter
- For public models (like BAAI/bge-base-en and BAAI/bge-reranker-base), authentication is not needed, so the parameter can be omitted

## Impact

- ✅ Models will now load successfully during Docker build
- ✅ Models will load correctly at runtime in Google Cloud Run
- ✅ Reranking functionality will work properly
- ✅ No more "Reranking models not available, using fallback" warnings

## Testing

After deployment, you should see in the logs:

```
BGE models loaded successfully from bundled path
```

OR

```
BGE models loaded successfully from HuggingFace
```

Instead of the error messages.

## References

- sentence-transformers documentation: https://www.sbert.net/
- The parameter change was introduced to align with HuggingFace transformers library conventions
