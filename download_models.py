#!/usr/bin/env python3
"""
Script to download and bundle models for the Docker image.
This is executed during the Docker build process.
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    
    print('Downloading and bundling BAAI/bge-base-en embedding model...')
    model1 = SentenceTransformer('BAAI/bge-base-en', trust_remote_code=True)
    model1.save('/app/models/bge-base-en')
    print('BAAI/bge-base-en model bundled successfully')
    
    print('Downloading and bundling BAAI/bge-reranker-base model...')
    model2 = CrossEncoder('BAAI/bge-reranker-base', trust_remote_code=True, tokenizer_args={'use_fast': False})
    model2.save('/app/models/bge-reranker-base')
    print('BAAI/bge-reranker-base model bundled successfully')
    
    print('Verifying models...')
    assert os.path.exists('/app/models/bge-base-en'), 'Embedding model not found'
    assert os.path.exists('/app/models/bge-reranker-base'), 'Reranker model not found'
    print('All models verified successfully')
    
    sys.exit(0)
    
except Exception as e:
    print(f'Model bundling failed: {e}. Models will be downloaded at runtime.')
    import traceback
    traceback.print_exc()
    sys.exit(0)  # Exit with success to continue the build
