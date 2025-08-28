# YouTube Recommendation System

A sophisticated LangGraph-based recommendation pipeline for YouTube content curation with vector similarity search, two-stage reranking, and newsletter generation.

## Features

- **Vector-based Retrieval**: Uses Qdrant for efficient similarity search
- **Two-stage Reranking**: Cross-encoder + pairwise reranking with MongoDB extractive summaries
- **Diversity Filtering**: MMR-based diversity optimization
- **Newsletter Generation**: Automated newsletter creation and storage
- **Modular Architecture**: Clean LangGraph pipeline with separate nodes

## Architecture

```
User Request → Fetch User Data → Vector Retrieval → Rerank Videos → Diversity Filter → Store Newsletter
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Qdrant
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# MongoDB
MONGODB_URI=your_mongodb_uri
```

3. Run the pipeline:

```python
from src.pipelines.orchestrator import run_recommendation_pipeline

result = await run_recommendation_pipeline(user_id="user123")
```

## Components

### Pipelines

- `orchestrator.py` - Main workflow coordination
- `fetch_user_data_node.py` - User preferences retrieval
- `vector_retrieval_node.py` - Vector similarity search
- `rerank_videos_node.py` - Two-stage reranking
- `diversity_filter_node.py` - MMR diversity filtering
- `store_newsletter_node.py` - Newsletter storage

### Services

- `rerank.py` - Advanced reranking with MongoDB integration
- `retrieval_service.py` - Vector search operations
- `user_preferences_service.py` - User preference management

### Database Clients

- `supabase_client.py` - User data and newsletter storage
- `qdrant_client.py` - Vector database operations
- `mongodb_client.py` - Extractive summaries retrieval

## API Endpoints

- `POST /api/recommendations/run-workflow` - Trigger recommendation pipeline
- `POST /api/newsletters/` - Create newsletter
- `GET /api/newsletters/{newsletter_id}` - Get newsletter details
