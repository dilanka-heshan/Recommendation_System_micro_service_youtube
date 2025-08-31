# Daily User Vector Update Pipeline - Rocchio's Algorithm Implementation Plan

## Overview

This document outlines the i### State Management (`UserVectorUpdateState`)

````python
class UserVectorUpdateState(TypedDict):
    date_range: Dict[str, str]  # start_date, end_date
    user_feedback_data: Dict[str, List[Dict]]  # user_id -> [{"video_id": str, "rating": int, "timestamp": str}]
    user_embedding_ids: Dict[str, str]  # user_id -> embedding_id
    current_user_vectors: Dict[str, List[float]]  # embedding_id -> current_vector
    video_embeddings: Dict[str, List[float]]  # video_id -> get from qdrant
    updated_user_vectors: Dict[str, List[float]]  # embedding_id -> new_vector  replace previous vector with new vector
    new_embedding_ids: Dict[str, str]  # user_id -> new_embedding_id (for new users)
    pipeline_metrics: Dict[str, Any]  # execution statistics
    errors: List[str]  # error tracking
``` ion plan for a daily automated user vector update pipeline using Rocchio's Algorithm. The pipeline will analyze user feedback and clicked newsletter videos to update user preference vectors for improved personalized recommendations.

## Architecture Overview

### Core Components
1. **Daily Scheduler Pipeline** - LangGraph workflow for automatic execution
2. **Feedback Analysis Module** - Process user feedback data
3. **Click Analysis Module** - Analyze newsletter video clicks
4. **Rocchio Vector Update Engine** - Implement Rocchio's algorithm
5. **Vector Database Integration** - Retrieve/update vectors in Qdrant
6. **Database Integration** - Interface with feedback and newsletter_videos tables

## Data Sources

### Primary Tables
1. **feedback** table
   - Fields: user_id, video_id, rating (1-5 scale), timestamp
   - Purpose: Explicit user feedback signals (1=strongly dislike, 5=strongly like)

2. **newsletter_videos** table
   - Fields: user_id, video_id, clicked
   - Purpose: Implicit feedback from newsletter interactions

3. **users** table
   - Fields: user_id, embedding_id, other_user_data
   - Purpose: Store user preference vectors (embedding_id)

4. **Qdrant Vector Database**
   - Collection: video_title_desc (video embeddings)
   - Purpose: Video and user embeddings storage and retrieval

## Rocchio's Algorithm Implementation

### Mathematical Foundation
````

Updated*User_Vector = α * Original*User_Vector + β * Relevant_Docs_Centroid - γ \* Non_Relevant_Docs_Centroid

````

### Parameters Configuration
- **α (Alpha)**: 0.7 - Weight for original user vector (maintain historical preferences)
- **β (Beta)**: 0.3 - Weight for positive feedback vectors (ratings 4-5)
- **γ (Gamma)**: 0.1 - Weight for negative feedback vectors (ratings 1-2)

### Rating-Based Weight System
```python
rating_weights = {
    5: 1.0,    # Strongly positive - full weight
    4: 0.75,   # Positive - reduced weight
    3: 0.0,    # Neutral - ignored or minimal weight
    2: 0.75,   # Negative - reduced weight
    1: 1.0     # Strongly negative - full weight
}

# Classification for Rocchio's algorithm
positive_ratings = [4, 5]  # Used in β term
negative_ratings = [1, 2]  # Used in γ term
neutral_ratings = [3]      # Ignored or minimal impact
````

### Feedback Classification

1. **Positive Signals**:

   - High ratings (rating >= 4) from feedback table
   - Newsletter video clicks (clicked = true)
   - Strong positive preference (rating = 5)

2. **Negative Signals**:

   - Low ratings (rating <= 2) from feedback table
   - Newsletter videos sent but not clicked
   - Strong negative preference (rating = 1)

3. **Neutral Signals**:
   - Medium ratings (rating = 3) - treated as weak positive or ignored
   - Sent newsletters with no interaction data

## Pipeline Architecture

### LangGraph Workflow Nodes

#### 1. Data Extraction Node (`extract_daily_feedback_node.py`)

**Functionality:**

- Query feedback table for yesterday's user ratings (1-5 scale)
- Query newsletter_videos table for yesterday's sent newsletters (in news letter table exist a column like sent at)
- Query users table to get current embedding_ids for active users
- Aggregate data by user_id with rating-based weighting
- Filter for active users with feedback/clicks

**Input:** Date range (yesterday)
**Output:** User feedback data with ratings and current embedding_ids

#### 2. User Vector Retrieval Node (`retrieve_user_vectors_node.py`)

**Functionality:**

- Extract embedding_ids from users table for active users
- Retrieve current user preference vectors from supabase users embedding_id (it contain user vector) user_embeddings collection
- Extract unique video_ids from feedback data (yesterday one)
- Batch retrieve video embeddings from Qdrant video collection
- Create mappings: user_id -> current_vector, video_id -> embedding
- Handle missing embeddings gracefully (new users, missing videos)

**Input:** User user_id and video_ids lists
**Output:** Current user vector (ffrom supabase) and video embeddings dictionaries (from qdrant)

#### 3. User Vector Calculation Node (`calculate_user_vectors_node.py`)

**Functionality:**

- Group feedback by user_id with rating-based weights
- Calculate weighted positive feedback centroids (ratings 4-5)
- Calculate weighted negative feedback centroids (ratings 1-2)
- Apply rating-specific weights: rating*5 * 1.0, rating*4 * 0.75, rating*2 * 0.75, rating*1 * 1.0
- Apply Rocchio's algorithm with current user vectors from users table
- Handle edge cases (no feedback, only positive/negative, new users)

**Input:** User feedback with ratings + current user vectors + video embeddings
**Output:** Updated user vectors dictionary with embedding_ids

#### 4. Vector Storage Node (`store_updated_vectors_node.py`)

**Functionality:**

- Update user vectors in supabase users table fields embedding_ids (it contain 768 dimension vector)
- Create new user vectors for first-time users (assign new embedding_ids)
- Log update statistics and success rates
- Handle batch updates for performance
- Maintain embedding_id consistency between users table

**Input:** Updated user vectors with embedding_ids
**Output:** Update confirmation, new embedding_ids, and statistics

#### 5. Monitoring Node (`monitor_update_pipeline_node.py`)

**Functionality:**

- Track pipeline execution metrics
- Log user update counts and success rates
- Generate daily reports
- Alert on failures or anomalies

**Input:** Pipeline execution data
**Output:** Monitoring reports and alerts

### State Management (`UserVectorUpdateState`)

```python
class UserVectorUpdateState(TypedDict):
    date_range: Dict[str, str]  # start_date, end_date
    user_feedback_data: Dict[str, List[Dict]]  # user_id -> feedback list
    video_embeddings: Dict[str, List[float]]  # video_id -> embedding
    updated_user_vectors: Dict[str, List[float]]  # user_id -> new vector
    pipeline_metrics: Dict[str, Any]  # execution statistics
    errors: List[str]  # error tracking
```

## Database Schema Requirements

### User Vectors Storage in supabase

There is column in users table called embedding\_\_id it contain vector
