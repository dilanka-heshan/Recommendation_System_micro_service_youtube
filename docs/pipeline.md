┌───────────────────────────────────────────────────────────────┐
│ [User Preferences Fetch] │
│ From Supabase: preference, embedding, previous high-rating │
│ feedback video id and title │
└──────────────┬────────────────────────────────────────────────┘
│ state.user_prefs
▼
┌───────────────────────────────────────────────────────────────┐
│ [Vectorize Preferences] │
│ Generate query embedding (user preference) │
└──────────────┬────────────────────────────────────────────────┘
│ state.user_embedding
▼
┌───────────────────────────────────────────────────────────────┐
│ Hybrid Retrieval Node │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Vector DB Search → Top-N video embeddings (semantic) │ │
│ │ Title Search → Top-M exact matches │ │
│ │ Merge & deduplicate candidate videos │ │
│ └─────────────────────────────────────────────────────────┘ │
└──────────────┬────────────────────────────────────────────────┘
│ state.candidate_videos
▼
┌───────────────────────────────────────────────────────────────┐
│ [Apply Time Decay] │
│ Boost recent videos │
└──────────────┬────────────────────────────────────────────────┘
│ state.boosted_candidates
▼
┌───────────────────────────────────────────────────────────────┐
│ [BAAI Pairwise Reranker] │
│ Compare candidates with user watch history → score │
└──────────────┬────────────────────────────────────────────────┘
│ state.reranked_videos
▼
┌───────────────────────────────────────────────────────────────┐
│ [Diversity Filter / MMR] │
│ Ensure variety in topics & channels │
└──────────────┬────────────────────────────────────────────────┘
│ state.final_list
▼
┌───────────────────────────────────────────────────────────────┐
│ [Store Recommendations] │
│ Save top-K in Supabase for frontend delivery │
└──────────────┬────────────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────────────────┐
│ [User Feedback Collection] │
│ Clicks / Watches / Likes │
└──────────────┬────────────────────────────────────────────────┘
│ feedback_vector
▼
┌───────────────────────────────────────────────────────────────┐
│ [Update User Embedding] │
│ Incorporate feedback into Supabase-stored embedding │
└──────────────┬────────────────────────────────────────────────┘
│
└────────────────────────────────────┐
│ (feeds future queries immediately)
▼
┌─────────────────────┐
│ Weekly Pipeline │
│ Loop: Re-run full │
│ process │
└─────────────────────┘
