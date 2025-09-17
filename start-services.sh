#!/bin/bash

# YouT   echo "   Required variables:"
   echo "   - SUPABASE_URL"
   echo "   - SUPABASE_ANON_KEY" 
   echo "   - QDRANT_HOST"
   echo "   - QDRANT_API_KEY"
   echo "   - MONGODB_CONNECTION_STRING"
   echo "   - OPENAI_API_KEY"Recommendation Service - Docker Startup Script

set -e

echo "🚀 Starting YouTube Recommendation Service..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your actual API keys and database URLs"
    echo "   Required variables:"
    echo "   - SUPABASE_URL"
    echo "   - SUPABASE_ANON_KEY" 
    echo "   - OPENAI_API_KEY"
    echo ""
    read -p "Press Enter after you've configured the .env file..."
fi

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check API health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "❌ API service is not responding"
    docker-compose logs youtube-recommendation-api
fi

echo ""
echo "🎉 Service started successfully!"
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""
echo "📋 To view logs: docker-compose logs -f"
echo "🛑 To stop services: docker-compose down"
