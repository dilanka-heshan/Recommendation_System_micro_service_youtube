@echo off
REM YouTube Recommendation Service - Docker Startup Script for Windows

echo 🚀 Starting YouTube Recommendation Service...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from template...
    copy .env.example .env
    echo 📝 Please edit .env file with your actual API keys and database URLs
    echo    Required variables:
    echo    - SUPABASE_URL
    echo    - SUPABASE_ANON_KEY
    echo    - QDRANT_HOST
    echo    - QDRANT_API_KEY
    echo    - MONGODB_CONNECTION_STRING
    echo    - OPENAI_API_KEY
    echo.
    pause
)

REM Build and start services
echo 🔨 Building and starting services...
docker-compose up --build -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check API health
curl -f http://localhost:8001/health >nul 2>&1
if errorlevel 1 (
    echo ❌ API service is not responding
    docker-compose logs youtube-recommendation-api
) else (
    echo ✅ API service is healthy
)

echo.
echo 🎉 Service started successfully!
echo 📖 API Documentation: http://localhost:8001/docs
echo 🔍 Health Check: http://localhost:8001/health
echo.
echo 📋 To view logs: docker-compose logs -f
echo 🛑 To stop services: docker-compose down
pause
