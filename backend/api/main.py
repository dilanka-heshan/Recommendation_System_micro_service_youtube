# Load environment variables from .env file first, before any other imports
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.api.routes import recommendations, run_workflow, newsletter, user_vector_update

app = FastAPI(title="YouTube Recommendation Services", version="1.0.0")

#Pydentic request model

app.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(run_workflow.router, prefix="/run-workflow", tags=["run-workflow"])
app.include_router(newsletter.router, prefix="/newsletter", tags=["newsletter"])
app.include_router(user_vector_update.router, prefix="/user-vector-update", tags=["user-vector-update"])


@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)