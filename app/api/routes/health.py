
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.groq_llm import MODEL_NAME

router = APIRouter()


@router.get("/health/db")
def database_health(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT version()"))
        version = result.scalar()

        return {
            "status": "connected",
            "database": "PostgreSQL",
            "version": version,
        }
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "Database connection failed",
            },
        )
