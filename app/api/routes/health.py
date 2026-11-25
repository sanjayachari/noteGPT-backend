
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

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
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
