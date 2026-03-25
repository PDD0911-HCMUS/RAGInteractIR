from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import ConfigDB

engine = create_engine(ConfigDB.SQLALCHEMY_DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)