from Database.db_session import SessionLocal
from Database.base_repository import BaseRepository


class UnitOfWork:
    def __enter__(self):
        self.session = SessionLocal()
        self.repo = BaseRepository(self.session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()