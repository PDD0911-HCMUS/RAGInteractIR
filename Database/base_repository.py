from sqlalchemy import insert, select, update, delete
from Entities.metadata_registry import MetadataRegistry


class BaseRepository:
    def __init__(self, session):
        self.session = session
        self.registry = MetadataRegistry()

    def get_table(self, table_name: str):
        return self.registry.get_table(table_name)

    def bulk_insert(self, table_name: str, rows: list[dict]):
        if not rows:
            return 0

        table = self.get_table(table_name)
        self.session.execute(insert(table), rows)
        return len(rows)

    def insert_one(self, table_name: str, row: dict):
        table = self.get_table(table_name)
        self.session.execute(insert(table).values(**row))
        return 1

    def find_all(self, table_name: str):
        table = self.get_table(table_name)
        stmt = select(table)
        return self.session.execute(stmt).fetchall()

    def find_by(self, table_name: str, filters: dict):
        table = self.get_table(table_name)
        stmt = select(table)

        for key, value in filters.items():
            stmt = stmt.where(table.c[key] == value)

        return self.session.execute(stmt).fetchall()

    def exists(self, table_name: str, filters: dict) -> bool:
        table = self.get_table(table_name)
        stmt = select(table)

        for key, value in filters.items():
            stmt = stmt.where(table.c[key] == value)

        result = self.session.execute(stmt).first()
        return result is not None

    def delete_by(self, table_name: str, filters: dict):
        table = self.get_table(table_name)
        stmt = delete(table)

        for key, value in filters.items():
            stmt = stmt.where(table.c[key] == value)

        result = self.session.execute(stmt)
        return result.rowcount

    def update_by(self, table_name: str, filters: dict, values: dict):
        table = self.get_table(table_name)
        stmt = update(table)

        for key, value in filters.items():
            stmt = stmt.where(table.c[key] == value)

        stmt = stmt.values(**values)
        result = self.session.execute(stmt)
        return result.rowcount