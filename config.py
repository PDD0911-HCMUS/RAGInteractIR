import os


class ConfigDB:
    HOSTNAME = os.environ.get("POSTGRES_HOST", "localhost")
    DATABASE = os.environ.get("POSTGRES_DB", "RetrievalSystemTraffic")
    USERNAME = os.environ.get("POSTGRES_USER", "postgres")
    PASSWORD = os.environ.get("POSTGRES_PASSWORD", "09111997")
    PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"postgresql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
