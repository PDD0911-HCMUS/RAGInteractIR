class ConfigDB:
    HOSTNAME = "localhost"
    DATABASE = "RetrievalSystemTraffic"
    USERNAME = "postgres"
    # PASSWORD = "123456"
    PASSWORD = "09111997"
    PORT = 5432 #5432 for Local #5433 for csMachine
    SQLALCHEMY_DATABASE_URI = f"postgresql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False