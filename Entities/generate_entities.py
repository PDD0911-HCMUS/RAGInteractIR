import os
import subprocess
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import ConfigDB


def generate_models():
    output_file = Path(__file__).resolve().parent / "entities.py"
    db_uri = ConfigDB.SQLALCHEMY_DATABASE_URI

    command = [
        "sqlacodegen",
        db_uri,
        "--outfile",
        str(output_file),
    ]

    print("Generating SQLAlchemy entities...")
    print(f"Output: {output_file}")
    subprocess.run(command, check=True)
    print("Done.")


if __name__ == "__main__":
    generate_models()