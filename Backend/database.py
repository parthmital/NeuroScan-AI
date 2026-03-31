import os
from sqlmodel import create_engine, Session, SQLModel
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

sqlite_file_name = "neuroscan.db"
base_dir = os.path.dirname(os.path.abspath(__file__))
default_url = f"sqlite:///{os.path.join(base_dir,sqlite_file_name)}"
sqlite_url = os.getenv("DATABASE_URL", default_url)
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
