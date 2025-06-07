import sys
import time
from functools import lru_cache, wraps
from urllib.parse import quote_plus

import sqlalchemy as sa
from dotenv import dotenv_values
from pymongo import MongoClient


def timer(start: float, end: float):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def elapsed_time(start: float):
    return timer(start, time.time())


@lru_cache
def load_dotenv_in_config():
    config = dotenv_values(".env", verbose=True, encoding="utf-8")
    return config


@lru_cache
def get_postgresql_engine():
    config = load_dotenv_in_config()

    conn_url = sa.URL.create(
        drivername="postgresql+psycopg2",
        username=config["PG_USERNAME"],
        password=config["PG_PASSWORD"],
        host=config["PG_HOST"],
        database=config["PG_DATABASE"],
        port=int(config["PG_PORT"]),
    )

    engine = sa.create_engine(conn_url, pool_size=128, max_overflow=256)
    return engine


@lru_cache
def get_mongo_client():
    config = load_dotenv_in_config()
    uri = "mongodb://%s:%s@%s" % (
        quote_plus(config["MONGO_USERNAME"]),
        quote_plus(config["MONGO_PASSWORD"]),
        config["MONGO_HOST"],
    )
    client = MongoClient(host=uri, port=int(config["MONGO_PORT"]))
    return client


def get_mongo_database(client: MongoClient):
    config = load_dotenv_in_config()
    db = client.get_database(config["MONGO_DATABASE"])
    return db


def logfile_enabled(prefix: str):
    """
    Redirects the print statements to a log file.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")

            # Backup the original stdout
            ori_stdout = sys.stdout

            try:
                with open(f"{prefix}-{timestr}.log", "a+") as f:
                    sys.stdout = f
                    return func(*args, **kwargs)
            finally:
                # Restore the original stdout
                sys.stdout = ori_stdout

        return wrapper

    return decorator
