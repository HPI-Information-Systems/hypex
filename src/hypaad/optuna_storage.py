import logging
import typing as t
from enum import Enum
from time import sleep

import numpy as np
import optuna
from MySQLdb import Connection as MySQLConnection
from redis import Redis

__all__ = ["OptunaStorage", "OptunaStorageType"]

_logger = logging.getLogger(__name__)


class OptunaStorageType(Enum):
    REDIS = "redis"
    MYSQL = "mysql"


class OptunaStorage:
    def __init__(
        self,
        type: OptunaStorageType,
        host: str,
        port: int,
        db: int,
        user: t.Optional[str] = None,
    ):
        self.type = type
        self.host = host
        self.db = db
        self.user = user

        self.port = port
        self.storage_backend = None

    def get_uri(self):
        if self.type == OptunaStorageType.REDIS:
            return f"redis://{self.host}:{self.port}/{self.db}"
        elif self.type == OptunaStorageType.MYSQL:
            return f"mysql://{self.user}@{self.host}:{self.port}/{self.db}"
        else:
            raise ValueError(f"Unknown storage type: {self.type}")

    @classmethod
    def get_port(cls):
        return np.random.randint(low=1000, high=10000)

    def get_ping_func(self):
        if self.type == OptunaStorageType.REDIS:
            url = self.get_uri()
            return Redis.from_url(url, socket_connect_timeout=1).ping
        elif self.type == OptunaStorageType.MYSQL:

            def _connect():
                conn = MySQLConnection(
                    host=self.host, port=self.port, user=self.user, connect_timeout=1
                )
                conn.query("CREATE DATABASE IF NOT EXISTS {}".format(self.db))
                conn.query("SET GLOBAL max_connections = 99999999;")

            return _connect
        else:
            raise ValueError(f"Unknown storage type: {self.type}")

    def get_docker_command(self):
        if self.type == OptunaStorageType.REDIS:
            return f"docker run -p 0.0.0.0:{self.port}:6379 -d redis:latest"
        elif self.type == OptunaStorageType.MYSQL:
            return f"docker run -p 0.0.0.0:{self.port}:3306 -e MYSQL_ALLOW_EMPTY_PASSWORD=1 -d mysql:latest"
        else:
            raise ValueError(f"Unknown storage type: {self.type}")

    def get_storage_backend(self):
        if self.storage_backend is None:
            if self.type == OptunaStorageType.REDIS:
                self.storage_backend = optuna.storages.RedisStorage(url=self.get_uri())
            elif self.type == OptunaStorageType.MYSQL:
                self.storage_backend = optuna.storages.RDBStorage(url=self.get_uri())
            else:
                raise ValueError(f"Unknown storage type: {self.type}")
        return self.storage_backend

    @classmethod
    def get(
        cls, scheduler_host: str, port: int, storage_type: OptunaStorageType
    ) -> "OptunaStorage":
        if storage_type == OptunaStorageType.REDIS:
            return OptunaStorage(
                type=storage_type,
                host=scheduler_host,
                port=port,
                db="hypex",
            )
        elif storage_type == OptunaStorageType.MYSQL:
            return OptunaStorage(
                type=storage_type,
                host=scheduler_host,
                port=port,
                db="hypex",
                user="root",
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
