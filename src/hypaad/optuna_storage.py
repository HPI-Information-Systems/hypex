import typing as t
from enum import Enum

import numpy as np
import optuna
from MySQLdb import Connection as MySQLConnection
from redis import Redis

__all__ = ["OptunaStorage", "OptunaStorageType"]


class OptunaStorageType(Enum):
    REDIS = "redis"
    MYSQL = "mysql"


class OptunaStorage:
    def __init__(
        self,
        type: OptunaStorageType,
        host: str,
        db: int,
        user: t.Optional[str] = None,
    ):
        self.type = type
        self.host = host
        self.db = db
        self.user = user

        self.port = None
        self.storage_backend = None

    def get_uri(self):
        port = self.get_port()
        if self.type == OptunaStorageType.REDIS:
            return f"redis://{self.host}:{port}/{self.db}"
        elif self.type == OptunaStorageType.MYSQL:
            return f"mysql://{self.user}@{self.host}:{port}/{self.db}"
        else:
            raise ValueError(f"Unknown storage type: {self.type}")

    def get_port(self):
        if self.port is None:
            self.port = np.random.randint(low=1000, high=10000)
        return self.port

    def get_ping_func(self):
        if self.type == OptunaStorageType.REDIS:
            url = self.get_uri()
            return Redis.from_url(url, socket_connect_timeout=1).ping
        elif self.type == OptunaStorageType.MYSQL:
            port = self.get_port()

            def _connect():
                conn = MySQLConnection(
                    host=self.host, port=port, user=self.user, connect_timeout=1
                )
                conn.query("CREATE DATABASE IF NOT EXISTS {}".format(self.db))
                conn.query("SET GLOBAL max_connections = 10000000;")

            return _connect
        else:
            raise ValueError(f"Unknown storage type: {self.type}")

    def get_docker_command(self):
        port = self.get_port()
        if self.type == OptunaStorageType.REDIS:
            return f"docker run -p 0.0.0.0:{port}:6379 -d redis:latest"
        elif self.type == OptunaStorageType.MYSQL:
            return f"docker run -p 0.0.0.0:{port}:3306 -e MYSQL_ALLOW_EMPTY_PASSWORD=1 -d mysql:latest"
        else:
            raise ValueError(f"Unknown storage type: {self.type}")

    def get_storage_backend(self):
        if self.storage_backend is None:
            if self.type == OptunaStorageType.REDIS:
                self.storage_backend = optuna.storages.RedisStorage(
                    url=self.get_uri()
                )
            elif self.type == OptunaStorageType.MYSQL:
                self.storage_backend = optuna.storages.RDBStorage(
                    url=self.get_uri()
                )
            else:
                raise ValueError(f"Unknown storage type: {self.type}")
        return self.storage_backend

    @classmethod
    def get(
        cls, scheduler_host: str, storage_type: OptunaStorageType
    ) -> "OptunaStorage":
        if storage_type == OptunaStorageType.REDIS:
            return OptunaStorage(
                type=storage_type,
                host=scheduler_host,
                db="hypaad",
            )
        elif storage_type == OptunaStorageType.MYSQL:
            return OptunaStorage(
                type=storage_type,
                host=scheduler_host,
                db="hypaad",
                user="root",
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
