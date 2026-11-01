import asyncio
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from src.storage.exceptions import StorageError

_store = None


def initialize(settings):
    global _store
    from src.storage.service import Store

    _store = Store(settings)


def execute(operation, name, payload):
    try:
        if operation == "close":
            _store.close()
            return ("ok", {})
        return ("ok", _store.execute(operation, name, payload))
    except StorageError as exc:
        return ("error", (exc.status, exc.detail))
    except (ValueError, TypeError) as exc:
        return ("error", (422, str(exc)))


class Worker:
    def __init__(self, settings):
        self.settings = settings
        self.ready = False
        self.busy = False
        self.pending = None

    async def start(self):
        self.pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=initialize,
            initargs=(self.settings,),
        )
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(self.pool, execute, "list", "", {}),
                self.settings.startup_timeout,
            )
            self.ready = True
        except BaseException:
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise

    async def close(self):
        self.ready = False
        if self.pending:
            await asyncio.gather(self.pending, return_exceptions=True)
        try:
            await asyncio.get_running_loop().run_in_executor(self.pool, execute, "close", "", {})
        finally:
            await asyncio.to_thread(self.pool.shutdown, wait=True)

    async def _run(self, operation, name, payload):
        try:
            status, value = await asyncio.get_running_loop().run_in_executor(
                self.pool, execute, operation, name, payload
            )
            if status == "error":
                raise StorageError(*value)
            if len(json.dumps(value, allow_nan=False).encode()) > self.settings.max_bytes:
                raise StorageError(413, "Response exceeds byte limit")
            return value
        except StorageError:
            raise
        except Exception as exc:
            self.ready = False
            raise StorageError(503, "Database worker failed; restart service") from exc
        finally:
            self.busy = False

    async def call(self, operation, name="", payload=None):
        if not self.ready or self.busy:
            raise StorageError(503, "Database worker unavailable or busy")
        self.busy = True
        self.pending = asyncio.create_task(self._run(operation, name, payload or {}))
        # Retrieve exceptions even when the HTTP caller leaves before completion.
        self.pending.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )
        try:
            return await asyncio.wait_for(asyncio.shield(self.pending), self.settings.timeout)
        except TimeoutError as exc:
            raise StorageError(504, "Operation still running; check state before retrying") from exc
