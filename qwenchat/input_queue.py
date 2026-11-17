"""Background input reader with a thread-safe message queue."""

from __future__ import annotations

from gc import callbacks
import sys
import threading
import queue
from typing import Callable, Optional

class InputQueue:
    """
    Continuously reads from stdin in a background thread and
    stores lines in a thread-safe queue.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_enqueue: Optional[Callable[[str], None]] = None # callback for new messages

    def set_on_enqueue(self, callback: Callable[[str], None]) -> None:
        self._on_enqueue = callback

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="InputQueueReader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break

            if line == "":
                # EOF
                break

            text = line.rstrip("\n")
            if text:
                self._queue.put(text)
                if self._on_enqueue:
                    try:
                        self._on_enqueue(text)
                    except Exception:
                        pass

    def get(self, block: bool = True, timeout: Optional[float] = None) -> str:
        """Get the next message (blocking by default)."""
        return self._queue.get(block=block, timeout=timeout)

    def get_nowait(self) -> Optional[str]:
        """Get the next message or None if empty."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def put(self, message: str) -> None:
        """Manually enqueue a message (useful for testing)."""
        self._queue.put(message)

    def clear(self) -> int:
        """Drop all pending messages. Returns how many were dropped."""
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    def pending(self) -> int:
        """Number of queued messages."""
        return self._queue.qsize()

    def snapshot(self) -> list[str]:
        """
        Return a copy of pending messages without consuming them.
        Note: this is a best-effort snapshot.
        """
        with self._queue.mutex:
            return list(self._queue.queue)