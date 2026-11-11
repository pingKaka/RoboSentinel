"""Keep desktop events independent of blocking hardware and network operations."""
import multiprocessing as mp
from queue import Empty, Full
from types import SimpleNamespace

import numpy as np


CLIENT_FIELDS = (
    "fault_result", "latest_visual", "status", "_manual_busy", "error",
    "initial_result", "latest_result", "task_id", "description", "server",
    "responses", "dropped", "latest_robot_state", "output_dir",
)


def _window_main(updates, stop, quit_requested, font_path):
    import cv2
    from .dashboard import DetectionDashboard

    try:
        dashboard = DetectionDashboard(font_path)
        cv2.namedWindow("RoboSentinel", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("RoboSentinel", 1440, 846)
        dirty = True

        def mouse(*args):
            nonlocal dirty
            dashboard.mouse(*args)
            dirty = True

        cv2.setMouseCallback("RoboSentinel", mouse)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        payload = dict(frames={"external": blank, "wrist": blank},
                       notice="正在连接本地硬件")
        while not stop.is_set():
            try:
                payload = updates.get_nowait()
                dirty = True
            except Empty:
                pass
            if dirty:
                cv2.imshow("RoboSentinel", dashboard.render(**payload))
                dirty = False
            if cv2.waitKey(15) & 0xFF == ord("q"):
                quit_requested.set()
                break
            if cv2.getWindowProperty("RoboSentinel", cv2.WND_PROP_VISIBLE) < 1:
                quit_requested.set()
                break
    finally:
        quit_requested.set()
        cv2.destroyAllWindows()


class DashboardProcess:
    def __init__(self, font_path=None):
        # Spawn avoids inheriting CUDA, Qt and hardware connections from the parent.
        ctx = mp.get_context("spawn")
        self._updates = ctx.Queue(maxsize=1)
        self._stop = ctx.Event()
        self.quit_requested = ctx.Event()
        self._process = ctx.Process(target=_window_main,
                                    args=(self._updates, self._stop, self.quit_requested, font_path),
                                    name="robosentinel-dashboard", daemon=True)
        self._process.start()

    def render(self, frames, client=None, **kwargs):
        snapshot = None if client is None else SimpleNamespace(
            **{name: getattr(client, name, None) for name in CLIENT_FIELDS})
        # Never wait for the UI from the robot loop. A full queue drops this preview.
        if self._updates.full():
            return
        payload = dict(frames={name: image.copy() for name, image in frames.items()},
                       client=snapshot, **kwargs)
        try:
            self._updates.put_nowait(payload)
        except Full:
            pass

    def close(self):
        self._stop.set()
        self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3)
        self._updates.cancel_join_thread()
        self._updates.close()
