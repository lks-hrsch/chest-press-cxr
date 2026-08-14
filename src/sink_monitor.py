import os
import time
import threading


class SinkMonitor:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = 0

    def _monitor_progress(self):
        start_t = time.time()
        while not self.stop_event.is_set():
            if os.path.exists(self.save_path):
                size_kb = os.path.getsize(self.save_path) / 1024
                elapsed = time.time() - start_t
                print(f"  [Writing] Size: {size_kb:.2f} KB | Time: {elapsed:.1f}s", end="\r")
            time.sleep(0.5)

    def __enter__(self):
        print(f"Start saving to {self.save_path}...")
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_progress)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        print()  # Newline

        end_time = time.time()
        print(f"Finished saving to {self.save_path}")

        # Statistics
        duration = end_time - self.start_time
        if os.path.exists(self.save_path):
            file_size_kb = os.path.getsize(self.save_path) / 1024  # KB

            print(f"Sink Statistics:")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  File Size: {file_size_kb:.2f} KB")
