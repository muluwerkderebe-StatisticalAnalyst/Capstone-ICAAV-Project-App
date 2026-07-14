"""
UDP Receiver — background thread that listens for JSON packets and stores
them in a thread-safe buffer.
 
Designed for Streamlit: create ONE instance via st.cache_resource so a single
listener survives reruns. The background thread NEVER touches st.session_state
(that is not thread-safe); it only writes to its own locked deque.
"""
 
import json
import socket
import threading
from collections import deque
 
 
class UDPReceiver:
    def __init__(self, host="0.0.0.0", port=5005, maxlen=5000):
        self.host = host
        self.port = port
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.packets_received = 0
        self.bind_error = None
 
    def start(self):
        if self.running:
            return
        self.running = True
        self.bind_error = None
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
 
    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            self.bind_error = str(e)
            self.running = False
            sock.close()
            return
        sock.settimeout(0.5)
        while self.running:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                row = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            with self.lock:
                self.buffer.append(row)
                self.packets_received += 1
        sock.close()
 
    def drain(self):
        """Return all buffered rows and clear the buffer."""
        with self.lock:
            items = list(self.buffer)
            self.buffer.clear()
            return items
 
    def peek_latest(self):
        with self.lock:
            return self.buffer[-1] if self.buffer else None
 
    def count(self):
        with self.lock:
            return self.packets_received
 
    def stop(self):
        self.running = False
 