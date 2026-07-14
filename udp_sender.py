"""
UDP Sender — streams rows of a CSV as JSON packets over UDP.
 
Simulates a live sensor feed for Tab 4 (Real-Time Testing) without a vehicle.
Run this in one terminal, then run the Streamlit app in another and select
"Live UDP Stream" in Tab 4.
 
Examples
--------
Stream every column at 10 Hz (default):
    python udp_sender.py --csv RawDataPoints.csv
 
Stream only chosen columns:
    python udp_sender.py --csv RawDataPoints.csv --cols V,str,ax,ay,HeartRate,bra
 
Faster / slower and loop forever:
    python udp_sender.py --csv RawDataPoints.csv --rate 20 --loop
"""
 
import argparse
import json
import socket
import time
 
import pandas as pd
 
 
def main():
    parser = argparse.ArgumentParser(description="Stream CSV rows as JSON over UDP.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file to stream.")
    parser.add_argument("--host", default="127.0.0.1", help="Destination host (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=5005, help="Destination UDP port (default 5005).")
    parser.add_argument("--rate", type=float, default=10.0, help="Rows per second (default 10).")
    parser.add_argument("--cols", default="", help="Comma-separated columns to send. Blank = all columns.")
    parser.add_argument("--loop", action="store_true", help="Loop back to the start when the file ends.")
    args = parser.parse_args()
 
    df = pd.read_csv(args.csv)
 
    # Column selection ("let me pick columns")
    if args.cols.strip():
        chosen = [c.strip() for c in args.cols.split(",") if c.strip()]
        missing = [c for c in chosen if c not in df.columns]
        if missing:
            raise SystemExit(f"These columns are not in the CSV: {missing}\nAvailable: {list(df.columns)}")
        df = df[chosen]
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.port)
    interval = 1.0 / max(args.rate, 0.001)
 
    print(f"Streaming {len(df)} rows x {df.shape[1]} cols -> {args.host}:{args.port} "
          f"at {args.rate} rows/sec. Ctrl+C to stop.")
    print(f"Columns: {list(df.columns)}")
 
    sent = 0
    try:
        while True:
            for _, row in df.iterrows():
                # Convert row to a plain dict of JSON-safe values
                payload = {}
                for col, val in row.items():
                    if pd.isna(val):
                        payload[col] = None
                    else:
                        payload[col] = float(val) if hasattr(val, "item") or isinstance(val, (int, float)) else val
                sock.sendto(json.dumps(payload).encode("utf-8"), dest)
                sent += 1
                if sent % 100 == 0:
                    print(f"  sent {sent} packets...", end="\r")
                time.sleep(interval)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()
        print(f"\nDone. Total packets sent: {sent}")
 
 
if __name__ == "__main__":
    main()
 