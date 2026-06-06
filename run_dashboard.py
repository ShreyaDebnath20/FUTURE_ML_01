import http.server
import socketserver
import webbrowser
import threading
import sys
import time
import socket

PORT = 8000

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress default HTTP request printouts, keeping console clean
        pass

def start_server(port):
    handler = QuietHandler
    # Restrict to loopback interface for local access security
    try:
        with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
            print(f"\n============================================================")
            print(f"   DASHBOARD SERVER ACTIVE")
            print(f"============================================================")
            print(f"   Local URL : http://localhost:{port}/dashboard.html")
            print(f"   Status    : Serving forecast datasets successfully.")
            print(f"   Shutdown  : Press Ctrl+C in this window to stop serving.")
            print(f"============================================================\n")
            httpd.serve_forever()
    except Exception as e:
        print(f"[Error] Failed to start HTTP server: {e}")

def main():
    port = PORT
    # Try finding an open port starting from 8000
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            s.close()
            break
        except OSError:
            # Port is busy, increment and try next
            port += 1
            if port > 8099:
                print("[Error] No open ports found in range 8000-8099. Aborting.")
                sys.exit(1)

    # Run the server in a daemon thread so it exits when main exits
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    # Sleep briefly to ensure socket is bound
    time.sleep(0.8)

    # Launch default web browser
    dashboard_url = f"http://localhost:{port}/dashboard.html"
    print(f"[Local System] Launching web browser for: {dashboard_url}")
    webbrowser.open(dashboard_url)

    # Wait for Ctrl+C to terminate
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Local System] Received stop signal. Shutting down server. Goodbye!")

if __name__ == "__main__":
    main()
