import socket
import http.server
import sys

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external IP to detect the local interface's IP
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def run_server(port=8000):
    ip = get_ip()
    print(f"\n🚀 Serving GLITCH at http://{ip}:{port}/")
    print(f"Check builds: http://{ip}:{port}/compiled/\n")
    print(f"📋 Copy-paste to your Android device (adb shell/Termux):")
    print(f"curl http://{ip}:{port}/compiled/android_arm64/glitch -o glitch && chmod +x glitch && ./glitch\n")
    
    server_address = ('', port)
    httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
