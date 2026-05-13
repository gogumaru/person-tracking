import socket

target_ip = "10.167.170.4" # Confirm this is the camera IP!
ports = [554, 8554, 8000, 8080, 8888, 1025, 2020, 5554, 10554]

print(f"Scanning {target_ip}...")
for port in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    if s.connect_ex((target_ip, port)) == 0:
        print(f"PORT {port} IS OPEN")
    s.close()
