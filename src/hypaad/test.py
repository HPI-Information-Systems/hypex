import os
import socket

from ssh2.session import Session

host = "ec2-18-196-34-210.eu-central-1.compute.amazonaws.com"
user = os.getlogin()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, 22))

session = Session()
session.handshake(sock)
session.userauth_publickey_fromfile(user, "/home/ubuntu/.ssh/id_rsa")

channel = session.open_session()
channel.execute("echo me; exit 2")
size, data = channel.read()
while size > 0:
    print(data)
    size, data = channel.read()
channel.close()
print("Exit status: %s" % channel.get_exit_status())
