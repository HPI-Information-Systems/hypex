# /bin/bash

# Install make
apt-get install -y make

# Install Python venv
apt-get install -y python3-venv python3-dev
apt-get install -y gcc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh
usermod -aG docker $USER
systemctl restart docker

# MySQL Setup
sudo apt-get install python3-dev default-libmysqlclient-dev build-essential
# On MacOS: brew install mysql
