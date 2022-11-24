# /bin/bash

# Install make
apt-get install -y make

# Install Python venv
apt-get install -y python3-venv python3-dev
apt-get install -y build-essential

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh
usermod -aG docker $USER
systemctl restart docker

# MySQL Setup
sudo apt-get install default-libmysqlclient-dev
# On MacOS: brew install mysql
