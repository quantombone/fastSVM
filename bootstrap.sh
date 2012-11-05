#!/bin/bash
set -e
bucket=fastsvm
fastSVM_path=fastSVM
svm_path=packaged-pruned.gz

mkdir -p /home/hadoop/contents
cd /home/hadoop/contents

sudo apt-get update
sudo apt-get install -y pkg-config python-pip python-imaging libfreeimage-dev libboost-iostreams-dev
sudo pip install boto
wget -S -T 10 -t 5 http://s3.amazonaws.com/$bucket/cutil_math.h
wget -S -T 10 -t 5 http://s3.amazonaws.com/$bucket/fastSVM.cu
wget -S -T 10 -t 5 http://s3.amazonaws.com/$bucket/Makefile

wget http://developer.download.nvidia.com/compute/cuda/4_2/rel/drivers/devdriver_4.2_linux_64_295.41.run
sudo sh devdriver_4.2_linux_64_295.41.run -s
wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_rhel4.8.run
sudo sh cudatoolkit_4.0.17_linux_64_rhel4.8.run --noexec --target toolkit
sudo toolkit/install-linux.pl auto
sudo /usr/bin/nvidia-xconfig -a
sudo nvidia-smi -c 3 
sudo nvidia-smi -e 0 

make

wget -S -T 10 -t 5 http://s3.amazonaws.com/$bucket/$svm_path

echo '[Credentials]' > .aws
echo aws_access_key_id = $1 >> .aws
echo aws_access_key_secret = $2 >> .aws
