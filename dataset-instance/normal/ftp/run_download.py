from myftp import myFTP
import os
import random

filenames = os.listdir('/home/ubuntu/figure')
num_iteration = 10000
host = '33.33.33.220'
user = 'ubuntu'
passwd = 'ubuntu'

for i in range(num_iteration):
    ftp_client = myFTP(host)
    ftp_client.login(user, passwd)
    filename = random.choice(filenames)
    # filename = 'xfce.png'
    print(f'[Downloading {i+1}/{num_iteration}]: {filename}')
    local_path = f'/tmp/{filename}'
    remote_path = f'/home/ubuntu/figure/{filename}'
    ftp_client.download_file(local_path, remote_path)
    ftp_client.close()