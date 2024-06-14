#!/usr/bin/env python
from _thread import interrupt_main

import paramiko
import pprint
import os, sys
import threading
from time import sleep

username = "root"
password = "Cube@6789"
hostname = "172.16.108.226"
port = 22

print('argument list: ', sys.argv)
audio_file = sys.argv[1]


class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are
            created under target.
        '''
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise


def progress():
    try:
        while True:
            print(".", end="")
            sleep(1)
    except KeyboardInterrupt:
        print("\n job done, program exist.")
        sys.exit()


def running():
    try:
        # client = paramiko.SSHClient()
        # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # client.connect(hostname, port, username, password)
        # t = client.get_transport()
        # sftp=paramiko.SFTPClient.from_transport(t)
        # d = sftp.put(r'C:\Users\lqsys3\Documents\whisper\infer.py', r'/root/whisper/whisper-large-v3/voice/infer.py')
        # pprint.pp(d)
        # client.exec_command(r'cd /root/whisper/whisper-large-v3')
        # stdin, stdout, stderr = client.exec_command(r'cd /root/whisper/whisper-large-v3 && ls -al ./voice')
        # result = stdout.readlines()
        # pprint.pp(result)
        # #
        # 实例化一个transport对象
        pprint.pp("generation text from this audio file: " + audio_file)
        source_path = r'C:\Users\user1\Documents\whisper-demo'
        target_path = r'/root/demo'

        trans = paramiko.Transport((hostname, port))
        # 建立连接
        trans.connect(username=username, password=password)
        ssh = paramiko.SSHClient()
        ssh._transport = trans
        # 执行命令，和传统方法一样
        sftp = MySFTPClient.from_transport(trans)
        sftp.mkdir(target_path, ignore_existing=True)
        sftp.put_dir(source_path + r'\voice', target_path + r'/voice')
        sftp.put(source_path + r'\infer.py', target_path + r'/infer.py')
        sftp.put(source_path + r'\set_env.sh', target_path + r'/set_env.sh')
        sftp.put(source_path + r'\decode.sh', target_path + r'/decode.sh')
        stdin, stdout, stderr = ssh.exec_command(r'cd ~/demo/ && chmod +x ./decode.sh')
        stdin, stdout, stderr = ssh.exec_command(
            r'cd ~/demo/ && docker exec -u root -t whisper bash -c "cd /root/demo && source ./set_env.sh && ./decode.sh ' + audio_file + r'"'
        )
        pprint.pp(stdout.read().decode())

        # 关闭连接
        trans.close()
        interrupt_main()
    except Exception:
        print('Exception!!')
        raise


thread1 = threading.Thread(target=running)
thread1.start()
progress()
