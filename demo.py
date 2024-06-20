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

# sample config
# 1. --transcribe ./voice/_7_7_7138666916734895261_1_73.wav
# 2. --train

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py [--transcribe filepath|--train]")
        sys.exit(1)  # Exit with an error message

    # The first argument is the script name, so we look at sys.argv[1] for the actual argument
    command = sys.argv[1]

    # Check the command and print accordingly
    if command == "--transcribe" and len(sys.argv) == 3:
        # Check if there is a filepath provided after the --decode flag
        filepath = sys.argv[2]
        pprint.pp("transcribing audio file: " + str(filepath))
        threading.Thread(target=running, args=(command, sys.argv[2])).start()
    elif command == "--train" and len(sys.argv) == 2:
        pprint.pp("fine-tuning whisper, and this would take long time ...")
        threading.Thread(target=running, args=(command, )).start()
    else:
        print("Usage: python script.py [--transcribe filepath|--train]")
        sys.exit(1)  # Exit with an error message

    progress()


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


def running(action_cmd, *args):
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
        source_path = os.path.dirname(os.path.realpath(__file__))
        target_path = r'/root/demo'

        trans = paramiko.Transport((hostname, port))
        # 建立连接
        trans.connect(username=username, password=password)
        ssh = paramiko.SSHClient()
        ssh._transport = trans
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 执行命令，和传统方法一样 sftp
        sftp = MySFTPClient.from_transport(trans)
        sftp.mkdir(target_path, ignore_existing=True)
        sftp.mkdir(target_path + r'/voice', ignore_existing=True)
        sftp.put_dir(source_path + r'\voice', target_path + r'/voice')
        sftp.put(source_path + r'\infer.py', target_path + r'/infer.py')
        sftp.put(source_path + r'\whisper-finetune.py', target_path + r'/whisper-finetune.py')
        sftp.put(source_path + r'\start.sh', target_path + r'/start.sh')
        sftp.put(source_path + r'\set_env.sh', target_path + r'/set_env.sh')
        sftp.put(source_path + r'\decode.sh', target_path + r'/decode.sh')
        sftp.put(source_path + r'\finetune.sh', target_path + r'/finetune.sh')

        # 执行命令，和传统方法一样 ssh
        stdin, stdout, stderr = ssh.exec_command(r'cd ~/demo/ && chmod +x ./decode.sh')
        stdin, stdout, stderr = ssh.exec_command(r'cd ~/demo/ && chmod +x ./finetune.sh')
        exec_cmd = None
        if action_cmd == "--transcribe":
            audio_file = str(args[0])
            exec_cmd = r'./decode.sh ' + audio_file + r'" '
        elif action_cmd == "--train":
            exec_cmd = r'./finetune.sh'+ r'" '

        stdin, stdout, stderr = ssh.exec_command(
            r'cd ~/demo/ && docker exec -u root -t whisper bash -c "cd /root/demo && source ./set_env.sh &&' + exec_cmd
            , get_pty=True)

        # Read the output as it comes in
        while True:
            output = stdout.readline()
            if output == '':
                break
            print(output.strip())

        # 关闭连接
        trans.close()
        ssh.close()
        interrupt_main()
    except Exception:
        print('Exception!!')
        raise


if __name__ == "__main__":
    print('argument list: ', sys.argv)
    main()