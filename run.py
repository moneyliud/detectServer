import os
import sys
import webbrowser
import subprocess
import configparser
from time import sleep

BASE_PATH = os.path.abspath('.')

# 重复点击开始即关闭原来服务重新打开
os.system('TASKKILL /F /IM detectServer.exe')


def run_main():
    sys.path.append("libs")
    config = configparser.ConfigParser()
    config.read('config.ini')
    mysql_items = dict(config.items('Server'))
    port = mysql_items['server_port']
    ip = mysql_items['server_ip']
    url = 'http://' + ip + '/detectClient'
    webbrowser.open_new(url)
    print(BASE_PATH)
    main = BASE_PATH + "\\detectServer.exe runserver " + port + " --noreload"
    print('--------------------------')
    print('系统已运行，可关闭此终端.')
    print('--------------------------')
    # os.system(main)
    proc = subprocess.Popen(main, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.close()  # 既然没有命令行窗口，那就关闭输入
    proc.wait()
    result = proc.stdout.read()  # 读取cmd执行的输出结果（是byte类型，需要decode）
    proc.stdout.close()
    print(result)


run_main()
