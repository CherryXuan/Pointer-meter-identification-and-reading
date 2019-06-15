'''
Author:CherryXuan
Email:shenzexuan1994@foxmail.com
Wechat:cherry19940614

File:Identify_picture.py
Name:
Version:v0.0.1
Date:2019/6/14 13:29
'''
from MeterReader import METER

if __name__ == '__main__':
    input_path = input('请输入图片地址：')
    METER(input_path).iden_pic()


