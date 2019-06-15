'''
Author:CherryXuan
Email:shenzexuan1994@foxmail.com
Wechat:cherry19940614

File:MeterReader.py
Name:仪表识别类
Version:v0.0.1
Date:2019/6/14 12:30
'''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class METER(object):
    def __init__(self,path):
        self.path = path

    # 读取图片
    def readData(self):
        imgs_path = []
        for filename in os.listdir(self.path):
            if filename.endswith('.jpg'):
                filename = self.path + '/' + filename
                # 图片归一化处理
                #res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # 按照比例缩放，如x,y轴均缩小一倍
                imgs_path.append(filename)
        return imgs_path

    # 图片归一化处理
    def normalized_picture(self):
        img = cv2.imread(self.path)
        nor = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # 按照比例缩放，如x,y轴均缩小一倍
        cv2.imshow('Normalized picture',nor)
        return nor

    # 颜色空间转换：灰度化
    def color_conversion(self,img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        cv2.imshow('Graying pictures', img_gray)
        return img_gray

    # 中值滤波去噪
    def median_filter(self,img):
        median = cv2.medianBlur(img, 1)  # 中值滤波
        cv2.imshow('Median filter', median)
        return median

    # 双边滤波去噪
    def bilateral_filter(self,img):
        bilateral = cv2.bilateralFilter(img, 9, 50, 50)
        cv2.imshow('Bilateral filter', bilateral)
        return bilateral

    # 高斯滤波去噪
    def gaussian_filter(self,img):
        gaussian = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imshow('Gaussian filter', gaussian)
        return gaussian

    # 图像二值化
    def binary_image(self,img):
        # 应用5种不同的阈值方法
        # ret, th1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        # ret, th2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        # ret, th3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
        # ret, th4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
        # ret, th5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
        # titles = ['Gray', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        # images = [img_gray, th1, th2, th3, th4, th5]
        # 使用Matplotlib显示
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     plt.imshow(images[i], 'gray')
        #     plt.title(titles[i], fontsize=8)
        #     plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        # plt.show()

        # Otsu阈值
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        cv2.imshow('Binary image', th)
        return th

    # 边缘检测
    def candy_image(self,img):
        edges = cv2.Canny(img, 60, 143, apertureSize=3)
        cv2.imshow('canny', edges)
        return edges

    # 开运算：先腐蚀后膨胀
    def open_operation(self,img):
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # 椭圆结构
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))     # 十字形结构
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
        cv2.imshow('Open operation', opening)
        return opening

    # 霍夫圆变换：检测表盘
    def detect_circles(self,gray,img):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param2=150, minRadius=160)
        circles = np.uint16(np.around(circles))  # 把circles包含的圆心和半径的值变成整数
        cir = img.copy()

        for i in circles[0, :]:
            cv2.circle(cir, (i[0], i[1]), i[2], (0, 255, 0), 2, cv2.LINE_AA)  # 画圆
            cv2.circle(cir, (i[0], i[1]), 2, (0, 255, 0), 2, cv2.LINE_AA)  # 画圆心
        cv2.imshow("circles", cir)
        return cir

    # 霍夫直线变换：检测指针
    def detect_pointer(self,cir):

        img = cv2.GaussianBlur(cir, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
        result = cir.copy()

        for line in lines[0]:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            rtheta = theta * (180 / np.pi)
            print('θ1:', rtheta)
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                a = int(
                    int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                b = int(result.shape[0] / 2)
                pt3 = (a, b)
                pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))
                # 绘制一条白线
                cv2.putText(result, 'theta1={}'.format(int(rtheta)), pt4, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.line(result, pt3, pt4, (0, 0, 255), 2, cv2.LINE_AA)
            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                a = int(
                    int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                b = int(result.shape[0] / 2)
                pt3 = (a, b)
                pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))
                # 绘制一条直线
                cv2.line(result, pt3, pt4, (0, 0, 255), 2, cv2.LINE_AA)

        for line in lines[2]:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            rtheta = theta * (180 / np.pi)
            print('θ2:', - rtheta - 90)
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                a = int(
                    int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                b = int(result.shape[0] / 2)
                pt3 = (a, b)
                pt4 = (int(int(int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)) + a) / 2),
                       int(int(int(b + result.shape[0]) / 2)))
                # 绘制一条白线
                cv2.putText(result, 'theta2={}'.format(int(- rtheta - 90)), pt4, cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255), 1)
                cv2.line(result, pt3, pt4, (255, 0, 0), 2, cv2.LINE_AA)
            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                a = int(
                    int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                b = int(result.shape[0] / 2)
                pt3 = (a, b)
                pt4 = (int(int(int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)) + a) / 2),
                       int(int(int(b + result.shape[0]) / 2)))
                # 绘制一条直线
                cv2.line(result, pt3, pt4, (255, 0, 0), 2, cv2.LINE_AA)

        # cv2.imshow('Canny', edges)
        cv2.imshow('Result', result)

        return result
    def iden_pic(self):

        image = METER(self.path)
        nor = image.normalized_picture()
        gray = image.color_conversion(nor)
        binary = image.binary_image(gray)
        median = image.median_filter(binary)
        #bilateral = image.bilateral_filter(median)
        gaussian = image.gaussian_filter(median)
        candy = image.candy_image(median)
        cir = image.detect_circles(gray,nor)
        pointer = image.detect_pointer(cir)

        nor = cv2.cvtColor(nor, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gaussian = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
        median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2RGB)
        candy = cv2.cvtColor(candy, cv2.COLOR_BGR2RGB)
        cir = cv2.cvtColor(cir, cv2.COLOR_BGR2RGB)
        pointer = cv2.cvtColor(pointer, cv2.COLOR_BGR2RGB)

        titles = ['Original', 'Gray', 'Gaussian', 'Mdian', 'Binary', 'Candy', 'Circle', 'Pointer']
        images = [nor, gray, gaussian, median, binary, candy, cir, pointer]
        # 使用Matplotlib显示
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i])
            plt.title(titles[i], fontsize=8)
            plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        plt.show()

        cv2.waitKey(0)