# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math import sqrt
class cm:
 
    def __init__(self,original_image,img,height,width,
                 euclid_threshold , # euclid 閥值
                 touch_range , # 關聯閥值
                 paint_threshold , # 向量長度閥值
                 percent # 向量閥值個數
                 ):
        self.original_image = original_image ; self.img = img ; self.height = height ; self.width = width
        self.euclid_threshold = euclid_threshold # euclid 閥值
        self.touch_range = touch_range # 關聯閥值
        self.paint_threshold = paint_threshold # 向量長度閥值
        self.percent = percent # 向量閥值個數
        self.block_vector = []
        self.truca = 16 # 想要濃縮地提取的訊息量
        self.hough_space = (self.height, self.width,2)
        self.hough_space = np.zeros(self.hough_space)
        self.shiftvector = []
 
    def detection_forgery(self):
        ''' 先對照片執行離散餘弦變換，接著 zig zag 掃描後，提取低頻的重要訊息 '''
        count = 0
        for r in range(0, self.height-8, 1):
            for c in range(0, self.width-8, 1):
                block = self.img[r:r+8, c:c+8]
                if(r+c==0):
                    print(block) # cv2.imshow("sonuc",block)
                imf = np.float32(block)
                dct = cv2.dct(imf)
                nko = np.array([
                    [2, 1, 1, 2, 3, 5, 6, 7],
                    [1, 1, 2, 2, 3, 7, 7, 7],
                    [2, 2, 2, 3, 5, 7, 8, 7],
                    [2, 2, 3, 3, 6,10,10, 7],
                    [2, 3, 4, 7, 8,13,12, 9],
                    [3, 4, 7, 8,10,12,14,11],
                    [6, 8, 9,10,12,15,14,12], 
                    [9,11,11,12,13,12,12,12]
                    ])
                ten = np.array([
                    [80, 60, 50, 80, 120, 200, 255, 255],
                    [55, 60, 70, 95, 130, 255, 255,255],
                    [70, 65, 80, 120, 200, 255, 255, 255],
                    [70, 85, 110, 145, 255, 255,255,255],
                    [90, 110, 185,  255, 255,255,255,255],
                    [120, 175, 255, 255,255,255, 255,255],
                    [255, 255,255,255, 255,255,255,255],
                    [255, 255,255,255, 255,255,255,255]])
                kni = np.array([
                    [3, 2, 2, 3, 5, 8, 10, 12],
                    [2, 2, 3, 4, 5, 12, 12, 11],
                    [3, 3, 3, 5 ,8, 11, 14, 11],
                    [3, 3, 4, 6, 10, 17, 16, 12],
                    [4, 4, 7, 11, 14, 22, 21, 15],
                    [5, 7, 11, 13, 16, 12, 23, 18],
                    [10, 13, 16, 17, 21, 24, 24, 21], 
                    [14, 18, 19, 20, 22, 20, 20, 20]
                    ])
                ''' 將離散餘弦變換的元素，畫分為量化矩陣，來進行壓縮 '''
                dct = np.round(np.divide(dct, nko)).astype(int) ; dct = (dct/4).astype(int)
                vector = [] ; n = len(dct) - 1 ; i = 0 ; j = 0
                count = 0
                for _ in range(n * 2):
                    vector.append(dct[i][j])
 
                    if j == n:   # right border
                        i += 1     # shift
                        while i != n:   # diagonal passage
                            vector.append(dct[i][j])
                            i += 1
                            j -= 1
                    elif i == 0:  # top border
                        j += 1
                        while j != 0:
                            vector.append(dct[i][j])
                            i += 1
                            j -= 1
                    elif i == n:   # bottom border
                        j += 1
                        while j != n:
                            vector.append(dct[i][j])
                            i -= 1
                            j += 1
                    elif j == 0:   # left border
                        i += 1
                        while i != 0:
                            vector.append(dct[i][j])
                            i -= 1
                            j += 1
                vector.append(dct[i][j])
                ''' 將每個各自八乘八區塊的初始座標，增加至序列尾端 '''
                del vector[(self.truca):(64)]
                vector.append(c)
                vector.append(r)
                self.block_vector.append(vector)
                '''千萬不要在這裡 print '''
        self.block_vector = np.array(self.block_vector) 
        print (self.block_vector) # 詞典排列
        self.block_vector = self.block_vector[
            np.lexsort(
                np.rot90(self.block_vector)
                [2 : self.truca+999, :]
                )
            ]
        ''' 確立向量的關聯 '''
        for i in range(len(self.block_vector)):
            if(i + self.touch_range >= len(self.block_vector)):
                self.touch_range -= 1
                # vektorleri asagiya dogru belirlenen esik kadar oklid benzerliklerine gore inceliyoruz.
            for j in range( i + 1 , i + self.touch_range + 1 ):
                if(self.euclid(self.block_vector[i], self.block_vector[j],self.truca) <= self.euclid_threshold):
                    # birbirlerine benzeyelen vektorlerin son indislerinde bulunan konumlarini yeni bir vektorde tutuyoruz.
                    v1=[]
                    v2=[]
                    v1.append(int(self.block_vector[i][-2])) #x1
                    v1.append(int(self.block_vector[i][-1])) #y1
                    v2.append(int(self.block_vector[j][-2])) #x2
                    v2.append(int(self.block_vector[j][-1])) #y2
                    # iliskili vektorlerlerin oklid ile uzunluklarini bularak belirlenen esik degerine gore kisa olanlari eliyoruz.
                    if(self.euclid(v1, v2 , 2) >= self.paint_threshold):
                        # son olarak belirlenen vektorlerin dogrultulari hesaplanarak , bu dogrultunun hough uzayindaki konumu bir arttırıliyor
                        # hangi blockvektorlerin hangi dogrultuyu arttirdigini kaybetmemek icin yeni bir vektor olusturularak once dogrultu sonra
                        # kendi koordinatlari vektorde tutuluyor ve bu vektorlerde block block shift vektore atiliyor.
                        c = abs(v2[0]-v1[0])
                        r = abs(v2[1]-v1[1])
                        if (v2[0]>=v1[0]):
                            if(v2[1]>=v1[1]):
                                z = 0
                            else:
                                z = 1
                        if (v1[0] > v2[0]):
                            if (v1[1] >= v2[1]):
                                z = 0
                            else:
                                z = 1
                        self.hough_space[r][c][z] += 1
                        vector=[]
                        vector.append(c)
                        vector.append(r)
                        vector.append(z)
                        vector.append(v1[0])
                        vector.append(v1[1])
                        vector.append(v2[0])
                        vector.append(v2[1])
                        self.shiftvector.append(vector) # 霍夫座標 向量座標
        ''' 根據同方向的移位向量個數，來確立照片中被移動複製竄偽的區域 '''
        max=-1
        for i in range(self.height):
            for j in range(self.width):
                for h in range(2):
                    if(self.hough_space[i][j][h]) > max:
                        max = self.hough_space[i][j][h]
        for i in range(self.height):
            for j in range(self.width):
                for h in range(2):
                    if (self.hough_space[i][j][h]) >= (max - (max*self.percent/100)):
                        for k in range(len(self.shiftvector)):
                            if (self.shiftvector[k][0]==j and self.shiftvector[k][1]==i and self.shiftvector[k][2]==h):
                                cv2.rectangle(
                                    self.original_image ,
                                    (self.shiftvector[k][3] , self.shiftvector[k][4]),
                                    (self.shiftvector[k][3]+8, self.shiftvector[k][4]+8),
                                    (255,255,255), -1)
                                cv2.rectangle(
                                    self.original_image ,
                                    (self.shiftvector[k][5] , self.shiftvector[k][6]) , 
                                    (self.shiftvector[k][5]+8 , self.shiftvector[k][6]+8),
                                    (255,255,255), -1)
        return np.uint8(self.original_image) # cv2.imshow("sonuc",self.img)
    def euclid(self,vector1,vector2,size):
        sum=0
        for i in range(size):
            sum += (vector2[i]-vector1[i])**2
        return sqrt(sum)
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
def oas(filename):
    #plt.cla()
    global b
    b = tk.Label(root)
    ori = cv2.cvtColor(cv2.resize(cv_imread(filename),(500,250)),cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(cv_imread(filename),cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(cv2.resize(cv_imread(filename),(500,250)),cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(cv_imread(filename),cv2.COLOR_BGR2GRAY)
    gtw = cv2.cvtColor(cv_imread(filename),cv2.COLOR_BGR2GRAY)
    print(gtw.shape[0])
    '''
    y = np.zeros((256))
    for i in range(0,gray.shape[0]):
        for j in range(0,gray.shape[1]):
            y[gray[i,j]] += 1
    '''
    #output = btcoding(g2,250,500,4)
    asd = cm(original,gtw,gtw.shape[0],gtw.shape[1],
             3.5 , # euclid 閥值
             8 , # 關聯閥值 
             100 , # 向量長度閥值
             5 # 向量閥值個數
             )
    a = asd.detection_forgery()
    btc = Image.fromarray(a)
    imgtk = ImageTk.PhotoImage(image = btc)
    b = tk.Label(image=imgtk)
    b.pack()
    b.imgtk = imgtk
    b.configure(image = imgtk)
    cvphoto = Image.fromarray(ori)
    imgtk = ImageTk.PhotoImage(image = cvphoto)
    media.imgtk = imgtk
    media.configure(image = imgtk)
    #plt.bar(np.arange(0,256),y,color="gray",align="center")
    #canva.draw()
def opfile():
    b.destroy()
    sfname = filedialog.askopenfilename(title='選擇',filetypes=[('All Files','*'),("jpeg files","*.jpg"),("png files","*.png"),("gif files","*.gif")])
    return sfname
def oand():
    filename = opfile()
    oas(filename)
def main():
    global root
    root = tk.Tk()
    global b
    b = tk.Label(root)
    mediaFrame = tk.Frame(root).pack()
    global media
    media = tk.Label(mediaFrame)
    media.pack()
    #fig = plt.figure()
    #plot =fig.add_subplot(111)
    #global canva
    #canva = FigureCanvasTkAgg(fig,root)
    #canva.get_tk_widget().pack(side='right')
    b1 = tk.Button(root, text="打開",command = oand).pack()
    root.mainloop()
if __name__=='__main__':
    main()