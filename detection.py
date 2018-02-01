import cv2
import numpy as np
import matplotlib.pyplot as plt

class numberdetect():
    firstnum_x_left=0
    firstnum_x_right=0
    firstnum_y_up=0
    firstnum_y_down=0
    secondnum_x_left=0
    secondnum_x_right=0
    secondnum_y_up=0
    secondnum_y_down=0
    img=0

    def __init__(self, filename):
        self.img = cv2.imread(filename)
        chk = 0
        foundfirstnum = 0
        for i in range(52):
            numexist = 0
            for j in range(26):
                if np.array_equal(self.img[j][i], [192, 192, 192])==False:
                    self.img[j][i]=[255, 255, 255]
                    numexist = 1
                    if chk == 0 and foundfirstnum == 0:
                        chk=1
                        self.firstnum_x_left=i
                        self.firstnum_y_up=j
                        self.firstnum_y_down=j
                    elif chk == 1 and foundfirstnum == 0:
                        self.firstnum_y_up=min(self.firstnum_y_up, j)
                        self.firstnum_y_down=max(self.firstnum_y_down, j)
                    elif chk == 0 and foundfirstnum == 1:
                        chk = 1
                        self.secondnum_x_left=i
                        self.secondnum_y_up=j
                        self.secondnum_y_down=j
                    elif chk == 1 and foundfirstnum == 1:
                        self.secondnum_y_up=min(self.secondnum_y_up, j)
                        self.secondnum_y_down=max(self.secondnum_y_down, j)
                else:
                    self.img[j][i]=[0, 0, 0]
            if numexist == 0:
                if chk == 1 and foundfirstnum == 0:
                    self.firstnum_x_right = i
                    foundfirstnum = 1
                    chk = 0
                elif chk == 1 and foundfirstnum == 1:
                    self.secondnum_x_right = i
                    chk = 0
    def firstnum(self):
        return self.firstnum_x_left, self.firstnum_y_up, self.firstnum_x_right, self.firstnum_y_down
    def secondnum(self):
        return self.secondnum_x_left, self.secondnum_y_up, self.secondnum_x_right, self.secondnum_y_down
    def firstnum_img(self):
        x0, y0, x1, y1 = self.firstnum()
        img_trans = self.img[y0-1:y1+1, x0-1:x1+1]
        resize = cv2.resize(img_trans, dsize=(20, 20), interpolation = cv2.INTER_LINEAR)
        gray_image = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        size = np.zeros((28, 28))
        size[4:24, 4:24] = gray_image
        return size
    def secondnum_img(self):
        x0, y0, x1, y1 = self.secondnum()
        img_trans = self.img[y0-1:y1+1, x0-1:x1+1]
        resize = cv2.resize(img_trans, dsize=(20, 20), interpolation = cv2.INTER_LINEAR)
        gray_image = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        size = np.zeros((28, 28))
        size[4:24, 4:24] = gray_image
        return size
