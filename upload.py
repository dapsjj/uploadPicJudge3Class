import os
import cv2
from tkinter import messagebox
from tkinter import Tk
from tkinter import Button
import time
import training
from tkinter.filedialog import askopenfilename

def get_files():
    filename = askopenfilename(title="上传图片", filetypes=[('图片', 'jpg'), ('图片', 'jpeg'),('图片', 'png')])
    i = 1
    dir = 'D:\\deal_pics\\' + time.strftime('%Y-%m-%d')
    if not os.path.exists(dir):
        os.makedirs(dir)
    if filename:
        img = cv2.imread(filename)
        img = cv2.resize(img, (208, 208), interpolation=cv2.INTER_CUBIC)
        str1 = training.evaluate_one_image(img)
        newFile = filename.split('/')[-1]
        name = newFile.split('.')[0]
        cv2.imwrite(dir+'\\'+name+'.'+'jpg', img)  # 保存
        cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        messagebox.showinfo('提示', str1)
    else:
        messagebox.showinfo('提示', '未选择图片!')
root = Tk()
root.title('上传')
button = Button(root, text="点此上传", command=get_files,width=20,height=10)
button.pack()
root.geometry('300x200+500+300')
root.mainloop()
