from tkinter import *
from tkinter import filedialog
from tkinter import Spinbox
from tkinter import messagebox
import cv2
import os
import random
import numpy as np


images = []
img_names = []
global file
global path
state = []
counter = 0


def img_flip(img, name):
    spin = sp_flip.get()
    spin = int(spin)
    std_list = [-1, 0, 1]
    flip_list = std_list
    for i in range(spin):
        r = random.choice(flip_list)
        f = cv2.flip(img, r)
        flip_list.remove(r)
        save = path + '/' + str(name) + '_flipped_' + str(i) + '.jpg'
        cv2.imwrite(save, f)


def img_gray(img, name):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save = path + '/' + str(name) + '_gray.jpg'
    cv2.imwrite(save, g)


def img_rotate(img, name):
    spin = sp_rotate.get()
    spin = int(spin)
    rot_list = range(181)
    sign_list = [-1, 1]
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    scale = 1
    for i in range(spin):
        angle = int(random.choice(rot_list) * random.choice(sign_list))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        r = cv2.warpAffine(img, rotation_matrix, (w, h))
        save = path + '/' + str(name) + '_rotated_' + str(i) + '.jpg'
        cv2.imwrite(save, r)


def img_translation(img, name):
    h, w = img.shape[:2]
    w_list = range(w // 3)
    h_list = range(h // 3)
    spin = sp_trans.get()
    spin = int(spin)
    for i in range(spin):
        t_w = random.choice(w_list)
        t_h = random.choice(h_list)
        trans_mat = np.float32([[1, 0, t_w], [0, 1, t_h]])
        t = cv2.warpAffine(img, trans_mat, (w, h))
        save = path + '/' + str(name) + '_translated_' + str(i) + '.jpg'
        cv2.imwrite(save, t)


def img_crop(img, name):
    h, w = img.shape[:2]
    spin = sp_crop.get()
    spin = int(spin)
    for i in range(spin):
        w_list = range(3 * w // 4)
        h_list = range(3 * h // 4)
        c1_w = random.choice(w_list)
        c1_h = random.choice(h_list)
        w_list = range(3 * w // 4, w)
        h_list = range(3 * h // 4, h)
        c2_w = random.choice(w_list)
        c2_h = random.choice(h_list)
        c = img[c1_h: c2_h, c1_w: c2_w]
        save = path + '/' + str(name) + '_cropped_' + str(i) + '.jpg'
        cv2.imwrite(save, c)


def img_scale(img, name):
    spin = sp_scale.get()
    spin = int(spin)
    for i in range(spin):
        s_list = range(2, 20)
        scale = random.choice(s_list) / 10
        h, w = img.shape[:2]
        aspect = w/h
        s_h = int(scale * h)
        s_w = int(aspect * s_h)
        scaled = cv2.resize(img, (s_w, s_h), interpolation=cv2.INTER_AREA)
        save = path + '/' + str(name) + '_scaled_' + str(i) + '.jpg'
        cv2.imwrite(save, scaled)


def img_blur(img, name):
    h, w = img.shape[:2]
    if h > 500:
        b_list = range(3, int(h/100))
    else:
        b_list = [3, 4, 5]
    spin = sp_blur.get()
    spin = int(spin)
    for i in range(spin):
        fac = random.choice(b_list)
        if fac % 2 == 0:
            fac += 1
        kernel = (fac, fac)
        b = cv2.GaussianBlur(img, kernel, 0)
        save = path + '/' + str(name) + '_blurred_' + str(i) + '.jpg'
        cv2.imwrite(save, b)


def sel():
    global file
    file = filedialog.askopenfilename(title='choose images', multiple=True, filetype=(
        ('.png files', '*.png'), ('.jpg files', '*.jpg'), ('all files', '*.*')))
    for i in file:
        images.append(cv2.imread(i))
        scr_name = i.split('/')
        scr_name = scr_name.pop(-1)
        scr_name = scr_name.split('.')
        scr_name.pop(-1)
        img_names.append(scr_name)


def aug():
    global path
    global counter
    if len(images) == 0:
        messagebox.showinfo('error', 'error')
    else:
        directory = "Data set"
        files = file[0].split('/')
        files.pop(-1)
        path = files[0]
        files.pop(0)
        for i in files:
            path = path + '/' + i
        path = os.path.join(path, directory)
        if os.path.isdir(path):
            path = path + '_' + str(counter)
            os.mkdir(path)
            counter += 1
        else:
            os.mkdir(path)
        for k in range(len(images)):
            img = images[k]
            name = img_names[k]
            if flip_st.get() == 1:
                img_flip(img, name)
            if rotate_st.get() == 1:
                img_rotate(img, name)
            if trans_st.get() == 1:
                img_translation(img, name)
            if crop_st.get() == 1:
                img_crop(img, name)
            if scale_st.get() == 1:
                img_scale(img, name)
            if blur_st.get() == 1:
                img_blur(img, name)
            if gray_st.get() == 1:
                img_gray(img, name)
            text = comb.curselection()
            if len(text) > 0:
                for combined in range(int(sp_comb.get())):
                    scr = img
                    for j in text:
                        if j == 0:
                            std_list = [-1, 0, 1]
                            img = cv2.flip(img, random.choice(std_list))
                        if j == 1:
                            rot_list = range(181)
                            sign_list = [-1, 1]
                            (h, w) = img.shape[:2]
                            center = (w // 2, h // 2)
                            scale = 1
                            angle = int(random.choice(rot_list) * random.choice(sign_list))
                            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
                            img = cv2.warpAffine(img, rotation_matrix, (w, h))
                        if j == 2:
                            s_list = range(2, 20)
                            scale = random.choice(s_list) / 10
                            h, w = img.shape[:2]
                            aspect = w / h
                            s_h = int(scale * h)
                            s_w = int(aspect * s_h)
                            img = cv2.resize(img, (s_w, s_h), interpolation=cv2.INTER_AREA)
                        if j == 3:
                            h, w = img.shape[:2]
                            w_list = range(3 * w // 4)
                            h_list = range(3 * h // 4)
                            c1_w = random.choice(w_list)
                            c1_h = random.choice(h_list)
                            w_list = range(3 * w // 4, w)
                            h_list = range(3 * h // 4, h)
                            c2_w = random.choice(w_list)
                            c2_h = random.choice(h_list)
                            img = img[c1_h: c2_h, c1_w: c2_w]
                        if j == 4:
                            h, w = img.shape[:2]
                            w_list = range(w // 3)
                            h_list = range(h // 3)
                            t_w = random.choice(w_list)
                            t_h = random.choice(h_list)
                            trans_mat = np.float32([[1, 0, t_w], [0, 1, t_h]])
                            img = cv2.warpAffine(img, trans_mat, (w, h))
                        if j == 5:
                            h, w = img.shape[:2]
                            if h > 500:
                                b_list = range(3, int(h / 100))
                            else:
                                b_list = [3, 4, 5]
                            fac = random.choice(b_list)
                            if fac % 2 == 0:
                                fac += 1
                            kernel = (fac, fac)
                            img = cv2.GaussianBlur(img, kernel, 0)
                        if j == 6:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    save = path + '/' + str(name) + '_combo_' + str(combined) + '.jpg'
                    cv2.imwrite(save, img)
                    img = scr
        messagebox.showinfo('done', 'The augmented data saved in the same directory you have selected')
        images.clear()


# window
main_win = Tk()
main_win.title('Image Augmentation')
main_win.geometry('1000x700')

# labels
Label(main_win, text='Augmentation Properties:', font="50").place(x=35, y=20)
Label(main_win, text='Please select images folder: ', font="50").place(x=50, y=300)
Label(main_win, text='Please select combined augmentation processes: ', font="5").place(x=450, y=30)

# check box
flip_st = IntVar()
rotate_st = IntVar()
scale_st = IntVar()
crop_st = IntVar()
trans_st = IntVar()
blur_st = IntVar()
gray_st = IntVar()

c_flip = Checkbutton(main_win, text='Flip        ', variable=flip_st, onvalue=1, offvalue=0, height=2, width=10)
c_rotate = Checkbutton(main_win, text='Rotate    ', variable=rotate_st, onvalue=1, offvalue=0, height=2, width=10)
c_scale = Checkbutton(main_win, text='Scale       ', variable=scale_st, onvalue=1, offvalue=0, height=2, width=10)
c_crop = Checkbutton(main_win, text='Crop       ', variable=crop_st, onvalue=1, offvalue=0, height=2, width=10)
c_trans = Checkbutton(main_win, text='Translate', variable=trans_st, onvalue=1, offvalue=0, height=2, width=10)
c_blur = Checkbutton(main_win, text='Blur        ', variable=blur_st, onvalue=1, offvalue=0, height=2, width=10)
c_gray = Checkbutton(main_win, text='Gray        ', variable=gray_st, onvalue=1, offvalue=0, height=2, width=10)


c_flip.place(x=50, y=50)
c_rotate.place(x=50, y=80)
c_scale.place(x=50, y=110)
c_crop.place(x=50, y=140)
c_trans.place(x=50, y=170)
c_blur.place(x=50, y=200)
c_gray.place(x=50, y=230)


# spin box
sp_flip = Spinbox(main_win, from_=1, to=3, width=3)
sp_flip.place(x=170, y=60)
sp_rotate = Spinbox(main_win, from_=1, to=5, width=3)
sp_rotate.place(x=170, y=90)
sp_scale = Spinbox(main_win, from_=1, to=5, width=3)
sp_scale.place(x=170, y=120)
sp_crop = Spinbox(main_win, from_=1, to=5, width=3)
sp_crop.place(x=170, y=150)
sp_trans = Spinbox(main_win, from_=1, to=5, width=3)
sp_trans.place(x=170, y=180)
sp_blur = Spinbox(main_win, from_=1, to=5, width=3)
sp_blur.place(x=170, y=210)
sp_gray = Spinbox(main_win, from_=1, to=1, width=3)
sp_gray.place(x=170, y=240)
sp_comb = Spinbox(main_win, from_=1, to=5, width=3)
sp_comb.place(x=500, y=200)

# list box
comb = Listbox(main_win, selectmode="multiple", height=7, width=12)
comb.place(x=480, y=70)
x = ["  Flipping", "  Rotation", "  Scaling", "  Cropping", "  Translation", "  Blurring", "  Gray"]
for each_item in range(len(x)):
    comb.insert(END, x[each_item])

# buttons
img_btn = Button(main_win, text="Browse..", width=10, height=1, command=sel, bg='cyan')
img_btn.place(x=320, y=300)
aug_btn = Button(main_win, text="augment", width=20, height=2, command=aug, bg='cyan')
aug_btn.place(x=500, y=350)

mainloop()
