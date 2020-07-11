import time
import array as arr
import numpy as np
import ctypes
import datetime
import matplotlib.pyplot as plt
import imutils as imutils
from PIL import Image, ImageTk, ImageDraw
from absl import app, flags, logging
from absl.flags import FLAGS
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from tkinter.ttk import Combobox
from tkinter import ttk
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes
from yolov3_tf2.utils import draw_outputs_clean
from yolov3_tf2.utils import draw_outputs

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from ToolTip import CreateToolTip

flags.DEFINE_string('classes', './data/labels/playerball.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3-custom4.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('logo', './data/initialLogo.jpg',
                    'path to initial logo file')
flags.DEFINE_string('output', 'output.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')
# flags.DEFINE_string('output1', 'output', 'path to output directory to store snapshots')


# global yolo
# global class_names
running = True

mouseX = 0
mouseY = 0

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

width_screen = user32.GetSystemMetrics(0)
height_screen = user32.GetSystemMetrics(1)


class FUTOTAL:

    def __init__(self, master, argv):
        super().__init__()
        self.master = master
        self.argv = argv
        self.pause = False

        self.selectFrame = 0
        numframes = 0
        self.numframes = numframes
        self.frame_paused = 0
        self.isLogo = True
        self.times = []

        # Variaveis zoom
        self.zoom = 100
        self.zona_de_zoom = ""
        self.w_zoom = 0
        self.h_zoom = 0

        # Variaveis das cores
        self.color_line = (255, 125, 255)
        self.color_rectangle = (0, 255, 255)
        self.color_elipse = (0, 150, 250)
        self.color_passed = (0, 255, 255)
        self.color_movement = (255, 0, 0)
        self.color_polly = (0, 255, 0)
        self.color_select = (255, 0, 255)

        # Variaveis dos nomes
        self.array_lists_id_with_names = []
        self.array_lists_names_of_player = []

        # Variaveis para selecionar jogador
        self.selectON = False
        self.selecteds = arr.array('i', [])
        self.initArray(self.selecteds)
        self.objects_positions_id = arr.array('i', [])
        self.objects_positions_x_min = arr.array('i', [])
        self.objects_positions_y_min = arr.array('i', [])
        self.objects_positions_x_max = arr.array('i', [])
        self.objects_positions_y_max = arr.array('i', [])
        self.initArray(self.objects_positions_id)
        self.initArray(self.objects_positions_x_min)
        self.initArray(self.objects_positions_y_min)
        self.initArray(self.objects_positions_x_max)
        self.initArray(self.objects_positions_y_max)

        # Variaveis para adicionar linhas
        self.LineON = False
        self.line_player_1 = []
        self.line_player_2 = []
        self.line_active = []

        # Variaveis screenshot
        self.screenshot = False
        self.screenshot_folder = ""

        # self.line_player1 = arr.array('i', [])
        # self.line_player2 = arr.array('i', [])
        # self.initArray(self.line_player1)
        # self.initArray(self.line_player2)
        self.players_selecionados = 0
        self.line_dropON = False

        # Variaveis para a elipse
        self.elipseON = False
        self.frame_elipse_create = arr.array('i', [])
        self.coordinates_elipse_x_init = arr.array('i', [])
        self.coordinates_elipse_y_init = arr.array('i', [])
        self.coordinates_elipse_x_final = arr.array('i', [])
        self.coordinates_elipse_y_final = arr.array('i', [])
        self.initArray(self.frame_elipse_create)
        self.initArray(self.coordinates_elipse_x_init)
        self.initArray(self.coordinates_elipse_y_init)
        self.initArray(self.coordinates_elipse_x_final)
        self.initArray(self.coordinates_elipse_y_final)
        self.num_of_click_elipse = 0

        # Variaveis para o quadrado
        self.quadradoON = False
        self.frame_rectangle_create = arr.array('i', [])
        self.coordinates_rectangle_x_init = arr.array('i', [])
        self.coordinates_rectangle_y_init = arr.array('i', [])
        self.coordinates_rectangle_x_final = arr.array('i', [])
        self.coordinates_rectangle_y_final = arr.array('i', [])
        self.initArray(self.frame_rectangle_create)
        self.initArray(self.coordinates_rectangle_x_init)
        self.initArray(self.coordinates_rectangle_y_init)
        self.initArray(self.coordinates_rectangle_x_final)
        self.initArray(self.coordinates_rectangle_y_final)
        self.num_of_click_rectangle = 0

        # Variaveis para o seta
        self.seta_passeON = False
        self.setaON = False
        self.frame_arrow_create = arr.array('i', [])
        self.coordinates_arrow_x_init = arr.array('i', [])
        self.coordinates_arrow_y_init = arr.array('i', [])
        self.coordinates_arrow_x_final = arr.array('i', [])
        self.coordinates_arrow_y_final = arr.array('i', [])
        self.arrow_type = arr.array('i', [])
        self.initArray(self.frame_arrow_create)
        self.initArray(self.coordinates_arrow_x_init)
        self.initArray(self.coordinates_arrow_y_init)
        self.initArray(self.coordinates_arrow_x_final)
        self.initArray(self.coordinates_arrow_y_final)
        self.initArray(self.arrow_type)
        self.num_of_click_arrow = 0

        # Variaveis para area entre jogadores
        self.pollyON = False
        self.array_lists_polly_players = []
        self.array_lists_polly_players_ONOFF = []
        self.count_pollys = -1
        self.polly_dropON = False

        # Variaveis para a caixa de texto
        self.textBoxON = False
        self.frame_textBox_create = arr.array('i', [])
        self.coordinates_textBox_x_init = arr.array('i', [])
        self.coordinates_textBox_y_init = arr.array('i', [])
        self.coordinates_textBox_text = []
        self.initArray(self.frame_textBox_create)
        self.initArray(self.coordinates_textBox_x_init)
        self.initArray(self.coordinates_textBox_y_init)
        self.initArray(self.coordinates_textBox_text)

        # Definition of the parameters
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.nms_max_overlap = 1.0

        # initialize deep sort
        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

        physical_devices = tf.config.experimental.list_physical_devices('GPU')  ##verifica e há GPUs compativeis
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.yolo = YoloV3(classes=FLAGS.num_classes)

        self.yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')

        self.class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')

        self.filename = FLAGS.logo

        # Captura o video
        try:
            self.cap = cv2.VideoCapture(int(self.filename))
        except:
            self.cap = cv2.VideoCapture(self.filename)

        self.createMenuTop()
        self.createMenuLeft()
        self.createMenuRight()

        self.master.title("DeepSports Eleven - Sports Analysis Software")
        self.master.bind('<Escape>', lambda e: self.master.quit())

        self.lmain = tk.Label(self.master, width=width_screen - 400, height=height_screen - 200)
        self.lmain.pack()

        # NUMBER OF FRAMES

        property_id = int(cv2.CAP_PROP_FRAME_COUNT)
        self.length = int(cv2.VideoCapture.get(self.cap, property_id))

        self.createMenuBottom()

        self.show_frame()
        self.master.tk.call('wm', 'iconphoto', self.master._w, tk.PhotoImage(file=r'.\data\dp11.gif'))

    def donothing(self):
        filewin = tk.Toplevel(self.master)
        button = tk.Button(filewin, text="Do nothing button")
        button.pack()

    def initArray(self, array):
        if len(array) == 0:
            for x in range(100):
                array.insert(0, 0)
        else:
            for x in range(100):
                array[x] = 0

    def createMenuTop(self):
        self.menuBar = tk.Menu(self.master)

        self.master.config(menu=self.menuBar)
        filemenu = tk.Menu(self.menuBar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_video)
        filemenu.add_command(label="Save", command=self.donothing)
        filemenu.add_command(label="Save as...", command=self.donothing)
        filemenu.add_command(label="Settings", command=self.windowsSettings)
        filemenu.add_command(label="Exit", command=self.master.quit)
        self.menuBar.add_cascade(label="File", menu=filemenu)

        editmenu = tk.Menu(self.menuBar, tearoff=0)
        editmenu.add_command(label="Screen shot", command=self.makeScreenshot)

        editmenu.add_separator()

        editmenu.add_command(label="Draw arrow of movement", command=self.setaONOFF)
        editmenu.add_command(label="Draw arrow of pass", command=self.seta_passeONOFF)
        editmenu.add_command(label="Draw elipse", command=self.elipseONOFF)
        editmenu.add_command(label="Draw rectangle", command=self.quadradoONOFF)
        editmenu.add_command(label="Delete all forms", command=self.clean_arrays)
        editmenu.add_command(label="Delete all lines", command=self.line_drop_all)
        self.menuBar.add_cascade(label="Edit", menu=editmenu)

        viewmenu = tk.Menu(self.menuBar, tearoff=0)
        viewmenu.add_command(label="Zoom out", command=self.zoomout)
        viewmenu.add_command(label="Zoom in", command=self.zoomin)
        self.menuBar.add_cascade(label="View", menu=viewmenu)

        helpmenu = tk.Menu(self.menuBar, tearoff=0)
        helpmenu.add_command(label="Help Index", command=self.donothing)
        helpmenu.add_command(label="About...", command=self.donothing)
        self.menuBar.add_cascade(label="Help", menu=helpmenu)
        self.master.config(menu=self.menuBar)

    def windowsSettings(self):
        app = tk.Tk()
        app.title("Settings")
        app.geometry("500x460")
        color = "#%02x%02x%02x" % (66, 162, 80)

        #app.tk.call('wm', 'iconphoto', app._w, tk.PhotoImage(file=r'.\data\dp11.gif'))
        app.iconbitmap(r'.\data\dp11.ico')
        m1 = tk.PanedWindow(app, orient=tk.VERTICAL)
        m1.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        m2 = tk.PanedWindow(m1, orient=tk.VERTICAL,height=30)
        m1.add(m2)
        m2.pack(fill=tk.BOTH, expand=0, side=tk.TOP)
        tk.Label(m2, text='Duration in frames:',font='Helvetica 14 bold',fg=color).pack(side=tk.LEFT)
        m3 = tk.PanedWindow(m1, orient=tk.VERTICAL)
        m1.add(m3)
        m3.pack(fill=tk.BOTH, expand=1, side=tk.TOP)

        m4 = tk.PanedWindow(m3, orient=tk.VERTICAL)
        m3.add(m4)
        m4.pack(fill=tk.BOTH, expand=1, side=tk.LEFT)

        m6 = tk.PanedWindow(m4, orient=tk.VERTICAL)
        m4.add(m6)
        m6.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m6, text='Arrow of Pass:').pack(side=tk.LEFT)
        var = tk.DoubleVar(value=2)
        w1 = tk.Spinbox(m6, from_=0, to=1000,width=5, textvariable=var)
        w1.pack(side=tk.RIGHT)

        m8 = tk.PanedWindow(m4, orient=tk.VERTICAL)
        m4.add(m8)
        m8.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m8, text='Arrow of Movement:').pack(side=tk.LEFT)
        w = tk.Spinbox(m8, from_=0, to=1000, width=5)
        w.pack(side=tk.RIGHT)

        m9 = tk.PanedWindow(m4, orient=tk.VERTICAL)
        m4.add(m9)
        m9.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m9, text='Text Box:').pack(side=tk.LEFT)
        w = tk.Spinbox(m9, from_=0, to=1000, width=5)
        w.pack(side=tk.RIGHT)

        m5 = tk.PanedWindow(m3, orient=tk.VERTICAL)
        m3.add(m5)
        m5.pack(fill=tk.BOTH, expand=1, side=tk.RIGHT)
        m7 = tk.PanedWindow(m5, orient=tk.VERTICAL)
        m5.add(m7)
        m7.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m7, text='Rectangle:').pack(side=tk.LEFT)
        w = tk.Spinbox(m7, from_=0, to=1000, width=5)
        w.pack(side=tk.RIGHT)

        m10 = tk.PanedWindow(m5, orient=tk.VERTICAL)
        m5.add(m10)
        m10.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m10, text='Elipse:').pack(side=tk.LEFT)
        w = tk.Spinbox(m10, from_=0, to=1000, width=5)
        w.pack(side=tk.RIGHT)
        m11 = tk.PanedWindow(m5, orient=tk.VERTICAL)
        m5.add(m11)
        m11.pack(fill=tk.BOTH, expand=1, side=tk.TOP)

        m12 = tk.PanedWindow(m1, orient=tk.VERTICAL)
        m1.add(m12)
        m12.pack(fill=tk.BOTH, expand=1, side=tk.TOP)

        m13 = tk.PanedWindow(m12, orient=tk.VERTICAL)
        m12.add(m13)
        m13.pack(fill=tk.BOTH, expand=1, side=tk.LEFT)

        m15 = tk.PanedWindow(m13, orient=tk.VERTICAL, height=30)
        m13.add(m15)
        m15.pack(fill=tk.BOTH, expand=0, side=tk.TOP)
        tk.Label(m15, text='Opacity of objects:', font='Helvetica 14 bold',fg=color).pack(side=tk.LEFT)

        m16 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m16)
        m16.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m16, text='Arrow of Pass:').pack(side=tk.LEFT)
        w = tk.Spinbox(m16, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m17 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m17)
        m17.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m17, text='Arrow of Movement:').pack(side=tk.LEFT)
        w = tk.Spinbox(m17, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m18 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m18)
        m18.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m18, text='Retangle:').pack(side=tk.LEFT)
        w = tk.Spinbox(m18, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m23 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m23)
        m23.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m23, text='Text Box:').pack(side=tk.LEFT)
        w = tk.Spinbox(m23, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m19 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m19)
        m19.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m19, text='Area between players:').pack(side=tk.LEFT)
        w = tk.Spinbox(m19, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m20 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m20)
        m20.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m20, text='Line between players:').pack(side=tk.LEFT)
        w = tk.Spinbox(m20, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m21 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m21)
        m21.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m21, text='Select player:').pack(side=tk.LEFT)
        w = tk.Spinbox(m21, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m22 = tk.PanedWindow(m13, orient=tk.VERTICAL)
        m13.add(m22)
        m22.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m22, text='Detect player:').pack(side=tk.LEFT)
        w = tk.Spinbox(m22, from_=0, to=100, width=5)
        w.pack(side=tk.RIGHT)

        m14 = tk.PanedWindow(m12, orient=tk.VERTICAL)
        m12.add(m14)
        m14.pack(fill=tk.BOTH, expand=1, side=tk.RIGHT)

        m20 = tk.PanedWindow(m14, orient=tk.VERTICAL, height=30)
        m14.add(m20)
        m20.pack(fill=tk.BOTH, expand=0, side=tk.TOP)
        tk.Label(m20, text='Text of Player:',font='Helvetica 14 bold',fg=color).pack(side=tk.LEFT)
        m24 = tk.PanedWindow(m14, orient=tk.VERTICAL, height=30)
        m14.add(m24)
        m24.pack(fill=tk.BOTH, expand=1, side=tk.TOP)

        m27 = tk.PanedWindow(m24, orient=tk.VERTICAL)
        m24.add(m27)
        m27.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m27, text='Font Size:').pack(side=tk.LEFT)
        w = tk.Spinbox(m27, from_=0, to=36, width=5)
        w.pack(side=tk.RIGHT)

        m28 = tk.PanedWindow(m24, orient=tk.VERTICAL)
        m24.add(m28)
        m28.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m28, text='Type:').pack(side=tk.LEFT)
        var = tk.StringVar()
        var.set("Helvetica")
        data = ("System","Terminal","Fixedsys","Modern","Helvetica","Roman","Script","Courier","MS Serif","MS Sans Serif","Small Fonts","Marlett","Arial","Calibri","Consolas")
        w = Combobox(m28, values=data)
        w.current(0)
        w.pack(side=tk.RIGHT)

        m29 = tk.PanedWindow(m24, orient=tk.VERTICAL)
        m24.add(m29)
        m29.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m29, text='Color:').pack(side=tk.LEFT)
        var = tk.StringVar()
        var.set("black")
        data = (
            "green", "red", "blue", "yellow", "gray",
            "orange",
            "black")
        w = Combobox(m29, values=data)
        w.current(0)
        w.pack(side=tk.RIGHT)

        m30 = tk.PanedWindow(m24, orient=tk.VERTICAL)
        m24.add(m30)
        m30.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m30, text='Style:').pack(side=tk.LEFT)
        var = tk.StringVar()
        var.set("normal")
        data = (
            "normal", "bold", "italic")
        w = Combobox(m30, values=data)
        w.current(0)
        w.pack(side=tk.RIGHT)

        m26 = tk.PanedWindow(m14, orient=tk.VERTICAL, height=30)
        m14.add(m26)
        m26.pack(fill=tk.BOTH, expand=0, side=tk.TOP)

        tk.Label(m26, text='Text Box:',font='Helvetica 14 bold',fg=color).pack(side=tk.LEFT)
        m25 = tk.PanedWindow(m14, orient=tk.VERTICAL, height=30)
        m14.add(m25)
        m25.pack(fill=tk.BOTH, expand=1, side=tk.TOP)

        m31 = tk.PanedWindow(m25, orient=tk.VERTICAL)
        m25.add(m31)
        m31.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m31, text='Font Size:').pack(side=tk.LEFT)
        w = tk.Spinbox(m31, from_=0, to=36, width=5)
        w.pack(side=tk.RIGHT)

        m31 = tk.PanedWindow(m25, orient=tk.VERTICAL)
        m25.add(m31)
        m31.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m31, text='Type:').pack(side=tk.LEFT)
        var = tk.StringVar()
        var.set("Helvetica")
        data = ("System", "Terminal", "Fixedsys", "Modern","Helvetica", "Roman", "Script", "Courier", "MS Serif", "MS Sans Serif",
                "Small Fonts", "Marlett", "Arial", "Calibri", "Consolas")
        w = Combobox(m31, values=data)
        w.current(0)
        w.pack(side=tk.RIGHT)

        m32 = tk.PanedWindow(m25, orient=tk.VERTICAL)
        m25.add(m32)
        m32.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m32, text='Color:').pack(side=tk.LEFT)
        var = tk.StringVar()
        var.set("black")
        data = (
            "green", "red", "blue", "yellow", "gray",
            "orange",
            "black")
        w = Combobox(m32, values=data)
        w.current(0)
        w.pack(side=tk.RIGHT)

        m33 = tk.PanedWindow(m25, orient=tk.VERTICAL)
        m25.add(m33)
        m33.pack(fill=tk.BOTH, expand=1, side=tk.TOP)
        tk.Label(m33, text='Style:').pack(side=tk.LEFT)
        var = tk.StringVar()
        var.set("normal")
        data = (
            "normal", "bold", "italic")
        w = Combobox(m33, values=data)
        w.current(0)
        w.pack(side=tk.RIGHT)

        app.mainloop()

    def makeScreenshot(self):
        self.screenshot = True
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes - 1)
        self.screenshot_folder = filedialog.askdirectory()
        if self.screenshot_folder == "":
            self.screenshot = False
        else:
            self.show_frame()
            self.screenshot_folder = ""
            self.screenshot = False

    def createMenuLeft(self):
        m1 = tk.PanedWindow()
        m1.pack(fill=tk.BOTH, expand=0, side=tk.LEFT)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

        m2 = tk.PanedWindow(m1, orient=tk.VERTICAL)
        m1.add(m2)

        # Creating a photoimage object to use image
        self.elipse = tk.PhotoImage(file=r'./data/elipse.png')
        self.line = tk.PhotoImage(file=r'./data/line.png')
        self.line_drop = tk.PhotoImage(file=r'./data/line_drop.png')
        self.quadrado = tk.PhotoImage(file=r'./data/quadrado.png')
        self.seta = tk.PhotoImage(file=r'./data/seta.png')
        self.seta_yellow = tk.PhotoImage(file=r'./data/seta_passe.png')
        self.icon = tk.PhotoImage(file=r'./data/icon.png')
        self.textBox = tk.PhotoImage(file=r'./data/textBox.png')

        # here, image option is used to
        # set image on button
        self.Elipse = tk.Button(m2, text="Draw Ellipse", image=self.elipse, command=self.elipseONOFF, compound="top")
        self.Elipse.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Elipse, text='Click on the exact place where the ellipse starts.\n'
                                        'And then click on the point where it ends.')

        # here, image option is used to
        # set image on button
        self.Line = tk.Button(m2, text="Draw Line", image=self.line, command=self.lineONOFF, compound="top")
        self.Line.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Line, text='Click on the player where the line will start.\n'
                                      'Then click on the second player where the line will end.')

        self.Line_Drop = tk.Button(m2, text="Drop Line", image=self.line_drop, command=self.line_dropONOFF,
                                   compound="top")
        self.Line_Drop.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Line_Drop, text='Click on the player that contains the line.')

        self.Line_Drop_All = tk.Button(m2, text="Drop Line All", image=self.line_drop, command=self.line_drop_all,
                                       compound="top")
        self.Line_Drop_All.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Line_Drop_All, text='Click on this button and delete all the lines.')

        self.Polly = tk.Button(m2, text="Draw polly\nbetween players", image=self.line, command=self.pollyONOFF,
                               compound="top")
        self.Polly.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Polly, text='Click on the players that will be part of the polygon.')

        self.Polly_Drop = tk.Button(m2, text="Remove polly\nbetween players", image=self.line,
                                    command=self.polly_dropONOFF,
                                    compound="top")
        self.Polly_Drop.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Polly_Drop, text='Click on the polygon player you want to remove.')

        self.Quadrado = tk.Button(m2, text="Draw Area", image=self.quadrado, command=self.quadradoONOFF, compound="top")
        self.Quadrado.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Quadrado, text='Click on the exact place where the square starts.\n'
                                          'And then click on the point where it ends.')

        self.Seta = tk.Button(m2, text="Draw Movement", image=self.seta, command=self.setaONOFF, compound="top")
        self.Seta.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Seta, text='Click on the exact spot where the movement starts.\n'
                                      'And then click on the place where it ends.')
        self.Seta_Passe = tk.Button(m2, text="Draw Pass", image=self.seta_yellow, command=self.seta_passeONOFF,
                                    compound="top")
        self.Seta_Passe.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Seta_Passe, text='Click on the exact spot where the pass starts.\n'
                                            'And then click on the place where it ends.')

        self.Icon = tk.Button(m2, text="Select Player", image=self.icon, command=self.selectONOFF, compound="top")
        self.Icon.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Icon, text='Click on the player you want to select.\n'
                                      'Click again to deselect')

        self.TextBox = tk.Button(m2, text="Draw textbox", image=self.textBox, command=self.textBoxONOFF, compound="top")
        self.TextBox.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.TextBox, text='Click on the exact spot where the textbox starts.\n'
                                         'And then click on the exact spot where the textbox ends')

        # variavel que serve para guardar a cor original do butao
        self.orig_color = self.Icon.cget("background")

        def colorsButtons():
            if self.selectON == False:
                self.Icon.configure(bg=self.orig_color)
            if self.LineON == False:
                self.Line.configure(bg=self.orig_color)
            if self.elipseON == False:
                self.Elipse.configure(bg=self.orig_color)
            if self.quadradoON == False:
                self.Quadrado.configure(bg=self.orig_color)
            if self.setaON == False:
                self.Seta.configure(bg=self.orig_color)
            if self.line_dropON == False:
                self.Line_Drop.configure(bg=self.orig_color)
            if self.seta_passeON == False:
                self.Seta_Passe.configure(bg=self.orig_color)
            if self.pollyON == False:
                self.Polly.configure(bg=self.orig_color)
            if self.polly_dropON == False:
                self.Polly_Drop.configure(bg=self.orig_color)
            if self.textBoxON == False:
                self.TextBox.configure(bg=self.orig_color)

            m2.after(1000, colorsButtons)

        colorsButtons()

    def createMenuRight(self):
        m1 = tk.PanedWindow()
        m1.pack(fill=tk.BOTH, expand=0, side=tk.RIGHT)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

        m2 = tk.PanedWindow(m1, orient=tk.VERTICAL)
        m1.add(m2)
        var = tk.StringVar()
        var.set("Select player")
        data = (
            "Select player", "Lines between players", "Polly between players", "Rectangle", "Elipse",
            "Arrow of Movement",
            "Arrow of Passed")
        self.cb = Combobox(m2, values=data)
        self.cb.current(0)
        self.cb.pack(fill=tk.BOTH, side=tk.TOP)

        m6 = tk.PanedWindow(m2, orient=tk.VERTICAL)
        m2.add(m6)
        m6.pack(fill=tk.BOTH, side=tk.TOP)
        self.frame1 = tk.Frame(m6)
        self.frame1.pack(fill=tk.BOTH, side=tk.LEFT)
        m9 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m9.pack(fill=tk.BOTH, side=tk.TOP, expand=1)
        blue = tk.Button(m9, bg='blue', command=self.blue)
        blue.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        yellow = tk.Button(m9, bg='yellow', command=self.yellow)
        yellow.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        green = tk.Button(m9, bg='green', command=self.green)
        green.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        red = tk.Button(m9, bg='red', command=self.red)
        red.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        m10 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m10.pack(fill=tk.BOTH, side=tk.TOP, expand=1)
        orange = tk.Button(m10, bg='orange', command=self.orange)
        orange.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        black = tk.Button(m10, bg='black', command=self.black)
        black.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        gray = tk.Button(m10, bg='gray', command=self.gray)
        gray.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        white = tk.Button(m10, bg='white', command=self.white)
        white.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        m15 = tk.PanedWindow(self.frame1, orient=tk.HORIZONTAL, height=10)
        m15.pack(fill=tk.BOTH, side=tk.TOP)
        m8 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m8.pack(fill=tk.BOTH, side=tk.TOP, expand=1)
        tk.Label(m8, text='Custom color:').pack(side=tk.LEFT)
        m11 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m11.pack(fill=tk.BOTH, side=tk.TOP, expand=1)
        self.color = tk.PanedWindow(m11, orient=tk.HORIZONTAL, bg='white', height=30)
        self.color.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        m3 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m3.pack(fill=tk.BOTH, side=tk.TOP, expand=1)
        tk.Label(m3, text='Red:').pack(side=tk.LEFT)
        self.vermelho = tk.Scale(m3, from_=0, to=255, orient=tk.HORIZONTAL, command=self.misturar)
        self.vermelho.pack(side=tk.RIGHT)
        m4 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m4.pack(fill=tk.BOTH, side=tk.TOP)
        tk.Label(m4, text='Green:').pack(side=tk.LEFT)
        self.verde = tk.Scale(m4, from_=0, to=255, orient=tk.HORIZONTAL, command=self.misturar)
        self.verde.pack(side=tk.RIGHT)
        m5 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m5.pack(fill=tk.BOTH, side=tk.TOP)
        tk.Label(m5, text='Blue:').pack(side=tk.LEFT)
        self.azul = tk.Scale(m5, from_=0, to=255, orient=tk.HORIZONTAL, command=self.misturar)
        self.azul.pack(side=tk.RIGHT)

        # List of players
        self.listbox = tk.Listbox(m2)
        self.listbox.pack(fill=tk.BOTH, side=tk.TOP)

        self.list = []
        # painel para mudar o nome
        m16 = tk.PanedWindow(self.frame1, orient=tk.HORIZONTAL, height=10)
        m16.pack(fill=tk.BOTH, side=tk.TOP)
        m14 = tk.PanedWindow(self.frame1, orient=tk.VERTICAL)
        m14.pack(fill=tk.BOTH, side=tk.TOP)
        tk.Label(m14, text='List of objects detected:').pack(side=tk.LEFT)
        m12 = tk.PanedWindow(m2, orient=tk.VERTICAL, height=30)
        m12.pack(fill=tk.BOTH, side=tk.TOP, expand=0)
        tk.Label(m12, text='Name:').pack(side=tk.LEFT)
        self.name = tk.StringVar()
        self.nameEntered = ttk.Entry(m12, width=15, textvariable=self.name)
        self.nameEntered.pack(fill=tk.BOTH, side=tk.RIGHT, expand=1)
        m13 = tk.PanedWindow(m2, orient=tk.VERTICAL, height=30, width=200)
        m13.pack(fill=tk.BOTH, side=tk.TOP, expand=0)
        changeName = tk.Button(m13, text='Change name of player', state=tk.DISABLED, command=self.changeNamePlayer)
        changeName.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)

        def updateButton():
            selecionado = str(self.listbox.curselection())
            if selecionado != "":
                changeName.config(state=tk.NORMAL)
            m2.after(100, updateButton)

        updateButton()

    def changeNamePlayer(self):
        try:
            selecionado = str(self.listbox.curselection())
            cont_virgula = 2
            id_selecionado = selecionado[1]
            while cont_virgula < len(selecionado) and selecionado[cont_virgula] != ',':
                id_selecionado = id_selecionado + selecionado[cont_virgula]
                cont_virgula = cont_virgula + 1

            print(id_selecionado)
            resultado_selecionado = self.list[int(id_selecionado)]
            print(self.list)
            cont_espaço = 1
            id_player_selecionado = resultado_selecionado[0]
            while cont_espaço < len(resultado_selecionado) and resultado_selecionado[cont_espaço] != ' ':
                id_player_selecionado = id_player_selecionado + resultado_selecionado[cont_espaço]
                cont_espaço = cont_espaço + 1

            print(resultado_selecionado)
            print(id_player_selecionado)
            print(self.name.get())
            if self.array_lists_id_with_names.count(int(id_player_selecionado)) > 0:
                cont_encontrar_id = 0
                while cont_encontrar_id < len(self.array_lists_id_with_names) and self.array_lists_id_with_names[
                    cont_encontrar_id] != int(id_player_selecionado):
                    cont_encontrar_id = cont_encontrar_id + 1
                self.array_lists_names_of_player[cont_encontrar_id] = str(self.name.get())
            else:
                self.array_lists_id_with_names.append(int(id_player_selecionado))
                self.array_lists_names_of_player.append(str(self.name.get()))
            self.nameEntered.delete(0, tk.END)
        except:
            self.nameEntered.delete(0, tk.END)

        self.start()

    def blue(self):
        self.verde.set(0)
        self.azul.set(255)
        self.vermelho.set(0)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def green(self):
        self.verde.set(128)
        self.azul.set(0)
        self.vermelho.set(0)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def red(self):
        self.verde.set(0)
        self.azul.set(0)
        self.vermelho.set(255)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def yellow(self):
        self.verde.set(255)
        self.azul.set(0)
        self.vermelho.set(255)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def white(self):
        self.verde.set(255)
        self.azul.set(255)
        self.vermelho.set(255)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def gray(self):
        self.verde.set(128)
        self.azul.set(128)
        self.vermelho.set(128)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def black(self):
        self.verde.set(0)
        self.azul.set(0)
        self.vermelho.set(0)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def orange(self):
        self.verde.set(128)
        self.azul.set(0)
        self.vermelho.set(255)
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        self.color.configure(bg=cor)

    def misturar(self, v):
        cor = "#%02x%02x%02x" % (int(self.vermelho.get()),
                                 int(self.verde.get()),
                                 int(self.azul.get()))
        # self.canvas.delete('retangulo')
        # self.canvas.create_rectangle(50, 25, 150, 75, fill=cor, tag='retangulo')
        self.color.configure(bg=cor)
        # self.rgb['text'] = cor
        self.vermelho.focus_force()

        if self.cb.get() == "Lines between players":
            self.color_line = (int(self.azul.get()),
                               int(self.verde.get()),
                               int(self.vermelho.get()))
        if self.cb.get() == "Arrow of Movement":
            self.color_movement = (int(self.azul.get()),
                                   int(self.verde.get()),
                                   int(self.vermelho.get()))
        if self.cb.get() == "Arrow of Passes":
            self.color_passed = (int(self.azul.get()),
                                 int(self.verde.get()),
                                 int(self.vermelho.get()))
        if self.cb.get() == "Select player":
            self.color_select = (int(self.azul.get()),
                                 int(self.verde.get()),
                                 int(self.vermelho.get()))
        if self.cb.get() == "Polly between players":
            self.color_polly = (int(self.azul.get()),
                                int(self.verde.get()),
                                int(self.vermelho.get()))
        if self.cb.get() == "Rectangle":
            self.color_rectangle = (int(self.azul.get()),
                                    int(self.verde.get()),
                                    int(self.vermelho.get()))
        if self.cb.get() == "Ellipse":
            self.color_elipse = (int(self.azul.get()),
                                 int(self.verde.get()),
                                 int(self.vermelho.get()))

    def createMenuBottom(self):
        m1 = tk.PanedWindow(orient=tk.VERTICAL)
        m1.pack(fill=tk.BOTH, expand=0, side=tk.BOTTOM)

        m0 = tk.PanedWindow(m1, orient=tk.HORIZONTAL)
        m1.add(m0)

        self.w2 = tk.Scale(m0, from_=0, to=self.length, variable=self.selectFrame, orient=tk.HORIZONTAL,
                           command=self.selectFrameScale)
        self.w2.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)

        m2 = tk.PanedWindow(m1, orient=tk.HORIZONTAL)
        m1.add(m2)

        var = tk.StringVar()
        labelframes = tk.Label(m2, textvariable=var, relief=tk.RAISED, borderwidth=0)
        if (self.isLogo == False):
            var.set("Frames:" + str(self.numframes) + "/" + str(self.length))
        else:
            var.set("")
        labelframes.pack(fill=tk.BOTH, side=tk.LEFT)

        def scaleFrames():
            self.w2.set(self.numframes)
            m2.after(1000, scaleFrames)

        scaleFrames()

        def labelFrames():
            if (self.isLogo == False):
                var.set("Frames:" + str(self.numframes) + "/" + str(self.length))
            else:
                var.set("")
            labelframes.config(text=var)
            m2.after(100, labelFrames)

        labelFrames()

        m3 = tk.PanedWindow(m2, orient=tk.HORIZONTAL)
        m2.add(m3)
        m3.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relheight=1.0)

        self.imageplay = tk.PhotoImage(file=r'./data/play.png')
        self.imagepause = tk.PhotoImage(file=r'./data/pause.png')
        self.imagezoom = tk.PhotoImage(file=r'./data/zoom.png')
        self.imagezoomout = tk.PhotoImage(file=r'./data/zoom_out.png')
        self.imagenext = tk.PhotoImage(file=r'./data/proximo.png')
        self.imageback = tk.PhotoImage(file=r'./data/anterior.png')

        Back = tk.Button(m3, image=self.imageback, command=self.back)
        Back.pack(fill=tk.BOTH, side=tk.LEFT)
        Play = tk.Button(m3, image=self.imageplay, command=self.start)
        Play.pack(fill=tk.BOTH, side=tk.LEFT)
        Pause = tk.Button(m3, image=self.imagepause, command=self.stop)
        Pause.pack(fill=tk.BOTH, side=tk.LEFT)
        Next = tk.Button(m3, image=self.imagenext, command=self.next)
        Next.pack(fill=tk.BOTH, side=tk.LEFT)

    def line_drop_all(self):
        self.line_player_1 = []
        self.line_player_2 = []
        self.line_active = []
        self.start()

    def show_frame(self):
        t1 = time.time()
        global running
        _, frame = self.cap.read()
        # frame = cv2.flip(frame, 0)
        if self.pause == False:
            self.numframes = self.numframes + 1
        print(self.numframes)

        frame = imutils.resize(frame, width=width_screen - 400)

        print(self.pause)

        # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Image.fromarray( obj , mode = None )
        # obj - Objeto com interface de matriz
        # mode - Modo a ser usado (será determinado a partir do tipo se None) Consulte:
        # img1=img
        # img1 = imutils.resize(img1, width=900)

        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_in = tf.expand_dims(img1, 0)
        img_in = transform_images(img_in, FLAGS.size)
        # img_in = Image.fromarray(img_in);

        # original: shape=(1, 288, 288, 3)
        # simple_example(MISTURA_COM_CV2).py: shape=(1, 288, 288, 4)
        boxes, scores, classes, nums = self.yolo.predict(img_in)

        classes = classes[0]
        names = []

        for i in range(len(classes)):
            names.append(self.class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(frame, boxes[0])
        features = self.encoder(frame, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        # initialize color map
        # cmap = plt.get_cmap('tab20b')
        # colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        cont_objects_positions_id = 0
        cont_objects_positions_x_min = 0
        cont_objects_positions_y_min = 0
        cont_objects_positions_x_max = 0
        cont_objects_positions_y_max = 0
        self.list = []

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            if self.list.count(str(track.track_id) + " - " + str(class_name)):
                print("")
            else:
                self.list.append(str(track.track_id) + " - " + str(class_name))

            # Se o id do track estiver no array de ids dos jogadores selecionados o rectagulo irá ser desenhado com uma cor diferente
            if self.contain(int(track.track_id)):
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), 2)
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                #              (int(bbox[0]) + (len(str(track.track_id))) * 5, int(bbox[1])),
                #              (255, 0, 255), -1)

                overlay_detect = frame.copy()
                alpha_detect = 0.4

                cv2.ellipse(overlay_detect, (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[3])), (36, 4), 0, 0,
                            360,
                            self.color_select, -1, 15)

                print(self.color_select)
                red, green, blue = str(self.color_select).split(',')
                red = str(red)[1:]
                blue = str(blue)[:len(blue) - 1]

                cv2.ellipse(overlay_detect, (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[3])), (36, 4), 0, 0,
                            360,
                            (int(red) - 100, int(green) - 100, int(blue) - 100), 2, 15)
                frame = cv2.addWeighted(overlay_detect, alpha_detect, frame, 1 - alpha_detect, 0)

            else:
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255) , 2)
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                #              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])),  (0, 0, 255), -1)
                if self.pause == True and self.screenshot == False:
                    overlay_detect = frame.copy()
                    alpha_detect = 0.4
                    cv2.ellipse(overlay_detect, (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[3])), (36, 4), 0, 0,
                                360,
                                (200, 200, 200), -1, 15)
                    cv2.ellipse(overlay_detect, (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[3])), (36, 4), 0, 0,
                                360,
                                (100, 100, 100), 2, 15)
                    frame = cv2.addWeighted(overlay_detect, alpha_detect, frame, 1 - alpha_detect, 0)

            if self.pause == False:
                if self.array_lists_id_with_names.count(track.track_id) > 0:
                    count = 0
                    while self.array_lists_id_with_names[count] != track.track_id:
                        count = count + 1
                    cv2.putText(frame, str(self.array_lists_names_of_player[count]),
                                (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                                (255, 255, 255), 1)
                    try:
                        id = self.list.index(str(track.track_id) + " - Player", 0, len(self.list))
                        self.list[id] = str(track.track_id) + " - " + str(self.array_lists_names_of_player[count])
                    except:
                        print("Player nao encontrado")

                else:
                    cv2.putText(frame, "", (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255),
                                1)

            else:
                if self.screenshot == False:
                    if self.array_lists_id_with_names.count(track.track_id) > 0:
                        count = 0
                        while self.array_lists_id_with_names[count] != track.track_id:
                            count = count + 1
                        cv2.putText(frame, str(track.track_id) + " - " + str(self.array_lists_names_of_player[count]),
                                    (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                                    (255, 255, 255), 1)
                        try:
                            id = self.list.index(str(track.track_id) + " - Player", 0, len(self.list))
                            self.list[id] = str(track.track_id) + " - " + str(self.array_lists_names_of_player[count])
                        except:
                            print("Player nao encontrado")

                    else:
                        cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255),
                                    1)

            self.listbox.delete(0, tk.END)

            for item in self.list:
                self.listbox.insert(tk.END, item)

            # cada jogador selecionado irá ter o id registado no array de ids, a posicao do x no array dos x e a posicao do y no array do y
            # self.objects_positions_id.insert(cont_objects_positions_id,track.track_id)
            # self.objects_positions_x.insert(cont_objects_positions_x,int(bbox[0]))
            # self.objects_positions_y.insert(cont_objects_positions_y,int(bbox[1]))
            self.objects_positions_id[cont_objects_positions_id] = track.track_id
            self.objects_positions_x_min[cont_objects_positions_x_min] = int(bbox[0])
            self.objects_positions_y_min[cont_objects_positions_y_min] = int(bbox[1])
            self.objects_positions_x_max[cont_objects_positions_x_max] = int(bbox[2])
            self.objects_positions_y_max[cont_objects_positions_y_max] = int(bbox[3])

            # Incrementar contadores dos arrays para que cada posicao dos arrays coecidir com um jogador
            cont_objects_positions_id = cont_objects_positions_id + 1
            cont_objects_positions_x_min = cont_objects_positions_x_min + 1
            cont_objects_positions_y_min = cont_objects_positions_y_min + 1
            cont_objects_positions_x_max = cont_objects_positions_x_max + 1
            cont_objects_positions_y_max = cont_objects_positions_y_max + 1

        # Criacao das multiplas linhas
        def arrayLenght(array):
            cont = len(array) - 1
            while array[cont] == 0 and cont >= 0:
                cont = cont - 1
            return cont + 1

        cont_line_player1_id = 0
        cont_line_player2_id = 0
        if len(self.line_active) > 0:
            while cont_line_player1_id < len(self.line_active):
                player1 = 0
                player2 = 0
                cont = 0

                for n in self.objects_positions_id:
                    if int(n) == int(self.line_player_1[cont_line_player1_id]) and player1 == 0:
                        player1 = cont
                    if int(n) == int(self.line_player_2[cont_line_player2_id]) and player2 == 0:
                        player2 = cont
                    cont = cont + 1
                if self.line_active[cont_line_player1_id] > 0:
                    if self.line_player_1[cont_line_player1_id] == self.line_player_2[cont_line_player2_id]:
                        x_new_player1 = self.objects_positions_x_min[player1] + (
                                (self.objects_positions_x_max[player1] - self.objects_positions_x_min[player1]) / 2)
                        x_new_player2 = self.objects_positions_x_min[player2] + (
                                (self.objects_positions_x_max[player2] - self.objects_positions_x_min[player2]) / 2)

                        cv2.line(frame, (int(x_new_player1), self.objects_positions_y_max[player1]),
                                 (int(x_new_player2), self.objects_positions_y_max[player2]), (255, 255, 255), 5)
                    else:
                        x_new_player1 = self.objects_positions_x_min[player1] + (
                                (self.objects_positions_x_max[player1] - self.objects_positions_x_min[player1]) / 2)
                        x_new_player2 = self.objects_positions_x_min[player2] + (
                                (self.objects_positions_x_max[player2] - self.objects_positions_x_min[player2]) / 2)

                        cv2.line(frame, (int(x_new_player1), self.objects_positions_y_max[player1]),
                                 (int(x_new_player2), self.objects_positions_y_max[player2]), self.color_line, 5)
                cont_line_player1_id = cont_line_player1_id + 1
                cont_line_player2_id = cont_line_player2_id + 1

        # criacao das setas
        if self.frame_arrow_create[0] != 0:
            contador_setas = 0
            while contador_setas < arrayLenght(self.frame_arrow_create):
                start_point = (
                    int(self.coordinates_arrow_x_init[contador_setas]),
                    int(self.coordinates_arrow_y_init[contador_setas]))

                end_point = (int(self.coordinates_arrow_x_final[contador_setas]),
                             int(self.coordinates_arrow_y_final[contador_setas]))

                if int(self.coordinates_arrow_x_final[contador_setas]) == 0 and int(
                        self.coordinates_arrow_y_final[contador_setas]) == 0:
                    end_point = (int(self.coordinates_arrow_x_init[contador_setas]),
                                 int(self.coordinates_arrow_y_init[contador_setas]))

                color = self.color_passed
                thickness = 1
                type = cv2.LINE_AA
                v_seta= 0.2
                if self.arrow_type[contador_setas] == 1:
                    color = self.color_movement
                    thickness = 3
                    type = 0
                    v_seta = 0.1


                if int(self.numframes) - int(self.frame_arrow_create[contador_setas]) < 25:
                    cv2.arrowedLine(frame, start_point, end_point, color, thickness,type,0,v_seta)

                contador_setas = contador_setas + 1

        # criacao de elipses
        if self.frame_elipse_create[0] != 0:
            contador_elipse = 0
            while contador_elipse < arrayLenght(self.frame_elipse_create):
                start_point = (int(self.coordinates_elipse_x_init[contador_elipse]),
                               int(self.coordinates_elipse_y_init[contador_elipse]))

                end_point = (int(self.coordinates_elipse_x_final[contador_elipse]),
                             int(self.coordinates_elipse_y_final[contador_elipse]))

                if int(self.coordinates_elipse_x_final[contador_elipse]) == 0 and int(
                        self.coordinates_elipse_y_final[contador_elipse]) == 0:
                    end_point = (int(self.coordinates_elipse_x_init[contador_elipse]),
                                 int(self.coordinates_elipse_y_init[contador_elipse]))

                color = self.color_elipse

                if start_point[0] < end_point[0]:
                    center_x = int(start_point[0] + ((end_point[0] - start_point[0]) / 2))
                    elipse_height = int(((end_point[0] - start_point[0]) / 2) + 1)
                else:
                    center_x = int(end_point[0] + ((start_point[0] - end_point[0]) / 2))
                    elipse_height = int(((start_point[0] - end_point[0]) / 2) + 1)

                if start_point[1] < end_point[1]:
                    center_y = int(start_point[1] + ((end_point[1] - start_point[1]) / 2))
                    elipse_widht = int(((end_point[1] - start_point[1]) / 2) + 1)
                else:
                    center_y = int(end_point[1] + ((start_point[1] - end_point[1]) / 2))
                    elipse_widht = int(((start_point[1] - end_point[1]) / 2) + 1)

                if int(self.numframes) - int(self.frame_elipse_create[contador_elipse]) < 25:
                    overlay_elipse = frame.copy()
                    alpha_elipse = 0.5
                    cv2.ellipse(overlay_elipse, (center_x, center_y),
                                (elipse_height, elipse_widht), 0, 0, 360, color, -1)

                    frame = cv2.addWeighted(overlay_elipse, alpha_elipse, frame, 1 - alpha_elipse, 0)

                contador_elipse = contador_elipse + 1

        # criacao de quadrados/ retangulos
        if self.frame_rectangle_create[0] != 0:
            contador_rectangulos = 0
            while contador_rectangulos < arrayLenght(self.frame_rectangle_create):
                start_point = (int(self.coordinates_rectangle_x_init[contador_rectangulos]),
                               int(self.coordinates_rectangle_y_init[contador_rectangulos]))

                end_point = (int(self.coordinates_rectangle_x_final[contador_rectangulos]),
                             int(self.coordinates_rectangle_y_final[contador_rectangulos]))

                if int(self.coordinates_rectangle_x_final[contador_rectangulos]) == 0 and int(
                        self.coordinates_rectangle_y_final[contador_rectangulos]) == 0:
                    end_point = (int(self.coordinates_rectangle_x_init[contador_rectangulos]),
                                 int(self.coordinates_rectangle_y_init[contador_rectangulos]))

                color = self.color_rectangle

                thickness = -1

                if int(self.numframes) - int(self.frame_rectangle_create[contador_rectangulos]) < 25:
                    overlay_detect = frame.copy()
                    alpha_detect = 0.4
                    cv2.rectangle(overlay_detect, start_point, end_point, color, thickness)
                    frame = cv2.addWeighted(overlay_detect, alpha_detect, frame, 1 - alpha_detect, 0)

                contador_rectangulos = contador_rectangulos + 1

            # if FLAGS.output:

        # criacao de poligunos entre os jogadores

        if arrayLenght(self.objects_positions_id) > 0:  # se existir jogadores podemos criar poligonos
            print(self.get_id_player(2))
            if len(self.array_lists_polly_players_ONOFF) > 0:  # se existir poligonos criados podemos criar as linhas
                contador_polly = 0  # contador de poligonos
                while contador_polly < len(
                        self.array_lists_polly_players_ONOFF):  # enquanto um nao passarmos pelos poligonos todos
                    if self.array_lists_polly_players_ONOFF[contador_polly] == 1:
                        pts = []
                        nppts = np.array([])
                        arr1 = np.array([])
                        arr2 = np.array([])
                        cont = 0  # contador de cada jogador de cada poligono
                        while cont < len(self.array_lists_polly_players[
                                             contador_polly]):  # enquanto nao passarmos pelos jogadores todos
                            try:
                                coordenates_player_1 = self.get_id_player(int(
                                    self.array_lists_polly_players[contador_polly][
                                        cont]))  # vamos buscar as coordenadas do jogador atual

                                # vamos adicionar ponto a ponto de cada jogador no array de pontos
                                pts.append([])
                                pts[cont].append(int(coordenates_player_1[0]))
                                pts[cont].append(int(coordenates_player_1[1]))

                                arr1 = np.append(arr1, [int(pts[cont][0])])
                                arr2 = np.append(arr2, [int(pts[cont][1])])
                            except:
                                print("Error polly")
                            # incrementar contador
                            cont = cont + 1
                        # Draw polygon
                        nppts = np.stack((arr1, arr2), axis=1)
                        nppts = nppts.astype(int)
                        overlay_poly = frame.copy()
                        alpha_poly = 0.2

                        cv2.fillConvexPoly(overlay_poly, nppts, self.color_polly)
                        frame = cv2.addWeighted(overlay_poly, alpha_poly, frame, 1 - alpha_poly, 0)

                    contador_polly = contador_polly + 1

        # out.write(img)
        # cv2.imshow('output', img)

        if self.frame_textBox_create[0] != 0:
            contador_textBox = 0
            while contador_textBox < arrayLenght(self.frame_textBox_create):
                start_point = (int(self.coordinates_textBox_x_init[contador_textBox]),
                               int(self.coordinates_textBox_y_init[contador_textBox]))

                color = (0, 0, 0)
                # textInput = "Passe errado do Sergio Ramos"
                # textInput = input("Text to add.\n")

                if int(self.numframes) - int(self.frame_textBox_create[contador_textBox]) < 25:
                    overlay_textBox = frame.copy()
                    alpha_textBox = 0.3
                    try:
                        cv2.rectangle(overlay_textBox, (int(self.coordinates_textBox_x_init[contador_textBox] - 5),
                                                        int(self.coordinates_textBox_y_init[contador_textBox] - 30)),
                                      (int(self.coordinates_textBox_x_init[contador_textBox]) + (
                                              (len(self.coordinates_textBox_text[contador_textBox])) * 13) + 5,
                                       int(self.coordinates_textBox_y_init[contador_textBox] + 20)),
                                      (255, 255, 255), -1)
                        frame = cv2.addWeighted(overlay_textBox, alpha_textBox, frame, 1 - alpha_textBox, 0)
                        cv2.putText(frame, self.coordinates_textBox_text[contador_textBox], start_point,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except:
                        print("Waiting for text box")

                contador_textBox = contador_textBox + 1



        scale_percent = self.zoom  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        if self.numframes == 1:
            self.w_zoom = width
            self.h_zoom = height
        print("Local zoom")
        print(width)
        print(height)
        (rows, cols) = frame.shape[:2]

        if self.zoom == 150:
            if self.zona_de_zoom == "top_left":
                M = np.float32([[1, 0, (cols/6)], [0, 1, (rows/7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_left_medium":
                M = np.float32([[1, 0, (cols / 6)], [0, 1, (rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_left_center":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, (rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_left_center_medium":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, (rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right":
                M = np.float32([[1, 0, -(cols/6)], [0, 1, (rows/7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right_medium":
                M = np.float32([[1, 0, -(cols / 6)], [0, 1, (rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right_center":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, (rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right_center_medium":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, (rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left":
                M = np.float32([[1, 0, (cols/6)], [0, 1, -(rows/7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left_medium":
                M = np.float32([[1, 0, (cols / 6)], [0, 1, -(rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left_center":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, -(rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left_center_medium":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, -(rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right":
                M = np.float32([[1, 0, -(cols/6)], [0, 1, -(rows/7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right_medium":
                M = np.float32([[1, 0, -(cols / 6)], [0, 1, -(rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right_center":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, -(rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right_center_medium":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, -(rows / 10)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
        if self.zoom == 200:
            if self.zona_de_zoom == "top_left":
                M = np.float32([[1, 0, (cols/4)], [0, 1, (rows/5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_left_medium":
                M = np.float32([[1, 0, (cols/4)], [0, 1, (rows/7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_left_center":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, (rows / 5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_left_center_medium":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, (rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right":
                M = np.float32([[1, 0, -(cols/4)], [0, 1, (rows/5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right_medium":
                M = np.float32([[1, 0, -(cols/4)], [0, 1, (rows/7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right_center":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, (rows / 5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "top_right_center_medium":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, (rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left":
                M = np.float32([[1, 0, (cols/4)], [0, 1, -(rows/5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left_medium":
                M = np.float32([[1, 0, (cols / 4)], [0, 1, -(rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left_center":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, -(rows / 5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_left_center_medium":
                M = np.float32([[1, 0, (cols / 15)], [0, 1, -(rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right":
                M = np.float32([[1, 0, -(cols / 4)], [0, 1, -(rows / 5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right_medium":
                M = np.float32([[1, 0, -(cols/4)], [0, 1, -(rows/7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right_center":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, -(rows / 5)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))
            if self.zona_de_zoom == "bottom_right_center_medium":
                M = np.float32([[1, 0, -(cols / 15)], [0, 1, -(rows / 7)]])
                frame = cv2.warpAffine(frame, M, (cols, rows))

        dim = (width, height)

        # resize image
        img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        if self.screenshot == True:
            cv2.imwrite(self.screenshot_folder + "\\frame%d.jpg" % self.numframes, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Cria uma memória de imagem de um objeto que exporta a interface da matriz
        imgtk = ImageTk.PhotoImage(
            image=img)  # usada para exibir imagens (em escala de cinza ou em cores verdadeiras) em rótulos, botões, telas e widgets de texto
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)

        t2 = time.time()
        self.times.append(t2 - t1)
        self.times = self.times[-20:]
        print("fps: {:.2f}".format(round(1 / ((sum(self.times) / len(self.times))), 2)))

        # key = cv2.waitKey(1)
        self.lmain.bind('<MouseWheel>', self.change_zoom)
        self.lmain.bind('<Leave>', self.exit_)
        self.lmain.bind('<Button-1>',
                        self.motion)  # quando alguem clica na tela de jogo o irá imediatamente assionar a funcao self.motion

        if not self.pause:
            self.lmain.after(5, self.show_frame)

        # after (pai, ms, função = Nenhum, * args)
        # Parâmetros:
        # parent : é o objeto do widget ou da janela principal, o que estiver usando esta função.
        # ms : é o tempo em milissegundos.
        # função : que deve ser chamada.
        # * args : outras opções.

    def change_zoom(self,event):
        self.stop()
        print("Coordenates")
        x = event.x
        y = event.y
        if int(event.delta) > 0 and (self.zoom + 50) < 250:
            self.zoom = self.zoom + 50
            # warpAffine does appropriate shifting given the
            # translation matrix.

            if x < (self.w_zoom / 4) and y<(self.h_zoom/4):
                self.zona_de_zoom = "top_left"
            elif x < (self.w_zoom / 4)  and y < (self.h_zoom / 2):
                self.zona_de_zoom = "top_left_medium"
            elif x < (self.w_zoom / 2)  and y < (self.h_zoom / 4):
                self.zona_de_zoom = "top_left_center"
            elif x > (self.w_zoom / 4) and x < (self.w_zoom / 2) and y > (self.h_zoom / 4) and y < (self.h_zoom / 2):
                self.zona_de_zoom = "top_left_center_medium"
            elif x > (self.w_zoom - (self.w_zoom / 4))  and y < (self.h_zoom / 4):
                self.zona_de_zoom = "top_right"
            elif x > (self.w_zoom - (self.w_zoom / 4))  and y < (self.h_zoom / 2):
                self.zona_de_zoom = "top_right_medium"
            elif x > (self.w_zoom / 2)  and y < (self.h_zoom / 4):
                self.zona_de_zoom = "top_right_center"
            elif x > (self.w_zoom / 2) and x < (self.w_zoom - (self.w_zoom / 4)) and y > (self.h_zoom / 4) and y < (self.h_zoom / 2):
                self.zona_de_zoom = "top_right_center_medium"
            elif x < (self.w_zoom / 4)  and y > (self.h_zoom - (self.h_zoom / 4)):
                self.zona_de_zoom = "bottom_left"
            elif x < (self.w_zoom / 4)  and y > (self.h_zoom / 2):
                self.zona_de_zoom = "bottom_left_medium"
            elif x < (self.w_zoom / 2)  and y > (self.h_zoom - (self.h_zoom / 4)):
                self.zona_de_zoom = "bottom_left_center"
            elif x < (self.w_zoom / 2) and x > (self.w_zoom / 4) and y > (self.h_zoom / 2) and y < (self.h_zoom - (self.h_zoom / 4)):
                self.zona_de_zoom = "bottom_left_center_medium"
            elif x > (self.w_zoom - (self.w_zoom / 4)) and y > (self.h_zoom - (self.h_zoom / 4)):
                self.zona_de_zoom = "bottom_right"
            elif x > (self.w_zoom - (self.w_zoom / 4))  and y > (self.h_zoom / 2):
                self.zona_de_zoom = "bottom_right_medium"
            elif x > (self.w_zoom / 2)  and y > (self.h_zoom - (self.h_zoom / 4)):
                self.zona_de_zoom = "bottom_right_center"
            elif x > (self.w_zoom / 2) and x < (self.w_zoom - (self.w_zoom / 4)) and y > (self.h_zoom / 2) and y < (self.h_zoom - (self.h_zoom / 4)):
                self.zona_de_zoom = "bottom_right_center_medium"

            else:
                self.zona_de_zoom = "center"

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes - 1)
            self.show_frame()
        if int(event.delta) < 0 and (self.zoom - 50) > 50:
            self.zoom = self.zoom - 50
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes - 1)
            self.show_frame()

    # vamos criar um array que retorna a posicao atraves do id do jogador
    def get_id_player(self, number):
        cont = 0
        coordenates = (0, 0)
        array = []
        for o in self.objects_positions_id:
            x_player = self.objects_positions_x_min[cont] + (
                    (self.objects_positions_x_max[cont] - self.objects_positions_x_min[cont]) / 2)
            x_player_1 = self.objects_positions_y_max[cont]
            if o == number:
                array.insert(0, int(x_player))
                array.insert(1, int(x_player_1))
                coordenates = (int(x_player), int(x_player_1))
            cont = cont + 1

        return array

    def contain(self, number):
        for o in self.selecteds:
            if o == number:
                return True
        return False

    def motion(self, event):
        min_x = 0
        min_y = 0
        x, y = event.x, event.y  # x e y sao igualados ao x clicado no evento e ao y clicado no evento
        print('{}, {}'.format(x, y))

        def arrayLenght(array):
            cont = len(array) - 1
            while array[cont] == 0 and cont >= 0:
                cont = cont - 1
            return cont + 1

        cont_objects_positions_id = 0
        cont_objects_positions_x = 0
        cont_objects_positions_y = 0

        while self.objects_positions_id[cont_objects_positions_id] > 0:
            if ((x > self.objects_positions_x_min[cont_objects_positions_x]) and (
                    x < self.objects_positions_x_max[cont_objects_positions_x])):
                if ((y > self.objects_positions_y_min[cont_objects_positions_y]) and (
                        y < self.objects_positions_y_max[cont_objects_positions_y])):
                    # if ((x - self.objects_positions_x[cont_objects_positions_x]) > 0) & (
                    #        (y - self.objects_positions_y[cont_objects_positions_y]) > 0):
                    if self.selectON == True:  # caso a opcao de selecionar esteija ativa o program ira proceder a selecao
                        count = 0
                        while self.selecteds[count] > 0:
                            count = count + 1
                        if self.contain(self.objects_positions_id[cont_objects_positions_id]) == False:
                            self.selecteds[count] = self.objects_positions_id[cont_objects_positions_id]
                        else:
                            count = 0
                            while self.selecteds[count] != self.objects_positions_id[cont_objects_positions_id]:
                                count = count + 1
                            self.selecteds[count] = 0

                    if self.LineON == True:
                        count = 0
                        if self.players_selecionados == 0:
                            while count < len(self.line_player_1):
                                count = count + 1
                            # adicionar jogador no array player1
                            self.line_player_1.append(self.objects_positions_id[cont_objects_positions_id])
                            self.line_player_2.append(self.objects_positions_id[cont_objects_positions_id])
                            self.line_active.append(1)
                            self.players_selecionados = self.players_selecionados + 1
                        else:
                            while count < len(self.line_player_2):
                                count = count + 1
                                # adicionar jogador no array player1
                            self.line_player_2[count - 1] = self.objects_positions_id[cont_objects_positions_id]
                            self.players_selecionados = self.players_selecionados + 1
                            self.players_selecionados = 0

                    if self.line_dropON == True:
                        count = 0
                        while count < len(self.line_player_1):
                            if self.line_player_1[count] == self.objects_positions_id[cont_objects_positions_id]:
                                print("1137")
                                print(self.line_player_1[count])
                                print(self.objects_positions_id[cont_objects_positions_id])
                                self.line_active[count] = 0
                            count = count + 1
                        # remover jogador no array player1 e companheiro e companheiro do player1

                        # organizar array self.line_player1

                        # contador = 0
                        # while self.line_player1[contador] > 0:
                        #    contador = contador + 1
                        # while contador < (arrayLenght(self.line_player1) - 1):
                        #    self.line_player1[contador] = self.line_player1[contador + 1]
                        #    self.line_player2[contador] = self.line_player2[contador + 1]
                        #    contador = contador + 1

                        count = 0
                        while count < len(self.line_player_2):
                            if self.line_player_2[count] == self.objects_positions_id[cont_objects_positions_id]:
                                self.line_active[count] = 0
                            count = count + 1
                        # remover jogador no array player2 e companheiro do player1

                        # organizar array self.line_player2
                        # contador = 0
                        # while self.line_player2[contador] > 0:
                        #    contador = contador + 1
                        # while contador < (arrayLenght(self.line_player2) - 1):
                        #    self.line_player1[contador] = self.line_player1[contador + 1]
                        #    self.line_player2[contador] = self.line_player2[contador + 1]
                        #    contador = contador + 1

                    if self.pollyON == True:
                        if self.array_lists_polly_players[self.count_pollys].count(
                                self.objects_positions_id[cont_objects_positions_id]) < 1:
                            self.array_lists_polly_players[self.count_pollys].append(
                                self.objects_positions_id[cont_objects_positions_id])
                        print(self.array_lists_polly_players[self.count_pollys])

                    if self.polly_dropON == True:
                        contador_polly = 0
                        while contador_polly < len(self.array_lists_polly_players_ONOFF):
                            if self.array_lists_polly_players[contador_polly].count(
                                    self.objects_positions_id[cont_objects_positions_id]) > 0:
                                self.array_lists_polly_players_ONOFF[contador_polly] = 0
                            contador_polly = contador_polly + 1
            cont_objects_positions_id = cont_objects_positions_id + 1
            cont_objects_positions_x = cont_objects_positions_x + 1
            cont_objects_positions_y = cont_objects_positions_y + 1

        if self.setaON == True or self.seta_passeON == True:
            if self.num_of_click_arrow == 0:
                num = int(arrayLenght(self.frame_arrow_create))
                self.frame_arrow_create[num] = self.numframes
                self.coordinates_arrow_x_init[num] = x
                self.coordinates_arrow_y_init[num] = y
                if self.setaON == True:
                    self.arrow_type[num] = 1
                else:
                    if self.seta_passeON == True:
                        self.arrow_type[num] = 2
                self.num_of_click_arrow = self.num_of_click_arrow + 1

            else:
                num = int(arrayLenght(self.coordinates_arrow_x_final))
                self.coordinates_arrow_x_final[num] = x
                self.coordinates_arrow_y_final[num] = y
                self.num_of_click_arrow = 0

        if self.quadradoON == True:
            if self.num_of_click_rectangle == 0:
                num = int(arrayLenght(self.frame_rectangle_create))
                self.frame_rectangle_create[num] = self.numframes
                self.coordinates_rectangle_x_init[num] = x
                self.coordinates_rectangle_y_init[num] = y
                self.num_of_click_rectangle = self.num_of_click_rectangle + 1

            else:
                num = int(arrayLenght(self.coordinates_rectangle_x_final))
                self.coordinates_rectangle_x_final[num] = x
                self.coordinates_rectangle_y_final[num] = y
                self.num_of_click_rectangle = 0

        if self.elipseON == True:
            if self.num_of_click_elipse == 0:
                num = int(arrayLenght(self.frame_elipse_create))
                self.frame_elipse_create[num] = self.numframes
                self.coordinates_elipse_x_init[num] = x
                self.coordinates_elipse_y_init[num] = y
                self.num_of_click_elipse = self.num_of_click_elipse + 1

            else:
                num = int(arrayLenght(self.coordinates_elipse_x_final))
                self.coordinates_elipse_x_final[num] = x
                self.coordinates_elipse_y_final[num] = y
                self.num_of_click_elipse = 0

        if self.textBoxON == True:
            self.num = int(arrayLenght(self.frame_textBox_create))
            self.frame_textBox_create[self.num] = self.numframes
            self.coordinates_textBox_x_init[self.num] = x
            self.coordinates_textBox_y_init[self.num] = y

            self.masterTextBox = tk.Tk()

            tk.Label(self.masterTextBox, text="Text to add:").grid(row=0)
            self.e1 = tk.Entry(self.masterTextBox)
            self.e1.grid(row=1)
            tk.Button(self.masterTextBox, text='Confirm', command=self.saveTextBox).grid(row=3, sticky=tk.W, pady=5)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes - 1)
        self.show_frame()

    def saveTextBox(self):
        self.coordinates_textBox_text[self.num] = self.e1.get()
        self.masterTextBox.destroy()
        self.start()

    def selectONOFF(self):  # funcao que irá ativar a opcao selecionar
        if self.selectON == False:
            self.selectON = True
            self.Icon.configure(bg="gray")
            self.LineON = False
            self.quadradoON = False
            self.elipseON = False
            self.setaON = False
            self.line_dropON = False
            self.seta_passeON = False
            self.pollyON = False
            self.polly_dropON = False
            self.textBoxON = False
        else:
            self.selectON = False
            self.start()

    def lineONOFF(self):
        if self.LineON == False:
            self.LineON = True
            self.Line.configure(bg="gray")
            self.selectON = False
            self.quadradoON = False
            self.elipseON = False
            self.setaON = False
            self.line_dropON = False
            self.seta_passeON = False
            self.pollyON = False
            self.polly_dropON = False
            self.textBoxON = False
        else:
            self.LineON = False
            self.start()

    def line_dropONOFF(self):
        if self.line_dropON == False:
            self.line_dropON = True
            self.Line_Drop.configure(bg="gray")
            self.selectON = False
            self.quadradoON = False
            self.elipseON = False
            self.setaON = False
            self.LineON = False
            self.seta_passeON = False
            self.pollyON = False
            self.polly_dropON = False
            self.textBoxON = False
        else:
            self.line_dropON = False
            self.start()

    def elipseONOFF(self):
        if self.elipseON == False:
            self.elipseON = True
            self.Elipse.configure(bg="gray")
            self.selectON = False
            self.quadradoON = False
            self.LineON = False
            self.setaON = False
            self.line_dropON = False
            self.seta_passeON = False
            self.pollyON = False
            self.polly_dropON = False
            self.textBoxON = False
        else:
            self.LineON = False
            self.start()

    def quadradoONOFF(self):
        if self.quadradoON == False:
            self.quadradoON = True
            self.Quadrado.configure(bg="gray")
            self.selectON = False
            self.elipseON = False
            self.LineON = False
            self.setaON = False
            self.line_dropON = False
            self.seta_passeON = False
            self.pollyON = False
            self.polly_dropON = False
            self.textBoxON = False
        else:
            self.quadradoON = False
            self.start()

    def setaONOFF(self):
        if self.setaON == False:
            self.Seta.configure(bg="gray")
            self.setaON = True
            self.selectON = False
            self.elipseON = False
            self.LineON = False
            self.quadradoON = False
            self.line_dropON = False
            self.seta_passeON = False
            self.pollyON = False
            self.polly_dropON = False
            self.textBoxON = False
        else:
            self.setaON = False
            self.start()

    def seta_passeONOFF(self):
        if self.seta_passeON == False:
            self.Seta_Passe.configure(bg="gray")
            self.seta_passeON = True
            self.selectON = False
            self.elipseON = False
            self.LineON = False
            self.quadradoON = False
            self.line_dropON = False
            self.setaON = False
            self.pollyON = False
            self.polly_dropON = False
            self.textBoxON = False
        else:
            self.seta_passeON = False
            self.start()

    def pollyONOFF(self):
        if self.pollyON == False:
            self.Polly.configure(bg="gray")
            self.pollyON = True
            self.count_pollys = self.count_pollys + 1
            self.array_lists_polly_players.append([])
            self.array_lists_polly_players_ONOFF.append(1)
            self.selectON = False
            self.elipseON = False
            self.LineON = False
            self.quadradoON = False
            self.line_dropON = False
            self.setaON = False
            self.seta_passeON = False
            self.polly_dropON = False
            self.textBoxON = False

        else:
            self.pollyON = False
            self.start()

    def polly_dropONOFF(self):
        self.polly_dropON = False
        if self.polly_dropON == False:
            self.Polly_Drop.configure(bg="gray")
            self.polly_dropON = True
            self.pollyON = False
            self.selectON = False
            self.elipseON = False
            self.LineON = False
            self.quadradoON = False
            self.line_dropON = False
            self.setaON = False
            self.seta_passeON = False
            self.textBoxON = False
        else:
            self.polly_dropON = False
            self.start()

    def textBoxONOFF(self):
        if self.textBoxON == False:
            self.textBoxON = True
            self.TextBox.configure(bg="gray")
            self.selectON = False
            self.elipseON = False
            self.LineON = False
            self.quadradoON = False
            self.setaON = False
            self.line_dropON = False
            self.seta_passeON = False
            self.pollyON = False
            self.polly_dropON = False
        else:
            self.textBoxON = False
            self.start()

    def zoomin(self):
        if self.zoom < 300:
            self.zoom = self.zoom + 50
        self.start()

    def zoomout(self):
        if self.zoom > 50:
            self.zoom = self.zoom - 50
        self.start()

    def enter(self, event):
        self.stop()

    def exit_(self, event):
        # self.master.bind('<Enter>', self.enter)
        self.stop()

    def start(self):
        if self.pause == True:
            self.pause = False
            self.show_frame()

    def stop(self):
        self.pause = True
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes - 1)
        self.show_frame()

    def open_video(self):
        self.filename = askopenfilename(title="Select file", filetypes=(("*.mp4", "*.mp4"),
                                                                        ("*.wmv", "*.wmv"),
                                                                        ("*.avi", "*.avi")))
        self.pause = False
        print(self.filename)
        if self.filename:
            self.cap = cv2.VideoCapture(self.filename)
            # NUMBER OF FRAMES
            property_id = int(cv2.CAP_PROP_FRAME_COUNT)
            self.length = int(cv2.VideoCapture.get(self.cap, property_id))
            self.w2.config(to=self.length)
            self.isLogo = False
            self.numframes = 0

            self.clean_arrays()

        self.show_frame()

    def next(self):
        if (self.numframes + 100) < self.length:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes + 100)
            self.numframes = self.numframes + 100

            self.clean_arrays()

        self.start()

    def selectFrameScale(self, v):
        print("572: clicar no scale no")
        print(v)
        if (int(v) - self.numframes) > 20:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(v))
            self.numframes = int(v)
            self.clean_arrays()
            self.start()
        if (self.numframes - int(v)) > 20:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(v))
            self.numframes = int(v)
            self.clean_arrays()
            self.start()

    def clean_arrays(self):
        self.line_player_1 = []
        self.line_player_2 = []
        self.line_active = []

        self.array_lists_id_with_names = []
        self.array_lists_names_of_player = []

        self.array_lists_polly_players = []
        self.array_lists_polly_players_ONOFF = []

        self.initArray(self.selecteds)

        self.initArray(self.objects_positions_id)
        self.initArray(self.objects_positions_x_min)
        self.initArray(self.objects_positions_y_min)
        self.initArray(self.objects_positions_x_max)
        self.initArray(self.objects_positions_y_max)

        self.initArray(self.frame_arrow_create)
        self.initArray(self.coordinates_arrow_x_init)
        self.initArray(self.coordinates_arrow_y_init)
        self.initArray(self.coordinates_arrow_x_final)
        self.initArray(self.coordinates_arrow_y_final)
        self.initArray(self.arrow_type)

        self.initArray(self.frame_elipse_create)
        self.initArray(self.coordinates_elipse_x_init)
        self.initArray(self.coordinates_elipse_y_init)
        self.initArray(self.coordinates_elipse_x_final)
        self.initArray(self.coordinates_elipse_y_final)

        self.initArray(self.frame_rectangle_create)
        self.initArray(self.coordinates_rectangle_x_init)
        self.initArray(self.coordinates_rectangle_y_init)
        self.initArray(self.coordinates_rectangle_x_final)
        self.initArray(self.coordinates_rectangle_y_final)

        self.initArray(self.frame_textBox_create)
        self.initArray(self.coordinates_textBox_x_init)
        self.initArray(self.coordinates_textBox_y_init)
        self.initArray(self.coordinates_textBox_text)

    def back(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes - 100)
        if (self.numframes - 100) > 0:
            self.numframes = self.numframes - 100
            self.clean_arrays()
        self.start()

    def exit(self):
        print("destroy")
        self.master.destroy()


def main(_argv):
    root = tk.Tk()
    # root.title("DeepSports Eleven - Sports Analysis Software")
    # root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file=r'.\data\dp11.gif'))
    # var=str(width_screen-200) +"x"+ str(height_screen)
    # root.geometry(var)
    app = FUTOTAL(root, _argv)
    # root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file=r'.\data\dp11.gif'))
    # show_frame(FLAGS, yolo, class_names, cap, root, lmain, out)

    root.mainloop()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
