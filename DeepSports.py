import time
import array as arr
import numpy as np
import ctypes
import datetime
import matplotlib.pyplot as plt
import imutils as imutils
from PIL import Image,ImageTk
from absl import app, flags, logging
from absl.flags import FLAGS
import tkinter as tk
from tkinter.filedialog import askopenfilename
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
flags.DEFINE_string('video', './data/flamengo.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', 'output.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')
#flags.DEFINE_string('output1', 'output', 'path to output directory to store snapshots')


#global yolo
#global class_names
running = True


mouseX = 0
mouseY = 0

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

width_screen=user32.GetSystemMetrics(0)
height_screen=user32.GetSystemMetrics(1)



class FUTOTAL:

    def __init__(self, master, argv):
        super().__init__()
        self.master = master
        self.argv = argv
        self.pause = False
        self.zoom=100
        self.selectFrame=0
        numframes = 0
        self.numframes = numframes

        # Variaveis para selecionar jogador
        self.selectON=False
        self.selecteds=arr.array('i', [])
        self.initArray(self.selecteds)
        self.objects_positions_id = arr.array('i', [])
        self.objects_positions_x = arr.array('i', [])
        self.objects_positions_y = arr.array('i', [])
        self.objects_positions_x_min = arr.array('i', [])
        self.objects_positions_y_min = arr.array('i', [])
        self.initArray( self.objects_positions_id)
        self.initArray(self.objects_positions_x)
        self.initArray(self.objects_positions_y)
        self.initArray(self.objects_positions_x_min)
        self.initArray(self.objects_positions_y_min)

        # Variaveis para adicionar linhas
        self.LineON = False
        self.line_player1 = arr.array('i', [])
        self.line_player2 = arr.array('i', [])
        self.initArray(self.line_player1)
        self.initArray(self.line_player2)
        self.players_selecionados = 0
        self.line_dropON = False


        # Variaveis para o circulo
        self.circuloON = False

        # Variaveis para o quadrado
        self.quadradoON = False

        # Variaveis para o seta
        self.setaON = False
        self.frame_arrow_create = arr.array('i', [])
        self.coordinates_arrow_x_init = arr.array('i', [])
        self.coordinates_arrow_y_init = arr.array('i', [])
        self.coordinates_arrow_x_final = arr.array('i', [])
        self.coordinates_arrow_y_final = arr.array('i', [])
        self.initArray(self.frame_arrow_create)
        self.initArray(self.coordinates_arrow_x_init)
        self.initArray(self.coordinates_arrow_y_init)
        self.initArray(self.coordinates_arrow_x_final)
        self.initArray(self.coordinates_arrow_y_final)
        self.num_of_click_arrow = 0

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


        if FLAGS.tiny:
            self.yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            print(FLAGS.num_classes)
            self.yolo = YoloV3(classes=FLAGS.num_classes)

        self.yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')

        self.class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')

        # width, height = 800, 600

        # abrir o video
        #self.filename = askopenfilename(title="Select file", filetypes=(("MP4 files", "*.mp4"),
        #                                                      #          ("WMV files", "*.wmv"),
        #                                                                ("AVI files", "*.avi")))
        self.filename = './data/imageminicial.jpeg'
        self.createMenuTop()
        self.createMenuLeft()

        try:
            self.cap = cv2.VideoCapture(int(self.filename))
        except:
            self.cap = cv2.VideoCapture(self.filename)

        #NUMBER OF FRAMES
        property_id = int(cv2.CAP_PROP_FRAME_COUNT)
        self.length = int(cv2.VideoCapture.get(self.cap, property_id))

        self.out = None
        if FLAGS.output:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            self.out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
            # by default VideoCapture returns float instead of int
            list_file = open('detection.txt', 'w')
            frame_index = -1


        self.master.title("DeepSports - Sports Analysis Software")
        self.master.bind('<Escape>', lambda e: self.master.quit())
        self.lmain = tk.Label(self.master, width=width_screen-300, height=height_screen-225)
        self.lmain.pack()

        self.createMenuBottom()

        self.FLAGS = FLAGS

        self.show_frame()

    def donothing(self):
        filewin = tk.Toplevel(self.master)
        button = tk.Button(filewin, text="Do nothing button")
        button.pack()

    def initArray(self,array):
        if len(array) == 0:
            for x in range(100):
                array.insert(0,0)
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
        filemenu.add_command(label="Close", command=self.donothing)

        filemenu.add_command(label="Exit", command=self.master.quit)
        self.menuBar.add_cascade(label="File", menu=filemenu)
        editmenu = tk.Menu(self.menuBar, tearoff=0)
        editmenu.add_command(label="Scree shot", command=self.donothing)

        editmenu.add_separator()

        editmenu.add_command(label="Cut", command=self.donothing)
        editmenu.add_command(label="Copy", command=self.donothing)
        editmenu.add_command(label="Paste", command=self.donothing)
        editmenu.add_command(label="Delete", command=self.donothing)
        editmenu.add_command(label="Select All", command=self.donothing)

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

    def createMenuLeft(self):
        m1 = tk.PanedWindow()
        m1.pack(fill=tk.BOTH, expand=0, side=tk.LEFT)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

        m2 = tk.PanedWindow(m1, orient=tk.VERTICAL)
        m1.add(m2)



        # Creating a photoimage object to use image
        self.circle = tk.PhotoImage(file=r'./data/circle.png')
        self.line = tk.PhotoImage(file=r'./data/line.png')
        self.line_drop = tk.PhotoImage(file=r'./data/line_drop.png')
        self.quadrado = tk.PhotoImage(file=r'./data/quadrado.png')
        self.seta = tk.PhotoImage(file=r'./data/seta.png')
        self.icon = tk.PhotoImage(file=r'./data/icon.png')

        # here, image option is used to
        # set image on button
        self.Circle = tk.Button(m2,text="Draw Circle", image=self.circle,command=self.circuloONOFF,compound="top")
        self.Circle.pack(fill=tk.BOTH, side=tk.TOP)

        # here, image option is used to
        # set image on button
        self.Line = tk.Button(m2,text="Draw Line", image=self.line, command=self.lineONOFF,compound="top")
        self.Line.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Line, text='Click on the player where the line will start.\n'
                                      'Then click on the second player where the line will end.')

        self.Line_Drop = tk.Button(m2,text="Drop Line", image=self.line_drop, command=self.line_dropONOFF,compound="top")
        self.Line_Drop.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Line_Drop, text='Click on the player that contains the line.')

        self.Line_Drop_All = tk.Button(m2, text="Drop Line All", image=self.line_drop, command=self.line_drop_all,
                                   compound="top")
        self.Line_Drop_All.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Line_Drop_All, text='Click on this button and delete all the lines.')

        self.Quadrado = tk.Button(m2,text="Draw Area", image=self.quadrado, command=self.quadradoONOFF,compound="top")
        self.Quadrado.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Quadrado, text='Click on the exact place where the square starts.\n'
                                          'And then click on the point where it ends.')

        self.Seta = tk.Button(m2,text="Draw Movement", image=self.seta, command=self.setaONOFF,compound="top")
        self.Seta.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Seta, text='Click on the exact spot where the movement starts.\n'
                                      'And then click on the place where it ends.')

        self.Icon = tk.Button(m2,text="Select Player", image=self.icon,command=self.selectONOFF,compound="top")
        self.Icon.pack(fill=tk.BOTH, side=tk.TOP)
        CreateToolTip(self.Icon, text='Click on the player you want to select.\n'
                                      'Click again to deselect')

        # variavel que serve para guardar a cor original do butao
        self.orig_color = self.Icon.cget("background")



        def colorsButtons():
            if self.selectON == False:
                self.Icon.configure(bg=self.orig_color)
            if self.LineON == False:
                self.Line.configure(bg=self.orig_color)
            if self.circuloON == False:
                self.Circle.configure(bg=self.orig_color)
            if self.quadradoON == False:
                self.Quadrado.configure(bg=self.orig_color)
            if self.setaON == False:
                self.Seta.configure(bg=self.orig_color)
            if self.line_dropON == False:
                self.Line_Drop.configure(bg=self.orig_color)

            m2.after(1000, colorsButtons)
        colorsButtons()

    def line_drop_all(self):
        self.initArray(self.line_player1)
        self.initArray(self.line_player2)
        self.start()

    def createMenuBottom(self):
        m1 = tk.PanedWindow(orient=tk.VERTICAL)
        m1.pack(fill=tk.BOTH, expand=0,side=tk.BOTTOM)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

        m0 = tk.PanedWindow(m1, orient=tk.HORIZONTAL)
        m1.add(m0)

        self.w2 = tk.Scale(m0, from_=0, to=self.length,variable=self.selectFrame, orient=tk.HORIZONTAL, command=self.selectFrameScale)
        self.w2.pack(fill=tk.BOTH, side=tk.LEFT,expand=1)

        m2 = tk.PanedWindow(m1, orient=tk.HORIZONTAL)
        m1.add(m2)

        var = tk.StringVar()
        labelframes = tk.Label(m2, textvariable=var, relief=tk.RAISED, borderwidth=0)
        var.set("Frames:" + str(self.numframes) + "/" + str(self.length))
        labelframes.pack(fill=tk.BOTH, side=tk.LEFT)

        def labelFrames():
            var.set("Frames:" + str(self.numframes) + "/" + str(self.length))
            labelframes.config(text=var)
            self.w2.set(self.numframes)
            m2.after(1000,labelFrames)

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

    def show_frame(self):
        global running
        _, frame = self.cap.read()
        # frame = cv2.flip(frame, 0)
        self.numframes=self.numframes+1
        print(self.numframes)
        frame = imutils.resize(frame, width=width_screen-300)

        print(self.pause)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Image.fromarray( obj , mode = None )
        # obj - Objeto com interface de matriz
        # mode - Modo a ser usado (será determinado a partir do tipo se None) Consulte:
        # img1=img
        # img1 = imutils.resize(img1, width=900)

        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_in = tf.expand_dims(img1, 0)
        img_in = transform_images(img_in, self.FLAGS.size)
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
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

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
        cont_objects_positions_x = 0
        cont_objects_positions_y = 0
        cont_objects_positions_x_min = 0
        cont_objects_positions_y_min = 0

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            # Se o id do track estiver no array de ids dos jogadores selecionados o rectagulo irá ser desenhado com uma cor diferente
            if self.contain(int(track.track_id)):
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), 2)
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                #              (int(bbox[0]) + (len(str(track.track_id))) * 5, int(bbox[1])),
                #              (255, 0, 255), -1)
                cv2.ellipse(frame, (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[3])), (20, 4), 0, 0, 360,
                            (255, 0, 255), 2, 15)

                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 1)
            else:
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255) , 2)
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                #              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])),  (0, 0, 255), -1)
                cv2.ellipse(frame, (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[3])), (20, 4), 0, 0, 360,
                            (100, 255, 100), 2, 15)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 1)

            # cada jogador selecionado irá ter o id registado no array de ids, a posicao do x no array dos x e a posicao do y no array do y
            # self.objects_positions_id.insert(cont_objects_positions_id,track.track_id)
            # self.objects_positions_x.insert(cont_objects_positions_x,int(bbox[0]))
            #self.objects_positions_y.insert(cont_objects_positions_y,int(bbox[1]))
            self.objects_positions_id[cont_objects_positions_id]=track.track_id
            self.objects_positions_x[cont_objects_positions_x] = int(bbox[0])
            self.objects_positions_y[cont_objects_positions_y] = int(bbox[1])
            self.objects_positions_x_min[cont_objects_positions_x_min] = int(bbox[2])
            self.objects_positions_y_min[cont_objects_positions_y_min] = int(bbox[3])

            # Incrementar contadores dos arrays para que cada posicao dos arrays coecidir com um jogador
            cont_objects_positions_id = cont_objects_positions_id+1
            cont_objects_positions_x = cont_objects_positions_x+1
            cont_objects_positions_y = cont_objects_positions_y+1
            cont_objects_positions_x_min = cont_objects_positions_x_min+1
            cont_objects_positions_y_min = cont_objects_positions_y_min+1

        # Criacao das multiplas linhas
        def arrayLenght(array):
            cont = len(array)-1
            while array[cont] == 0 and cont >= 0:
                cont = cont - 1
            return cont + 1


        cont_line_player1_id = 0
        cont_line_player2_id = 0
        if arrayLenght(self.line_player1)>0:
            print(arrayLenght(self.line_player1))
            while cont_line_player1_id<arrayLenght(self.line_player1):
                player1 = 0
                player2 = 0
                cont = 0

                for n in self.objects_positions_id:
                    if int(n) == int(self.line_player1[cont_line_player1_id]) and player1 == 0:
                        player1 = cont
                    if int(n) == int(self.line_player2[cont_line_player2_id]) and player2 == 0:
                        player2 = cont
                    cont = cont + 1

                if self.line_player1[cont_line_player1_id] == self.line_player2[cont_line_player2_id]:
                    x_new_player1 = self.objects_positions_x[player1] + ((self.objects_positions_x_min[player1] - self.objects_positions_x[player1])/2)
                    x_new_player2 = self.objects_positions_x[player2] + (
                                (self.objects_positions_x_min[player2] - self.objects_positions_x[player2]) / 2)


                    cv2.line(frame, (int(x_new_player1), self.objects_positions_y_min[player1]),
                            (int(x_new_player2), self.objects_positions_y_min[player2]), (0, 125, 255), 5)
                else:
                    x_new_player1 = self.objects_positions_x[player1] + (
                                (self.objects_positions_x_min[player1] - self.objects_positions_x[player1]) / 2)
                    x_new_player2 = self.objects_positions_x[player2] + (
                            (self.objects_positions_x_min[player2] - self.objects_positions_x[player2]) / 2)

                    cv2.line(frame, (int(x_new_player1), self.objects_positions_y_min[player1]),
                            (int(x_new_player2), self.objects_positions_y_min[player2]), (255, 255, 255), 5)
                cont_line_player1_id = cont_line_player1_id + 1
                cont_line_player2_id = cont_line_player2_id + 1

        # criacao das setas
        if self.frame_arrow_create[0] != 0:
            contador_setas = 0
            while contador_setas<arrayLenght(self.frame_arrow_create):
                start_point = (int(self.coordinates_arrow_x_init[contador_setas]),int(self.coordinates_arrow_y_init[contador_setas]))

                end_point = (int(self.coordinates_arrow_x_final[contador_setas]),
                             int(self.coordinates_arrow_y_final[contador_setas]))

                if  int(self.coordinates_arrow_x_final[contador_setas]) == 0 and int(self.coordinates_arrow_y_final[contador_setas]) == 0:
                    end_point = (int(self.coordinates_arrow_x_init[contador_setas]),
                                 int(self.coordinates_arrow_y_init[contador_setas]))
                color = (0, 255, 0)
                thickness = 2
                if int(self.numframes) - int(self.frame_arrow_create[contador_setas]) < 25:
                    cv2.arrowedLine(frame, start_point, end_point,
                                    color, thickness)
                contador_setas = contador_setas + 1

            # if FLAGS.output:
        # out.write(img)
        # cv2.imshow('output', img)

        # Zoom aplicado, o self.zoom irá decidir a escala aplicada no video
        scale_percent = self.zoom  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Cria uma memória de imagem de um objeto que exporta a interface da matriz
        imgtk = ImageTk.PhotoImage(
            image=img)  # usada para exibir imagens (em escala de cinza ou em cores verdadeiras) em rótulos, botões, telas e widgets de texto
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)

        key = cv2.waitKey(1)

        self.lmain.bind('<Leave>', self.exit_)
        self.lmain.bind('<Button-1>', self.motion) # quando alguem clica na tela de jogo o irá imediatamente assionar a funcao self.motion

        if not self.pause:
             self.lmain.after(5, self.show_frame)

        # after (pai, ms, função = Nenhum, * args)
        # Parâmetros:
        # parent : é o objeto do widget ou da janela principal, o que estiver usando esta função.
        # ms : é o tempo em milissegundos.
        # função : que deve ser chamada.
        # * args : outras opções.

    def contain(self,number):
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
            cont = len(array)-1
            while array[cont] == 0 and cont >= 0:
                cont = cont - 1
            return cont+1

        cont_objects_positions_id = 0
        cont_objects_positions_x = 0
        cont_objects_positions_y = 0

        while self.objects_positions_id[cont_objects_positions_id] > 0:
            if ((x > self.objects_positions_x[cont_objects_positions_x]) and (x < self.objects_positions_x_min[cont_objects_positions_x])):
                if ((y > self.objects_positions_y[cont_objects_positions_y]) and (y <self.objects_positions_y_min[cont_objects_positions_y])):
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
                        print("line 625")
                        print(self.line_player1)
                        print(self.line_player2)
                        print(arrayLenght(self.line_player1))
                        count = 0
                        if self.players_selecionados == 0:
                            while count < arrayLenght(self.line_player1):
                                count = count + 1
                            # adicionar jogador no array player1
                            self.line_player1[count] = self.objects_positions_id[cont_objects_positions_id]
                            self.line_player2[count] = self.objects_positions_id[cont_objects_positions_id]
                            self.players_selecionados = self.players_selecionados + 1
                        else:
                            while count < arrayLenght(self.line_player2):
                                count = count + 1
                                # adicionar jogador no array player1
                            self.line_player2[count - 1] = self.objects_positions_id[cont_objects_positions_id]
                            self.players_selecionados = self.players_selecionados + 1
                            self.players_selecionados = 0

                    if self.line_dropON == True:
                        count = 0
                        while self.line_player1[count] != self.objects_positions_id[
                            cont_objects_positions_id] and count < (arrayLenght(self.line_player1)):
                            count = count + 1
                        # remover jogador no array player1 e companheiro e companheiro do player1
                        if self.line_player1[count] == self.objects_positions_id[cont_objects_positions_id]:
                            self.line_player1[count] = 0
                            self.line_player2[count] = 0

                        # organizar array self.line_player1

                        #contador = 0
                        #while self.line_player1[contador] > 0:
                        #    contador = contador + 1
                        #while contador < (arrayLenght(self.line_player1) - 1):
                        #    self.line_player1[contador] = self.line_player1[contador + 1]
                        #    self.line_player2[contador] = self.line_player2[contador + 1]
                        #    contador = contador + 1

                        count = 0
                        while self.line_player2[count] != self.objects_positions_id[
                            cont_objects_positions_id] and count < (arrayLenght(self.line_player2)):
                            count = count + 1
                        # remover jogador no array player2 e companheiro do player1
                        if self.line_player2[count] == self.objects_positions_id[cont_objects_positions_id]:
                            self.line_player1[count] = 0
                            self.line_player2[count] = 0

                        print("602")
                        print(self.line_player1)
                        print(self.line_player2)
                        # organizar array self.line_player2
                        #contador = 0
                        #while self.line_player2[contador] > 0:
                        #    contador = contador + 1
                        #while contador < (arrayLenght(self.line_player2) - 1):
                        #    self.line_player1[contador] = self.line_player1[contador + 1]
                        #    self.line_player2[contador] = self.line_player2[contador + 1]
                        #    contador = contador + 1

            cont_objects_positions_id = cont_objects_positions_id + 1
            cont_objects_positions_x = cont_objects_positions_x + 1
            cont_objects_positions_y = cont_objects_positions_y + 1

        if self.setaON == True:
            if self.num_of_click_arrow == 0:
                num = int(arrayLenght(self.frame_arrow_create))
                self.frame_arrow_create[num] = self.numframes
                self.coordinates_arrow_x_init[num] = x
                self.coordinates_arrow_y_init[num] = y
                self.num_of_click_arrow = self.num_of_click_arrow + 1
            else:
                num = int(arrayLenght(self.coordinates_arrow_x_final))
                self.coordinates_arrow_x_final[num] = x
                self.coordinates_arrow_y_final[num] = y
                self.num_of_click_arrow = 0

        self.start()
        time.sleep(0.1)
        self.stop()

    def selectONOFF(self): # funcao que irá ativar a opcao selecionar
        if self.selectON==False:
            self.selectON = True
            self.Icon.configure(bg="gray")
            self.LineON = False
            self.quadradoON = False
            self.circuloON = False
            self.setaON = False
            self.line_dropON = False
        else:
            self.selectON=False
            self.start()

    def lineONOFF(self):
        if self.LineON == False:
            self.LineON = True
            self.Line.configure(bg="gray")
            self.selectON = False
            self.quadradoON = False
            self.circuloON = False
            self.setaON = False
            self.line_dropON = False
        else:
            self.LineON = False
            self.start()

    def line_dropONOFF(self):
        if self.line_dropON == False:
            self.line_dropON = True
            self.Line_Drop.configure(bg="gray")
            self.selectON = False
            self.quadradoON = False
            self.circuloON = False
            self.setaON = False
            self.LineON = False
        else:
            self.line_dropON = False
            self.start()

    def circuloONOFF(self):
        if self.circuloON == False:
            self.circuloON = True
            self.Circle.configure(bg="gray")
            self.selectON = False
            self.quadradoON = False
            self.LineON = False
            self.setaON = False
            self.line_dropON = False
        else:
            self.LineON = False
            self.start()

    def quadradoONOFF(self):
        if self.quadradoON == False:
            self.quadradoON = True
            self.Quadrado.configure(bg="gray")
            self.selectON = False
            self.circuloON = False
            self.LineON = False
            self.setaON = False
            self.line_dropON = False
        else:
            self.quadradoON = False
            self.start()

    def setaONOFF(self):
        if self.setaON == False:
            self.Seta.configure(bg="gray")
            self.setaON = True
            self.selectON = False
            self.circuloON = False
            self.LineON = False
            self.quadradoON = False
            self.line_dropON = False

        else:
            self.setaON = False
            self.start()

    def zoomin(self):
        if self.zoom<300:
            self.zoom=self.zoom+50
        self.start()

    def zoomout(self):
        if self.zoom>50:
            self.zoom = self.zoom - 50
        self.start()

    def enter(self,event):
        self.start()

    def exit_(self,event):
        self.pause = True

    def start(self):
        if self.pause == True:
            self.pause = False
            self.show_frame()

    def stop(self):
        self.pause = True

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
            self.numframes=0
            self.w2.config(to=self.length)

            self.clean_arrays()

        self.show_frame()

    def next(self):
        if (self.numframes+100)<self.length:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes+100)
            self.numframes=self.numframes+100

            self.clean_arrays()

        self.start()

    def selectFrameScale(self,v):
        print("572: clicar no scale no")
        print(v)
        if (int(v)-self.numframes) > 20:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(v))
            self.numframes = int(v)
            self.clean_arrays()
            self.start()
        if (self.numframes-int(v)) > 20:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(v))
            self.numframes = int(v)
            self.clean_arrays()
            self.start()

    def clean_arrays(self):
        self.initArray(self.line_player1)
        self.initArray(self.line_player2)

        self.initArray(self.selecteds)

        self.initArray(self.objects_positions_id)
        self.initArray(self.objects_positions_x)
        self.initArray(self.objects_positions_y)
        self.initArray(self.objects_positions_x_min)
        self.initArray(self.objects_positions_y_min)

        self.initArray(self.frame_arrow_create)
        self.initArray(self.coordinates_arrow_x_init)
        self.initArray(self.coordinates_arrow_y_init)
        self.initArray(self.coordinates_arrow_x_final)
        self.initArray(self.coordinates_arrow_y_final)

    def back(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.numframes-100)
        if (self.numframes-100)>0:
            self.numframes=self.numframes-100
            self.clean_arrays()
        self.start()

    def exit(self):
        print("destroy")
        self.master.destroy()

def main(_argv):

    root = tk.Tk()
    root.title("DeepSports - Sports Analysis Software")
    #var=str(width_screen-200) +"x"+ str(height_screen)
    #root.geometry(var)
    app = FUTOTAL(root,_argv)
    #show_frame(FLAGS, yolo, class_names, cap, root, lmain, out)
    root.mainloop()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


# motion old
'''

    def motion(self,event):
        min_x=0
        min_y=0
        x, y = event.x, event.y # x e y sao igualados ao x clicado no evento e ao y clicado no evento
        print('{}, {}'.format(x, y))

        if self.selectON : # caso a opcao de selecionar esteija ativa o program ira proceder a selecao
            cont_objects_positions_id = 0
            cont_objects_positions_x = 0
            cont_objects_positions_y = 0

            while self.objects_positions_id[cont_objects_positions_id] > 0:
                if ((x - self.objects_positions_x[cont_objects_positions_x])  < 40) & ((y - self.objects_positions_y[cont_objects_positions_y]) < 60):
                   if ((x - self.objects_positions_x[cont_objects_positions_x])  >  0) & ((y - self.objects_positions_y[cont_objects_positions_y]) > 0):
                        count=0
                        while self.selecteds[count]>0:
                            count=count+1
                        if self.contain(self.objects_positions_id[cont_objects_positions_id]) == False:
                            self.selecteds[count] = self.objects_positions_id[cont_objects_positions_id]
                        else:
                            count = 0
                            while self.selecteds[count] != self.objects_positions_id[cont_objects_positions_id]:
                                count = count + 1
                            self.selecteds[count]=0

                cont_objects_positions_id = cont_objects_positions_id+1
                cont_objects_positions_x = cont_objects_positions_x+1
                cont_objects_positions_y = cont_objects_positions_y+1

        def arrayLenght(array):
            cont = 0
            while array[cont]>0:
                cont = cont + 1
            return cont


        if self.LineON==True :
            cont_objects_positions_id = 0
            cont_objects_positions_x = 0
            cont_objects_positions_y = 0
            print("Numero de elementos no line_player1:")
            print(arrayLenght(self.line_player1))
            while self.objects_positions_id[cont_objects_positions_id] > 0:
                if ((x - self.objects_positions_x[cont_objects_positions_x]) < 40) & (
                        (y - self.objects_positions_y[cont_objects_positions_y]) < 60):
                    if ((x - self.objects_positions_x[cont_objects_positions_x]) > 0) & (
                            (y - self.objects_positions_y[cont_objects_positions_y]) > 0):

                        print("line 625")
                        print(self.line_player1)
                        print(self.line_player2)
                        count = 0
                        if self.players_selecionados == 0:
                            while self.line_player1[count] > 0:
                                count = count + 1
                            # adicionar jogador no array player1
                            self.line_player1[count] = self.objects_positions_id[cont_objects_positions_id]
                            self.line_player2[count] = self.objects_positions_id[cont_objects_positions_id]
                            self.players_selecionados=self.players_selecionados+1
                        else :
                            while self.line_player2[count] > 0:
                                count = count + 1
                                # adicionar jogador no array player1
                            self.line_player2[count-1] = self.objects_positions_id[cont_objects_positions_id]
                            self.players_selecionados = self.players_selecionados + 1
                            self.players_selecionados = 0

                cont_objects_positions_id = cont_objects_positions_id + 1
                cont_objects_positions_x = cont_objects_positions_x + 1
                cont_objects_positions_y = cont_objects_positions_y + 1



        if self.line_dropON == True:
            cont_objects_positions_id = 0
            cont_objects_positions_x = 0
            cont_objects_positions_y = 0
            print("Numero de elementos no line_player1:")
            print(arrayLenght(self.line_player1))
            while self.objects_positions_id[cont_objects_positions_id] > 0:
                if ((x - self.objects_positions_x[cont_objects_positions_x]) < 40) & (
                        (y - self.objects_positions_y[cont_objects_positions_y]) < 60):
                    if ((x - self.objects_positions_x[cont_objects_positions_x]) > 0) & (
                            (y - self.objects_positions_y[cont_objects_positions_y]) > 0):

                        count = 0
                        while self.line_player1[count] != self.objects_positions_id[cont_objects_positions_id] and count< (arrayLenght(self.line_player1)):
                            count = count + 1
                        # remover jogador no array player1 e companheiro e companheiro do player1
                        self.line_player1[count] = 0
                        self.line_player2[count] = 0

                        # organizar array self.line_player1

                        contador = 0
                        while self.line_player1[contador] > 0:
                            contador = contador + 1
                        while contador < (arrayLenght(self.line_player1)-1):
                            self.line_player1[contador]=self.line_player1[contador+1]
                            self.line_player2[contador]=self.line_player2[contador+1]
                            contador = contador+1


                        count=0
                        while self.line_player2[count] != self.objects_positions_id[cont_objects_positions_id] and count< (arrayLenght(self.line_player2)):
                            count = count + 1
                        # remover jogador no array player2 e companheiro do player1
                        self.line_player1[count] = 0
                        self.line_player2[count] = 0

                        # organizar array self.line_player2
                        contador = 0
                        while self.line_player2[contador] > 0:
                            contador = contador + 1
                        while contador < (arrayLenght(self.line_player2) - 1):
                            self.line_player1[contador] = self.line_player1[contador + 1]
                            self.line_player2[contador] = self.line_player2[contador + 1]
                            contador = contador + 1

                cont_objects_positions_id = cont_objects_positions_id + 1
                cont_objects_positions_x = cont_objects_positions_x + 1
                cont_objects_positions_y = cont_objects_positions_y + 1
        print("array das linhas")
        print(self.line_player1)
        print(self.line_player2)

        self.start()
        time.sleep(0.5)
        self.stop()
'''