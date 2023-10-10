import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
from PIL import Image, ImageTk

import tkinter.messagebox as tm

import Detect as dt


bgcolor="#ffe6e6"
bgcolor1="#e60000"
fgcolor="#660000"


def Home():
        global window
        
        window = tk.Tk()
        window.title("MONITORING WINDOW")

 
        window.geometry('1280x720')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="Real Time Monitoring of Coma Patient using OpenCV" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
        message1.place(x=100, y=20)

        

       
        def monitorprocess():
                dt.process()
                
        browse = tk.Button(window, text="Start", command=monitorprocess  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse.place(x=550, y=350)

        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=1060, y=600)

        window.mainloop()
Home()

