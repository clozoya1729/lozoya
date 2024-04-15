from tkinter import *
from tkinter import filedialog as tkf
from textParser import textParser

iterations = 0
iFile = open('instructions.txt', 'w')
fname = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))

iFile.write(fname + "\n")


def setIter(iter):
    global iterations
    iterations = iter

def getIter():
    return iterations



def makeentry(parent, caption, width=None, case=0, **options):
    def callback():
        if not (case ==  0):
            point = entry.get()
            master.destroy()
            iFile.write(point + "\n")

        else:
            iter = entry.get()
            master.destroy()
            iFile.write(iter + "\n")
            setIter(int(iter))

    label = Label(parent, text=caption)
    label.pack(side=TOP)
    entry = Entry(parent, **options)
    entry.pack(side=LEFT)
    entry.focus_set()

    b = Button(master, text="Enter", width=10, command=callback)
    b.pack(side=RIGHT)

    mainloop()

master = Tk()
enterIter = makeentry(master, "Number of Extractions:", 10, 0)

for i in range(int(getIter())):
    master = Tk()
    enterStart = makeentry(master, "Enter Starting Point:", 10, 1)
    master = Tk()
    enterEnd = makeentry(master, "Enter Ending Point:", 10, 1)

iFile.close()

tP = textParser()