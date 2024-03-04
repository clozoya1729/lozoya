import configuration
import lozoya.gui

app = lozoya.gui_api.TSApp(
    name=configuration.name,
    root=configuration.root,
)
app.exec()

# import datetime
# import tkinter as tk
# import tkinter.ttk as ttk
#
# from Interface.InterfaceUtilities.CreateWindow import Window
# from Interface.Populators import PopulateGradationMenu
#
# current_year = datetime.datetime.now().year - 1
#
#
# class GradationMenu(Window):
#     def c_lo_run(self):
#         self.files, self.filename = 'single', 'report'
#         self.populator = PopulateGradationMenu.PopulateGradationMenu(self)
#
#     def help(self):
#         popup = tk.Tk()
#         popup.wm_title("!")
#         label = ttk.Label(popup, text="Christian Lozoya\n2017")
#         label.pack(side="top", fill="x", pady=10)
#         B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
#         B1.pack()
#         B1.mainloop()
#
#
# #   github, make it work on a mac, radial search,
#
# class GUI():
#     def __init__(self):
#         pass
#
#
# class Gradation():
#     def __init__(self):
#         self.percentagesVector = {}
#         self.args = []
#
#     def setPercentagesVector(self, *args):
#         for arg in args:
#             self.args.append(arg)
#
#         for i in range(10):
#             self.percentagesVector["Bin " + str(i + 1)] = args[i]
#
#         print(self.percentagesVector)
#
#
# gradation_1 = Gradation()
# percentagesInput = []
#
# for i in range(10):
#     pI = input("Bin " + str(i + 1) + " percent:")
#     percentagesInput.append(pI)
#
# gradation_1.setPercentagesVector(*percentagesInput)
