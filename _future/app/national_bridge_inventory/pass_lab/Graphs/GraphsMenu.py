"""
Author: Christian Lozoya, 2017
"""

from Interface.SubMenu import SubMenu
from Utilities.Constants import *
from Graphs.PlotGenerator import PlotGenerator

style.use("ggplot")

norm = True
yrs = [y for y in range(1992, 2017)]
f = Figure()


def animate(i):
    pass


class GraphsMenu(SubMenu):
    def populate(self, model):
        self.model = model
        self.sampler = model.sampler
        self.vS = self.model.vS
        self.populate_menubar()
        if True:
            pass
        else:
            from Interface.CreateWindow import FrameCanvas
            self.frame = FrameCanvas(self.top, self.title, f)

        self.cx = (1, len(self.model.yrs))
        self.cy = (0, len(self.vS))

        self.rLimsY = ((0, len(self.vS)), (0, 1) if norm == True else (0, len(self.model.yrs)))
        self.rLimsYH = ((0, len(self.vS)), (0, 1) if norm == True else (0, len(self.model.yrs)))

        self.rLimsS = ((1, 25), (0, 1) if norm == True else (0, len(self.model.data.T.columns)))
        self.rLimsSH = ((1, 26), (0, 1) if norm == True else (0, len(self.model.data.T.columns)))

        self.sLimsY = ((0, len(self.vS)), (0, 1) if norm == True else (0, int(self.model.iterations)))
        self.sLimsYH = ((0, len(self.vS)), (0, 1) if norm == True else (0, int(self.model.iterations)))

        self.sLimsS = ((1, 25), (0, 1) if norm == True else (0, int(self.model.iterations)))
        self.sLimsSH = ((1, 26), (0, 1) if norm == True else (0, int(self.model.iterations)))

        ani = animation.FuncAnimation(f, animate, interval=1000)
        self.top.mainloop()

    def populate_menubar(self):
        menubar = self.util.partial_drop(self.top, tearoff=0)
        filemenu = self.util.partial_drop(menubar, tearoff=0)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.cancel)
        menubar.add_cascade(label="File", menu=filemenu)

        trnData = (self.model.pdf.as_matrix().astype(np.float64))
        detData = (self.model.data.as_matrix().astype(np.float64), self.sampler.simulation)
        frqData = (0,)
        hstData = (0,)

        self.transition_probabilities_menu(menubar, data=trnData)
        self.deterioration_menu(menubar, data=detData)
        self.frequency_menu(menubar, data=frqData)
        self.histogram_menu(menubar, data=hstData)

        self.util.set_drop(self.top, menu=menubar)

    def transition_probabilities_menu(self, menubar, data):
        menubar.add_command(label=GRAPH_TITLES[0],
                            command=lambda: graph(data=data, pt=0, label=0, lims=((1, len(data[0])), (0, 1)),
                                                  t=0, c=0, legend=self.vS, fill=True))

    def deterioration_menu(self, menubar, data):
        deterioration_menu = self.util.partial_drop(menubar, tearoff=1)
        # Raw Deterioration
        deterioration_menu.add_command(label=GRAPH_TITLES[1],
                                       command=lambda: graph(data=data[0], pt=0, label=1, lims=(self.cx, self.cy),
                                                             t=1, c=1))

        # Simulation Deterioration
        deterioration_menu.add_command(label=GRAPH_TITLES[2],
                                       command=lambda: graph(data=data[1], pt=0, label=1, lims=(self.cx, self.cy),
                                                             t=2, c=2))

        menubar.add_cascade(label="Deterioration", menu=deterioration_menu)

    def frequency_menu(self, menubar, data):
        frequency_menu = self.util.partial_drop(menubar, tearoff=1)

        # Raw Per Year
        frequency_menu.add_command(label=GRAPH_TITLES[3],
                                   command=lambda: graph(data=self.model.raw_frq("year"), pt=0, label=3,
                                                         lims=self.rLimsY, t=3, c=1, legend=yrs, fill=True))

        # Simulation Per Year
        frequency_menu.add_command(label=GRAPH_TITLES[4],
                                   command=lambda: graph(data=self.sampler.sim_frq("year"), pt=0, label=3,
                                                         lims=self.sLimsY, t=4, c=2, legend=yrs, fill=True))

        # Raw Per State
        frequency_menu.add_command(label=GRAPH_TITLES[5],
                                   command=lambda: graph(data=self.model.raw_frq("state"), pt=0, label=2,
                                                         lims=self.rLimsS, t=5, c=0, legend=self.vS, fill=True))

        # Simulation Per State
        frequency_menu.add_command(label=GRAPH_TITLES[6],
                                   command=lambda: graph(data=self.sampler.sim_frq("state"), pt=0, label=2,
                                                         lims=self.sLimsS, t=6, c=0, legend=self.vS, fill=True))

        menubar.add_cascade(label="Frequency", menu=frequency_menu)

    def histogram_menu(self, menubar, data):
        histogram_menu = self.util.partial_drop(menubar, tearoff=1)

        # Raw Per Year
        histogram_menu.add_command(label=GRAPH_TITLES[3],
                                   command=lambda: graph(data=self.model.raw_hst("year"), pt=1, label=3,
                                                         lims=self.rLimsYH, t=3, c=1, legend=yrs))

        # Simulation Per Year
        histogram_menu.add_command(label=GRAPH_TITLES[4],
                                   command=lambda: graph(data=self.sampler.sim_hst("year"), pt=1, label=3,
                                                         lims=self.sLimsYH, t=4, c=2, legend=yrs))

        # Raw Per State
        histogram_menu.add_command(label=GRAPH_TITLES[5],
                                   command=lambda: graph(data=self.model.raw_hst("state"), pt=1, label=2,
                                                         lims=self.rLimsSH, t=5, c=0, legend=self.vS))

        # Simulation Per State
        histogram_menu.add_command(label=GRAPH_TITLES[6],
                                   command=lambda: graph(data=self.sampler.sim_hst("state"), pt=1, label=2,
                                                         lims=self.sLimsSH, t=6, c=0, legend=self.vS))

        menubar.add_cascade(label="Histogram", menu=histogram_menu)


def graph(data, pt, label, lims, t=0, c=0, legend=None, fill=False):
    plot = plt.subplot2grid((6, 4), (0, 0), rowspan=5, colspan=4)
    # Line Plot
    if pt == 0:
        for index, curve in enumerate(data): PlotGenerator.plot_curve(plot, index, curve, c, lims, fill)
    # Histogram
    elif pt == 1:
        for index, curve in enumerate(data): PlotGenerator.plot_histogram(plot, index, curve, color=c, lims=lims,
                                                                          norm=norm)
    PlotGenerator.plot_configure(plot, legend, GRAPH_TITLES[t], AXIS_LABELS[label], lims)
