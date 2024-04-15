"""
Author: Christian Lozoya, 2017
"""

import numpy as np
from Utilities.Constants import *


class PlotGenerator:
    @staticmethod
    def plot_curve(plt, index, curve, color, lims, fill):
        plt.plot([x for x in range(lims[0][0], lims[0][0] + lims[0][1])], curve,
                 GRAPH_COLORS[color][index % len(GRAPH_COLORS[color])],
                 marker=GRAPH_MARKERS[index % len(GRAPH_MARKERS)], markersize=2,
                 fillstyle=GRAPH_FILL_STYLES[index % len(GRAPH_FILL_STYLES)],
                 dashes=GRAPH_DASHES[index % len(GRAPH_DASHES)])
        if fill:
            plt.fill_between([x for x in range(lims[0][0], lims[0][0] + lims[0][1])], 0, curve,
                             facecolors=GRAPH_COLORS[color][index % len(GRAPH_COLORS[color])], alpha=0.5)

    @staticmethod
    def plot_histogram(plt, index, curve, color, lims, norm=False):
        plt.hist(curve, alpha=0.5, bins=np.arange(lims[0][1] + 1), normed=norm, align='left',
                 facecolor=GRAPH_COLORS[color][index % len(GRAPH_COLORS[color])])

    @staticmethod
    def plot_configure(plt, legend, title, labels, lims):
        if legend is not None:
            plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)

        plt.set_title(title)
        plt.set_xlabel(labels[0])
        plt.set_ylabel(labels[1])
        if lims:
            plt.set_xlim(lims[0])
            plt.set_ylim(lims[1])
            plt.axes.set_xticks([v for v in range(lims[0][0], lims[0][1] + 1)])


"""
            # Setting the font and style
            matplotlib.rcParams['font.serif'] = "Times New Roman"
            matplotlib.rcParams['font.family'] = "serif"
            self.font = {'fontname': 'Times New Roman'}
            plt.style.use(style)
    
        Bar_Chart_3D
            #TODO
            xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            ypos = y_input_data
            zpos = np.zeros_like(xpos)
            # Construct arrays with the dimensions for the bars.
            dx, dy = 0.05 * np.ones_like(zpos)
            dz = z_input_data
            # Creating the plot
            self.graph.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')
    
        generate_graph
        # Placing the axis labels and the title on the graph
            plt.xlabel(self.x_label, **self.font)
            plt.ylabel(self.y_label, **self.font)
    
        generage_graph_3d
            # Placing the axis labels and the title on the graph
            self.bar_chart_3d.set_xlabel(self.x_label)
            self.bar_chart_3d.set_ylabel(self.y_label)
            self.bar_chart_3d.set_zlabel(self.z_label)
    
        generate_subplot_3d
        # Creating the figure
            self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
            return self.bar_chart_3d
"""
