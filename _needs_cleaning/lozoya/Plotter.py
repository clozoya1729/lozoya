import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    def __init__(self, style='bmh'):
        # Setting the font and style
        matplotlib.rcParams['font.serif'] = "Times New Roman"
        matplotlib.rcParams['font.family'] = "serif"
        self.font = {'fontname': 'Times New Roman'}
        plt.style.use(style)

    def Plot(self, figure, input_data, subplot, step_size=1, legend_labels_list=[], starting_year=1,
             x_axis_label='Time (Years)',
             y_axis_label='State', title='Monte Carlo Simulation of Item Value'):
        # Setting labels and style
        self.y_axis, self.step_size, self.starting_year, self.x_axis_label, self.y_axis_label, self.title = input_data, step_size, starting_year, x_axis_label, y_axis_label, title
        self.figure = figure
        self.size = 0
        self.line_styles = ['.--', '*--', 'x--', '^--']  # The plotter wraps around these line styles

        # Creating the plot
        plt.figure(self.figure)
        plt.subplot(subplot)

        # This for loop is plotting all the points in each list in the input_data list.
        for i in range(self.y_axis.__len__()):
            self.x_axis = []
            self.x_axis = np.arange(self.starting_year, self.starting_year + self.y_axis[0].__len__(), self.step_size)
            if legend_labels_list.__len__() != 0:
                plt.plot(self.x_axis, self.y_axis[i], self.line_styles[i % self.line_styles.__len__()],
                         label=legend_labels_list[i % legend_labels_list.__len__()], linewidth=0.5)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)  # Placing a legend beside the plot
            else:
                plt.plot(self.x_axis, self.y_axis[i], self.line_styles[i % self.line_styles.__len__()], linewidth=0.5)

        self.Generate_Graph()

    def Histogram(self, figure, input_data, x_axis_label='State', y_axis_label='Frequency',
                  title='Frequency Distribution'):
        # Setting labels
        self.y_axis, self.x_axis_label, self.y_axis_label, self.title = input_data, x_axis_label, y_axis_label, title
        self.figure = figure

        # Creating the figure
        plt.figure(self.figure)
        self.title = str('Frequency Distribution at Year ' + str(int(self.figure) - 1))

        # Creating the plot
        plt.hist(self.y_axis, bins=20, facecolor='pink')

        self.Generate_Graph()

    def Bar_Chart_3D(self, graph, y_input_data, z_input_data, iterations, x_axis_label='Time (Years)',
                     y_axis_label='State', z_axis_label='Frequency', title='Frequency Distribution'):
        # Setting labels
        self.y_axis, self.z_axis, self.x_axis_label, self.y_axis_label, self.z_axis_label, self.title = y_input_data, z_input_data, x_axis_label, y_axis_label, z_axis_label, title
        self.iterations = iterations
        self.title = 'Frequency Distribution'
        self.graph = graph

        #TODO
        xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        ypos = y_input_data
        zpos = np.zeros_like(xpos)

        # Construct arrays with the dimensions for the bars.
        dx = 0.05 * np.ones_like(zpos)
        dy = 0.05 * np.ones_like(zpos)
        dz = z_input_data

        # Creating the plot
        self.graph.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal')

        self.Generate_Graph_3D()

    def Generate_Graph(self):
        # Placing the axis labels and the title on the graph
        plt.xlabel(self.x_axis_label, **self.font)
        plt.ylabel(self.y_axis_label, **self.font)
        plt.title(self.title, **self.font)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

    def Generate_Graph_3D(self):
        # Placing the axis labels and the title on the graph
        self.bar_chart_3d.set_xlabel(self.x_axis_label)
        self.bar_chart_3d.set_ylabel(self.y_axis_label)
        self.bar_chart_3d.set_zlabel(self.z_axis_label)
        plt.title(self.title)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # This and the preceding line ensure a full screen display upon rendering

    def Generate_Subplot_3D(self, figure):
        # Creating the figure
        self.bar_chart_3d = plt.figure(figure).add_subplot(111, projection='3d')
        return self.bar_chart_3d

    def Show(self):
        plt.grid(True)
        plt.show()


'''x = Plotter()
x.Bar_Chart_3D(1, [0,5,2,3,4,5,
                  3,2,4,5,6,7,
                  6,3,7,8,3,2,
                  2,4,6,7,2,3], [1], 5)
x.Show()'''

