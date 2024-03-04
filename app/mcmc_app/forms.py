class MarkovChainMenu:
    def __init__(self, GUI, parent, MASTERPATH):
        top = self.top = Toplevel(parent)
        self.parent = parent
        self.top.title("Markov Chain Monte Carlo")
        self.settings = [True, False]
        self.MASTERPATH = MASTERPATH
        self.figure = 1
        self.GUI = GUI
        self.paths = 1
        self.GUI.listGenerator.setNonNoneValueList()
        # Create a Tkinter variable to store the ITEM OF INTEREST (item)
        self.item = StringVar(self.top)
        self.initial_state = StringVar(self.top)
        # Dictionary with options
        self.valid_item_of_interest_names = self.GUI.listGenerator.getNonNoneName()
        self.valid_item_of_interest_numbers = self.GUI.listGenerator.getNonNoneNumber()
        self.item.set('Record Type')  # set the default option
        self.initial_state.set('None')
        # Loading user preferences
        with open(
            MASTERPATH + "\BridgeDataQuery\\Utilities14\Preferences\Markov_Chain_Preferences.txt",
            "r"
        ) as report_preferences:
            self.settings = [True if line.strip() == 'True' else False for line in report_preferences]
        # FILE PATHS
        label1 = Label(top, text="Input File Paths:")
        label1.grid(row=1, column=1, sticky=W)
        self.input_file_paths = Entry(top, width=30)
        self.input_file_paths.grid(row=1, column=3, columnspan=2)
        # PATH SELECTION BUTTON
        self.path_selection = Button(top, compound=LEFT, text="Open", command=self.load, cursor='hand2')
        self.path_selection.grid(row=1, column=5, columnspan=2)
        # ITEM OF INTEREST
        label2 = Label(top, text="Item of Interest:")
        label2.grid(row=self.paths + 1, column=1, sticky=W)
        popupMenu1 = OptionMenu(self.top, self.item, *self.valid_item_of_interest_names, command=self.itemSelection)
        popupMenu1.grid(row=self.paths + 1, column=2, columnspan=3, sticky=W + E)
        # INITIAL STATE
        label3 = Label(top, text="Initial State:")
        label3.grid(row=self.paths + 2, column=1, sticky=W)
        self.itemSelection('Record Type')
        # ITERATIONS
        label3 = Label(top, text="Iterations:")
        label3.grid(row=self.paths + 3, column=1, sticky=W)
        self.iterations = Entry(top, width=30)
        self.iterations.grid(row=self.paths + 3, column=3, columnspan=2)
        # SEPARATOR
        separator = ttk.Separator(top, orient=HORIZONTAL)
        separator.grid(row=self.paths + 5, column=1, columnspan=4, sticky=W + E)
        run_button = Button(top, text="Run", command=self.run, cursor='hand2')
        run_button.grid(row=self.paths + 6, column=4, sticky=E)
        cancel_button = Button(top, text="Cancel", command=self.cancel, cursor='hand2')
        cancel_button.grid(row=self.paths + 6, column=5, sticky=E)
        self.top.grab_set()

    def itemSelection(self, value):
        self.parameterNamesDictionary = self.GUI.listGenerator.getParameterNamesWithRespectToNumber()
        self.GUI.listGenerator.allowable_list_generator(self.parameterNamesDictionary[str(self.item.get())])
        self.valid_states = self.GUI.listGenerator.get_allowable_values_list()
        self.initial_state.set(self.valid_states[0])
        popupMenu2 = OptionMenu(self.top, self.initial_state, *self.valid_states)
        popupMenu2.grid(row=self.paths + 2, column=2, columnspan=3, sticky=W + E)

    def load(self):
        path = tkf.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        self.input_file_paths.delete(0, END)
        self.input_file_paths.insert(0, path)

    def toggle_csv(self):
        if self.settings[0]:
            self.settings[0] = False
        else:
            self.settings[0] = True

    def toggle_machine(self):
        if self.settings[1]:
            self.settings[1] = False
        else:
            self.settings[1] = True

    def run(self):
        self.settings[:] = []
        self.input_file_paths_list = []  # A list containing the file paths to be iterated over
        self.input_file_paths_list = [str(file) for file in re.findall(r'.+(?!,),?', self.input_file_paths.get())]
        self.transition_matrix_object = TransitionMatrix.TransitionMatrixBuilder(
            self.input_file_paths_list,
            self.parameterNamesDictionary[
                self.item.get()],
            [str(year) for year in
             range(1992, 2017)]
        )
        # The matrix of probability of transitioning between states (dictionary of dictionaries)
        self.transition_matrix = self.transition_matrix_object.get_Transition_Matrix()
        self.initial_state_vector = {}
        self.list_of_possible_states = self.transition_matrix_object.get_List_of_Possible_States()
        # The initial position of the probabilities (dictionary)
        self.initial_state_vector = {state: 1 if state == self.initial_state.get() else 0 for state in
                                     self.list_of_possible_states}

        # print("Initial State " + str(self.initial_state.get()))
        # print(self.initial_state_vector)
        MarkovChain.MarkovChain(self)

    def cancel(self):
        self.settings[:] = []
        self.top.destroy()

    def getInitialState(self):
        return {state: 1 if state == self.initial_state.get() else 0 for state in self.list_of_possible_states}

    def __init__(self, GUI, parent, MASTERPATH):
        top = self.top = Toplevel(parent)
        self.parent = parent
        self.top.title("Markov Chain Monte Carlo")
        self.settings = [True, False]
        self.MASTERPATH = MASTERPATH
        self.figure = 1
        self.GUI = GUI
        self.paths = 1
        self.GUI.listGenerator.setNonNoneValueList()
        # Create a Tkinter variable to store the ITEM OF INTEREST (item)
        self.item = StringVar(self.top)
        self.initial_state = StringVar(self.top)
        # Dictionary with options
        self.valid_item_of_interest_names = self.GUI.listGenerator.getNonNoneName()
        self.valid_item_of_interest_numbers = self.GUI.listGenerator.getNonNoneNumber()
        self.item.set('Record Type')  # set the default option
        self.initial_state.set('None')
        # Loading user preferences
        with open(
            MASTERPATH + "\BridgeDataQuery\\Utilities1\Preferences\Markov_Chain_Preferences.txt",
            "r"
        ) as report_preferences:
            self.settings = [True if line.strip() == 'True' else False for line in report_preferences]
        # FILE PATHS
        label1 = Label(top, text="Input File Paths:")
        label1.grid(row=1, column=1, sticky=W)
        self.input_file_paths = Entry(top, width=30)
        self.input_file_paths.grid(row=1, column=3, columnspan=2)
        # PATH SELECTION BUTTON
        self.path_selection = Button(top, compound=LEFT, text="Open", command=self.load, cursor='hand2')
        self.path_selection.grid(row=1, column=5, columnspan=2)
        # ITEM OF INTEREST
        label2 = Label(top, text="Item of Interest:")
        label2.grid(row=self.paths + 1, column=1, sticky=W)
        popupMenu1 = OptionMenu(self.top, self.item, *self.valid_item_of_interest_names, command=self.itemSelection)
        popupMenu1.grid(row=self.paths + 1, column=2, columnspan=3, sticky=W + E)
        # INITIAL STATE
        label3 = Label(top, text="Initial State:")
        label3.grid(row=self.paths + 2, column=1, sticky=W)
        self.itemSelection('Record Type')
        # ITERATIONS
        label3 = Label(top, text="Iterations:")
        label3.grid(row=self.paths + 3, column=1, sticky=W)
        self.iterations = Entry(top, width=30)
        self.iterations.grid(row=self.paths + 3, column=3, columnspan=2)
        # SEPARATOR
        separator = ttk.Separator(top, orient=HORIZONTAL)
        separator.grid(row=self.paths + 5, column=1, columnspan=4, sticky=W + E)
        run_button = Button(top, text="Run", command=self.run, cursor='hand2')
        run_button.grid(row=self.paths + 6, column=4, sticky=E)
        cancel_button = Button(top, text="Cancel", command=self.cancel, cursor='hand2')
        cancel_button.grid(row=self.paths + 6, column=5, sticky=E)
        self.top.grab_set()
