import lozoya.gui


class TSFormData(lozoya.gui.TSForm):
    def __init__(self, callback0, callback1):
        self.dialogFile = lozoya.gui.TSFileDialog(name='Path selection')
        self.selectFile = lozoya.gui.TSInputButton(
            text='Select File',
            callback=callback0,
        )
        self.labelFile = lozoya.gui.TSLabel(
            'No file selected.'
        )
        self.selectX = lozoya.gui.TSInputCombo(
            options=None,
            default=None,
            callback=callback1,
            name='x',
        )
        self.selectY = lozoya.gui.TSInputCombo(
            options=None,
            default=None,
            callback=callback1,
            name='y',
        )
        fields = [
            self.selectFile,
            self.labelFile,
            self.selectX,
            self.selectY,
        ]
        lozoya.gui.TSForm.__init__(self, fields=fields)

    def get_selected_xy(self):
        x = self.selectX.get_value()
        y = self.selectY.get_value()
        return x, y

    def update_file(self):
        self.filePath = str(self.dialogFile.getOpenFileUrl(None)[0].toString())
        self.labelFile.setText(self.filePath)

    def update_xy_choices(self, options):
        self.selectX.update_options(options=options, default=0)
        self.selectY.update_options(options=options, default=1)


class TSFormFit(lozoya.gui.TSForm):
    def __init__(self, callback):
        self.button = lozoya.gui.TSInputButton(
            text='Fit',
            callback=callback,
        )
        fields = [
            self.button,
        ]
        lozoya.gui.TSForm.__init__(self, fields)
