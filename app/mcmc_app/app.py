import configuration
import lozoya.gui


class App(lozoya.gui.TSApp):
    def __init__(self, name, root):
        lozoya.gui.TSApp.__init__(self, name, root)


App(
    name=configuration.name,
    root=configuration.root,
).exec()
