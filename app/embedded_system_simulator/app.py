import lozoya.gui
import lozoya.embedded
import configuration


class App(lozoya.gui.TSApp):
    def __init__(self, name, root):
        lozoya.gui.TSApp.__init__(self, name, root)


app = App(
    name=configuration.name,
    root=configuration.root,
).exec()
