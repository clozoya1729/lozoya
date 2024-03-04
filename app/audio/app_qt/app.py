import configuration
import lozoya.gui


def callback0():
    print('callback0')


def callback1():
    print('callback1')


app = lozoya.gui.TSApp(
    name=configuration.name,
    root=configuration.root,
)
form = lozoya.gui.TSForm()
form.add_label(lozoya.gui.TSLabel(name='audio'))
form.add_fields(
    [
        lozoya.gui.TSInputButton(name='field2', text='button'),
        lozoya.gui.TSInputCheckbox(name='field3'),
    ]
)
app.window.setCentralWidget(form)
app.exec()
