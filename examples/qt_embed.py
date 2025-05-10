# test_example = false

# based on the example from Rendercanvas/wgpu-py: https://github.com/pygfx/rendercanvas/blob/main/examples/qt_app.py

"""
An example demonstrating a qt app with a Shadertoy inside.
"""
import importlib
import time

# Normally you'd just write e.g.
# from PySide6 import QtWidgets

# For the sake of making this example Just Work, we try multiple QT libs
for lib in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
    try:
        QtWidgets = importlib.import_module(".QtWidgets", lib)
        break
    except ModuleNotFoundError:
        pass

from rendercanvas.qt import QRenderWidget

from wgpu_shadertoy import Shadertoy

class ExampleWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle("Rendering to a canvas embedded in a qt app")

        splitter = QtWidgets.QSplitter()

        # TODO: make this button pause/resume the shader (can you suspend the loop?)
        self.button = QtWidgets.QPushButton("Hello world", self)
        self.canvas = QRenderWidget(splitter, update_mode="continuous")
        self.output = QtWidgets.QTextEdit(splitter)

        self.button.clicked.connect(self.whenButtonClicked)

        splitter.addWidget(self.canvas)
        splitter.addWidget(self.output)
        # these are w
        splitter.setSizes([300, 400])

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.button, 0)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        self.show()

    def addLine(self, line):
        t = self.output.toPlainText()
        t += "\n" + line
        self.output.setPlainText(t)

    def whenButtonClicked(self):
        self.addLine(f"Clicked at {time.time():0.1f}")

# TODO: something more interesting
new_shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}
"""

app = QtWidgets.QApplication([])
example = ExampleWidget()

shader = Shadertoy(new_shader_code, resolution=(400,300), canvas=example.canvas)
example.canvas.request_draw(shader._draw_frame)
# the event loop is now handled by the QT app, so the shader.run() isn't needed anymore.

# Enter Qt event-loop (compatible with qt5/qt6)
app.exec() if hasattr(app, "exec") else app.exec_()