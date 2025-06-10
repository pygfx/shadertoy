# test_example = false

# based on the example from Rendercanvas/wgpu-py: https://github.com/pygfx/rendercanvas/blob/main/examples/qt_app.py https://github.com/pygfx/wgpu-py/blob/main/examples/gui_qt_embed.py

"""
An example demonstrating a qt app with a Shadertoy inside. And some button interactivity
"""
import importlib
import time
from wgpu_shadertoy import Shadertoy

# Normally you'd just write e.g.
# from PySide6 import QtWidgets

# For the sake of making this example Just Work, we try multiple QT libs
for lib in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
    try:
        QtWidgets = importlib.import_module(".QtWidgets", lib)
        break
    except ModuleNotFoundError:
        pass

from rendercanvas.qt import QRenderWidget  # noqa: E402


# shadertoy source: https://www.shadertoy.com/new by iq?
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
""".lstrip()


class ExampleWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1200, 600)
        self.setWindowTitle("Rendering to a canvas embedded in a qt app")

        # functional widgets
        self.pause_button = QtWidgets.QPushButton("Toggle pause", self)
        self.rewind_button = QtWidgets.QPushButton("Rewind", self)
        self.load_button = QtWidgets.QPushButton("Update shader", self)
        self.canvas = QRenderWidget(self, update_mode="continuous")
        self.text = QtWidgets.QTextEdit(self, text=new_shader_code)

        # button functionality
        self.pause_button.clicked.connect(self.pause_button_click)
        self.rewind_button.clicked.connect(self.rewind_button_click)
        self.load_button.clicked.connect(self.load_button_click)

        # button layout below the canvas
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.rewind_button)

        # full layout with text on the right side
        layout = QtWidgets.QGridLayout()
        layout.setColumnMinimumWidth(0, 800) #TODO: figure out how the keep the canvas 16:9
        layout.addLayout(button_layout, 1, 0)  # buttons bottom left
        layout.addWidget(self.canvas, 0, 0)  # canvas on the left side
        layout.addWidget(self.text, 0, 1)  # text on the right side
        layout.addWidget(self.load_button, 1, 1)  # load button below the text
        self.setLayout(layout)

        self.show()
        self._paused = False
        # load the initial shader
        self.load_shader(new_shader_code)

    def pause_button_click(self):
        # showcases how rendercanvas allows changes to sheduling interactively
        if self._paused:
            delattr(self.shader, "_last_time") # hack from a while ago -to avoid jumps in time
            # will be improved in the future when pausing is an actual feature.
            self.canvas.set_update_mode("continuous", max_fps=60)
            self.pause_button.setText("Pause")
            self._paused = False
        else:
            # with "manual", we only get a redraw when we call it specifically... can break resizing!
            self.canvas.set_update_mode("manual")
            self.pause_button.setText("Resume")
            self._paused = True

    def rewind_button_click(self):
        # just setting the time and frame numbers to 0 might do it...
        
        self.shader._uniform_data["time"] = 0.0
        self.shader._uniform_data["frame"] = 0
        if self._paused:
            # in case we are paused(manual) we request a draw here
            self.canvas.force_draw()

    def load_button_click(self):
        # load the shader from the text edit
        shader_code = self.text.toPlainText()
        self.load_shader(shader_code)

    def load_shader(self, shader_code):
        # the QRenderWidget is a rendercanvas and can therefore be passed to the canvas kwarg
        self.shader = Shadertoy(shader_code, canvas=self.canvas)
        # the event loop is now handled by the QT app, so the shader.run() isn't needed anymore.
        self.canvas.request_draw(self.shader._draw_frame)

app = QtWidgets.QApplication([])
example = ExampleWidget()


# Enter Qt event-loop (compatible with qt5/qt6)
app.exec() if hasattr(app, "exec") else app.exec_()