from .inputs import ShadertoyChannel, ShadertoyChannelTexture
from .passes import ImageRenderPass
from .shadertoy import Shadertoy

__version__ = "0.1.0"
version_info = tuple(map(int, __version__.split(".")))
