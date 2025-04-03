from .inputs import ShadertoyChannel, ShadertoyChannelTexture, ShadertoyChannelBuffer
from .passes import ImageRenderPass, BufferRenderPass
from .shadertoy import Shadertoy

__version__ = "0.1.0"
version_info = tuple(map(int, __version__.split("."))) #noqa
