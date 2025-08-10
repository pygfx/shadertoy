import os
import time
import av
import numpy as np
import subprocess
from tqdm.auto import tqdm

import wgpu

from wgpu_shadertoy import Shadertoy
from rendercanvas.auto import loop
from rendercanvas.base import BaseRenderCanvas, BaseCanvasGroup
# from rendercanvas.glfw import GlfwRenderCanvas

av.logging.set_level(av.logging.VERBOSE) # very useful as the errors mean something now!


# shadertoy source: https://www.shadertoy.com/view/7ds3zB by henryseg
# TODO find example with delta time and maybe accumulation to showoff faster/slower than realtime video export!
shader_code = """
// derived from BrunoLevy's RayTracing tutorial - step 1, shadertoy.com/view/wdfXR4
 
struct Camera {
    vec3 Obs;
    vec3 View;
    vec3 Up;
    vec3 Horiz;
    float H;
    float W;
    float z;
};

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

Camera camera(in vec3 Obs, in vec3 LookAt, in float aperture) {
   Camera C;
   C.Obs = Obs;
   C.View = normalize(LookAt - Obs);
   C.Horiz = normalize(cross(vec3(0.0, 0.0, 1.0), C.View));
   C.Up = cross(C.View, C.Horiz);
   C.W = float(iResolution.x);
   C.H = float(iResolution.y);
   C.z = (C.H/2.0) / tan((aperture * 3.1415 / 180.0) / 2.0);
   return C;
}

Ray launch(in Camera C, in vec2 XY) {
   return Ray(
      C.Obs,
      C.z*C.View+(XY.x-C.W/2.0)*C.Horiz+(XY.y-C.H/2.0)*C.Up 
   );
}

struct Sphere {
   vec3 Center;
   float R;
};

bool intersect_sphere(in Ray R, in Sphere S, out float t, out float t2) {
   vec3 CO = R.Origin - S.Center;
   float a = dot(R.Dir, R.Dir);
   float b = 2.0*dot(R.Dir, CO);
   float c = dot(CO, CO) - S.R*S.R;
   float delta = b*b - 4.0*a*c;
   if(delta < 0.0) {
      return false;
   }
   t = (-b-sqrt(delta)) / (2.0*a);
   t2 = (-b+sqrt(delta)) / (2.0*a);
   return true;
}

bool step_forward(in Ray R, inout float t, inout vec3 roundpoint, out int coord, in float max_t, in float cubeWidth, in float cubesRad){
    vec3 point = R.Origin + t*R.Dir;
    vec3 signDir = sign(R.Dir);
    /// solve for param: point + param*Dir = roundpoint + 0.5*(signDir)*cubeWidth component by component
    vec3 params = (roundpoint - point + 0.5*signDir*cubeWidth)/R.Dir;

    // find out which wall we hit next
    if(params.x < params.y){
        if(params.x < params.z){ coord = 0;}
        else{ coord = 2;}
    }
    else{
        if(params.y < params.z){ coord = 1;}
        else{ coord = 2;}
    }
    
    t += params[coord];
    vec3 move = vec3(0.0,0.0,0.0);
    move[coord] += cubeWidth;
    roundpoint += signDir*move;

    if(length(roundpoint) < cubesRad){ 
        roundpoint -= signDir*move; // take a step back to the cube before we hit the solid cube
        return false; } // we hit a cube 
    if(t > max_t){ coord = 3; 
        return false;
    }  // meaning that we were tangent to the sphere of cubes and didnt hit anything
 
    return true; // keep going
}

float mysmoothstep(in float x){
    float t = clamp(x, 0.0, 1.0);
    t = 1.0 - pow((1.0 - t),1.5); 
    return t * t * (3.0 - 2.0 * t);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
   float time = float(100 + iFrame)/100.0;
   float cubeWidth = 1.0/time;  
   //float cubeWidth = 0.2;
   float sphereRad = 1.0;
   float cubesRad = sphereRad - cubeWidth*0.5*sqrt(3.0); // cube with center within this rad is contained in sphere
 
   vec3 red = vec3(1.0,0.0,0.0);
   vec3 green = vec3(0.0,1.0,0.0);
   vec3 blue = vec3(0.0,0.0,1.0);
   
   float beta = 3.14159 / 4.0 + 0.2; // * time; // * time;
   float s = sin(beta);
   float c = cos(beta); 

   // Initialize the Camera 
   Camera C = camera(
       vec3(2.0*c, 2.0*s, 1.5),
       vec3(0.0, 0.0, 0.0),
       50.0       
   );
 
   Ray R = launch(C, fragCoord);
   Sphere S = Sphere(vec3(0.0, 0.0, 0.0), sphereRad);
   
   fragColor = vec4(0.5, 0.5, 0.5, 1.0);
   
   float t;
   float max_t;
   int coord;
   
   vec3 col = vec3(0.0,0.0,0.0);
   if(intersect_sphere(R,S,t,max_t)) {
      vec3 point = R.Origin + t*R.Dir;
      vec3 roundpoint = round(point/cubeWidth)*cubeWidth;
      
      bool cont = true;
      for(int i = 0; i <= 50; i++) { 
          cont = step_forward(R, t, roundpoint, coord, max_t, cubeWidth, cubesRad);
          if(cont == false){break;}
      }
      if(coord <= 2){ // hit a solid cube
          col[coord] = 1.0;
          
          vec3 point = R.Origin + t*R.Dir;
          vec3 signOctant = -sign(point);
          point -= roundpoint; // now relative to the cube center we just hit
          point *= signOctant; // now moving in the positive direction sends us to potential solid cubes
          point /= cubeWidth; // now in (-0.5,0.5)^2
          
          float brightness = 1.0;
          float d = 0.3; // ambient occlusion darkness amount
          
          vec3 move1 = vec3(0.0,0.0,0.0);
          move1[(coord + 1) % 3] += cubeWidth;
          vec3 neighbour1 = signOctant*roundpoint + move1;
          vec3 move2 = vec3(0.0,0.0,0.0);
          move2[(coord + 2) % 3] += cubeWidth;
          vec3 neighbour2 = signOctant*roundpoint + move2;
          vec3 move3 = move1 + move2;
          vec3 neighbour3 = signOctant*roundpoint + move3;
          bool n1solid = (length(neighbour1) < cubesRad);
          bool n2solid = (length(neighbour2) < cubesRad);
          bool n3solid = (length(neighbour3) < cubesRad);
          
          if(n1solid){
              brightness *= (1.0 - d) + d*mysmoothstep(0.5 - point[(coord + 1) % 3]);
          }
          if(n2solid){
              brightness *= (1.0 - d) + d*mysmoothstep(0.5 - point[(coord + 2) % 3]);
          }
          
          if(n3solid && (!n1solid && !n2solid)){
              float s1 = mysmoothstep(0.5 - point[(coord + 1) % 3]);
              float s2 = mysmoothstep(0.5 - point[(coord + 2) % 3]);
              float foo = 1.0 - (1.0 - s1)*(1.0 - s2);
              brightness *= (1.0 - d) + d*foo;
          }
          
          col *= brightness;          
          fragColor = vec4(col, 1.0);
      }
   }

}
 
 
"""



# naive offscreen implementation based on https://pyav.basswood-io.com/docs/stable/cookbook/numpy.html#generating-video
# TODO: should this be record_offscreen instead?
# for CLI usage it might be about this:
def record(shader: Shadertoy, output_file="output.mp4", **kwargs) -> None:
    # TODO: parameterize
    start_offset = kwargs.pop("start_offset", 0.0)
    duration = kwargs.pop("duration", 10.0)
    framerate = kwargs.pop("framerate", 60)
    mouse_pos = kwargs.pop("mouse_pos", (0.0, 0.0, 0.0, 0.0))
    target_size = kwargs.pop("target_size", 10.0) # in megabytes?
    bitrate = (target_size * 1000 * 1000 * 8) / duration # in bits per second not real megabytes for margin!?
    print(f"Recording {output_file} at {framerate} fps for {duration} seconds with bitrate {bitrate/1000:.2f} kbps")

    container = av.open(output_file, mode="w")
    # print(container.supported_codecs)
    stream: av.VideoStream = container.add_stream(
        "h264", # could be hardware specific like nvec or qsv etc... maybe use the device description to try a few and then fail?
        rate=framerate,
        width=shader.resolution[0],
        height=shader.resolution[1],
        pix_fmt="yuv420p", # for the output format, yuv4:2:0 is common for portanble video - but maybe we can get full 4:4:4 rgb instead for graphical details?
        bit_rate=bitrate, #900 kbps should be below 10MB for the 10 seconds duration
    )

    for frame_num in tqdm(range(int(duration * framerate)), desc="Recording", unit="frame"):
        timestamp = start_offset + frame_num / framerate
        time_delta = 1.0 / framerate
        # TODO: other uniforms, hint: https://github.com/Vipitis/shader_tracker/blob/f90fd5c3f28acbc88c23ccd7fe0c57ccf5778dda/capture.py
        frame_mem = shader.snapshot(
            time_float=timestamp, time_delta=time_delta, frame=frame_num, mouse_pos=mouse_pos
        )
        frame_arr = np.asarray(frame_mem, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(
            frame_arr, format="rgba"
        )  # format based on canvas._present_methods
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush the stream
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Recording finished: {output_file}")


class GLFWGrabber():
    """
    In theory this captures the gui while you can interact with it.
    """
    # TODO: why is the timestamp off, can we set it after the fact?
    def __init__(self, shader: Shadertoy, outfile: os.PathLike = "output_gui.mp4"):
        self.shader = shader
        self.canvas = shader._canvas

        # assert isinstance(self.canvas, GlfwRenderCanvas) # might break auto import...?
        info = self.canvas._rc_get_present_methods()
        hwdn = info["screen"]["window"]  # GLFW specific, might fail here on other backends!
        self.input: av.container.input.InputContainer = av.open(f"hwnd={hwdn}", format="gdigrab") #Windows specific!
        self.output: av.container.output.OutputContainer = av.open(outfile, mode="w")
        # TODO: add framerate? canvas.__sheduler._draw_stats?
        self.output_stream: av.VideoStream = self.output.add_stream(
            "h264",
            width=shader.resolution[0],
            height=shader.resolution[1],
            pix_fmt="yuv420p",
            bit_rate=900_000,
            rate=60,
        )
        self.canvas.request_draw(draw_function=self.draw_and_encode)
    
    def encode_last_frame(self):
        # just grab the "next" frame here?
        frame: av.VideoFrame = next(self.input.decode(video=0), None)
        # TODO: investigate seek

        if frame is not None:
            # TODO set .time or .pts or .dts to get the timestamps in order
            packets: list[av.Packet] = self.output_stream.encode(frame)
            for packet in packets:
                try:
                    self.output.mux(packet)
                except av.ValueError as e:
                    # ERROR is due to non monotonic DTS so we somehow need to ensure they are in sync.
                    # perhaps we can encode multiple frames here if needed or skip them?
                    # this throws errors, maybe due to logging?
                    pass

    def draw_and_encode(self):
        """
        meant as the new draw function
        """
        self.shader._draw_frame()
        self.encode_last_frame()

    def close(self):
        """
        Cleanup the input and output containers.
        """
        # TODO register to the close event?
        # does this actually work?? not sure as we got errors from here too.
        packets = self.output_stream.encode(None)
        for packet in packets:
            try:
                self.output.mux(packet)
            except av.ValueError as e:
                # this throws errors, maybe due to logging?
                pass
        self.output.close()
        self.input.close()


class LavfiCanvasGroup(BaseCanvasGroup):
    # needed?
    pass


class RecordingCanvas(BaseRenderCanvas):
    """
    Offscreen-like (or with ffplay as gui?) canvas to render to a video file or remote stream.
    """
    # https://rendercanvas.readthedocs.io/stable/backendapi.html

    _rc_canvas_group = LavfiCanvasGroup(loop) # this loop is from .auto!

    def __init__(self, outfile:str="canvas_output.mp4", *args, **kwargs):
        super().__init__(*args, **kwargs)
        res = kwargs.get("size", (800, 450)) # default size? messes up, as super has it's own defaults... and we can't access them from __kwargs_for_later?
        framerate = kwargs.get("max_fps", 60) #I think default might be 30...
        self._frame_counter = 0 #needed for pts?

        # TODO: container_kwargs?
        self._out_container = av.open(outfile, mode="w")
        # TODO: codec_kwargs?
        self._out_stream = self._out_container.add_stream(
            "h264",
            width=res[0],
            height=res[1],
            pix_fmt="yuv420p", # as this case is compressed... we throw out a lot of data!
            bit_rate=900_000, # this number might be widely wrong...
            rate=framerate,
        )

        self.gui_process = subprocess.Popen(
            [
                "ffplay",
                "-f", "rawvideo",
                "-pixel_format", "rgba",
                "-video_size", f"{res[0]}x{res[1]}",
                "-framerate", str(framerate),
                "-i", "pipe:"
            ],
            stdin=subprocess.PIPE
        )
        self._pipe_container = av.open(self.gui_process.stdin, format="rawvideo", mode="w")
        self._pipe_stream = self._pipe_container.add_stream(
            "rawvideo",
            width=res[0],
            height=res[1],
            pix_fmt="rgba",
            rate=framerate,
        )

        self._final_canvas_init() # must be called?

    def _rc_get_present_methods(self):
        # bare minimum I guess...
        return {
            "bitmap": {
                "formats": ["rgba-u8"],
            }
        }

    def _rc_request_draw(self):
        # as this should be continous we do have a loop
        loop = self._rc_canvas_group.get_loop()
        loop.call_soon(self._draw_frame_and_present)

    def _rc_get_physical_size(self) -> tuple[int, int]:
        return self._psize

    def _rc_get_pixel_ratio(self):
        return 1.0

    def _rc_get_logical_size(self) -> tuple[float, float]:
        return self._logical_size

    def _rc_set_logical_size(self, width, height):
        # gets called during _final_init_
        self._logical_size = width, height
        # ignores pixel aspect ratio currently.
        self._psize = int(width), int(height) # physical size needs to be in int!

    def _rc_close(self):
        self._out_container.close()
        self.gui_process.stdin.close()
        self.gui_process.terminate() # or kill?

    def _rc_get_closed(self):
        return_code = self.gui_process.poll()
        return return_code is not None

    def _rc_present_bitmap(self, *, data, format, **kwargs):
        # TODO: could this be directly from bytes or the memoryview?
        # could the texture be a frame already?
        frame = av.VideoFrame.from_ndarray(
            np.asanyarray(data), format="rgba"
        )
        # encode to file
        frame.pts = self._frame_counter
        for packet in self._out_stream.encode(frame):
            self._out_container.mux(packet)
        
        # write to the gui process
        for packet in self._pipe_stream.encode(frame):
            # we write the raw bytes to the stdin of ffplay
            self.gui_process.stdin.write(packet)
        self._frame_counter += 1


# next idea: download the texture after the draw and then encode it on the CPU... any GUI offscreen and onscreen!
# problem is that in rendercanvas we do _rc_draw_and_present... meaning no access in between - could be a limitation.
def download_texture(shader: Shadertoy) -> np.ndarray:
    current_texture = shader._present_context.get_current_texture() # is alive before present()!
    bpp = 4 # TODO read shader._format? not always a wgpu.TextureFormat anymore... but could be easier to parse
    # needs to be aligned to 256 bytes, but apparently can be padded here: https://docs.rs/wgpu/latest/wgpu/struct.TexelCopyBufferLayout.html#structfield.bytes_per_row
    bytes_per_row = (((bpp * current_texture.size[0])//256)+1) * 256
    nbytes = bytes_per_row * current_texture.size[1]

    # TODO can this be a mapped buffer?
    # TODO should be reused!
    gpu_buffer = shader._device.create_buffer(
        size=nbytes,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )
    command_encoder = shader._device.create_command_encoder()

    command_encoder.copy_texture_to_buffer(
        source={"texture": current_texture}, # sensible defaults exist!
        destination={
            "buffer": gpu_buffer,
            "bytes_per_row": bytes_per_row,
            # "rows_per_image": current_texture.size[1], # can be omitted as there is one image only.
        },
        copy_size=current_texture.size,
    )
    shader._device.queue.submit([command_encoder.finish()])
    frame_mem = shader._device.queue.read_buffer(gpu_buffer) # more like the memoryview

    # can we reuse this destination?
    frame_arr = np.asarray(frame_mem, dtype=np.uint8)
    frame_arr = frame_arr.reshape(
        current_texture.size[1],
        bytes_per_row // bpp,  # width in pixels
        4  # 4 color channels
    )
    frame_arr = frame_arr[:, :current_texture.size[0], :] # crop away the padding again

    return frame_arr

def encode_frame(frame_arr: np.ndarray, out_stream: av.VideoStream) -> None:
    """
    Encode a single frame from a memoryview to the output stream.
    """
    # TODO: can we directly use the memoryview/buffer here? -> VideoPlane?
    # .from_bytes, .from_numpy_buffer, .copy_bytes_to_plane etc - there might be a lower function that could be faster.
    frame = av.VideoFrame.from_ndarray(frame_arr, format="bgra") # TODO: rgba is a possibility here!
    # TODO: time and framerate?
    # maybe we need to accumulate a few frames before encoding them at once? not sure what is faster...
    for packet in out_stream.encode(frame):
        out_stream.container.mux(packet)


if __name__ == "__main__":
    shader = Shadertoy(shader_code=shader_code, resolution=(800, 450))
    # shader = Shadertoy.from_id("tXK3Rd", canvas=ffmpeg_canvas, resolution=(800, 450)) # I made one with mouse interactivity to test here!
    shader = Shadertoy.from_id("t3tXz8", resolution=(1280, 720), offscreen=True) # another one of mine...
    record(shader, output_file="terrain2.mp4", duration=60.0, framerate=60, mouse_pos=(299.0, 39.0, 5, 100), target_size=10.0)
    # 1minute of 720p 60fps h264 takes over 90 seconds here... not great given that it runs at over 165 fps without recording.

    
    # container = av.open("download_output.mp4", mode="w")
    # out_stream = container.add_stream(
    #     "h264",
    #     width=shader.resolution[0], # resolutions have to be divisible by 2 or 4 for h264
    #     height=shader.resolution[1],
    #     pix_fmt="yuv420p",
    #     bit_rate=20_000_000,
    #     rate=60,
    # )

    # def _draw_download_and_encode() -> None:
    #     """
    #     Draw the shader, download the texture and encode it to the output stream.
    #     """
    #     shader._draw_frame() # doesn't call present yet
    #     # TODO: this could be a toggle with a keybind to have in the future! (maybe indicate recording and time in the title?)
    #     frame_mem = download_texture(shader)
    #     encode_frame(frame_mem, out_stream) # seems really slow.. drops framerate from 165 to 48

    #     # present happens after this as part of the draw_and_present function


    # shader._canvas.request_draw(_draw_download_and_encode)
    # loop.run()

    print("done?")

# ideas: (tracking from https://github.com/pygfx/shadertoy/issues/52)
