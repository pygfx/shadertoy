import re
from imgui_bundle import imgui as ig #TODO: rename (git mv) the file instead.
from .utils import UniformArray
from wgpu.utils.imgui import ImguiWgpuBackend
from dataclasses import dataclass


# could imgui become just another RenderPass after Image? I got to understand backend vs renderer first.
# make become part of .passes??
# todo: raise error if imgui isn't installed (only if this module is required?)


@dataclass
class ShaderConstant:
    renderpass_pass: str #maybe this is a RenderPass pointer?
    line_number: int
    original_line: str
    name: str
    value: int | float
    shader_dtype: str # float, int, vec2, vec3, bool etc.

    def c_type_format(self) -> str:
        # based on these for the memoryview cast:
        # https://docs.python.org/3/library/struct.html#format-characters
        if self.shader_dtype == "float":
            return "f"
        elif self.shader_dtype == "int":
            return "i"
        elif self.shader_dtype == "uint":
            return "I"
        # add more types as needed
        return "?"

def parse_constants(code:str, common_code) -> list[ShaderConstant]:
    # todo:
    # WGSL variants??
    # re/tree-sitter/loops and functions?
    # parse and collect constants from shadercode (including common pass?)
    # get information about the line, the type and it's initial value
    # make up a range (maybe just the order of magnitude + 1 as max and 0 as min (what about negative values?))
    # what is the return type? (line(int), type(str), value(float/tuple/int?)) maybe proper dataclasss for once

    # for multipass shader this might need to be per pass (rpass.value) ?
    # mataches the macro: #define NAME VALUE
    # TODO there can be characters in numerical literals, such as x and o for hex and octal representation or e for scientific notation
    # technically the macros can also be an expression that is evaluated to be a number... such as # define DOF 10..0/30.0 - so how do we deal with that?
    define_pattern = re.compile(r"#\s*define\s+(\w+)\s+([\d.]+)") #for numerical literals right now.
    if_def_template = r"#(el)?if\s+" #preprocessor ifdef blocks can't become uniforms. replacing these dynamically will be difficult.

    constants = []
    for li, line in enumerate(code.splitlines()):
        match = define_pattern.match(line.strip())
        if match:
            name, value = match.groups()
            if_def_pattern = re.compile(if_def_template + name)
            if if_def_pattern.findall(code):
                #.findall over .match because because not only the beginning matters here
                print(f"skipping constant {name}, it needs to stay a macro")
                continue

            if "." in value: #value.isdecimal?
                # TODO: wgsl needs to be more specific (f32 for example?) - but there is no preprocessor anyways...
                dtype = "float" #default float (32bit)
                value = float(value)
            elif value.isdecimal(): # value.isnumeric?
                dtype = "int" # "big I (32bit)"
                value = int(value)
            else:
                # TODO complexer types?
                print(f"can't parse type for constant {name} with value {value}, skipping")
                continue

            constant = ShaderConstant(
                renderpass_pass="image",  # TODO: shouldn't be names.
                line_number=li,
                original_line=line.strip(),
                name=name,
                value=value,
                shader_dtype=dtype
            )
            # todo: remove lines here? (comment out better)
            constants.append(constant)
            print(f"In line {li} found constant: {name} with value: {value} of dtype {dtype}")

    # maybe just named tuple instead of dataclass?
    return constants

def make_uniform(constants) -> UniformArray:
    arr_data = []
    for constant in constants:
        arr_data.append(tuple([constant.name, constant.c_type_format(), 1]))
    data = UniformArray(*arr_data)

    # init data
    for constant in constants:
        data[constant.name] = constant.value

    # TODO:
    # is there issues with padding? (maybe solve in the class)
    # figure out order due to padding/alignment: https://www.w3.org/TR/WGSL/#alignment-and-size
    # return a UniformArray object (cycling import?)
    # (does this need to be a class to update the values?)
    return data

def construct_imports(constants: list[ShaderConstant], constant_binding_idx=10) -> str:
    # codegen the import block for this uniform (including binding? - which number?)
    # could be part of the UniformArray class maybe?
    # to be pasted near the top of the fragment shader code.
    # alternatively: insert these in the ShadertoyInputs uniform?
    # better yet: use push constants
    # TODO: can you even import a uniform struct and then have these available as global?
    # maybe I got to add them back in as #define name = constant.name or something
    var_init_lines = []
    var_mapping_lines = []
    for const in constants:
        var_init_lines.append(f"{const.shader_dtype} {const.name};")
        var_mapping_lines.append(f"# define {const.name} const_input.{const.name}")

    new_line = "\n" # pytest was complaining about having blackslash in an f-string
    code_construct = f"""
    uniform struct ConstantInput {{
        {new_line.join(var_init_lines)}
    }};
    layout(binding = {constant_binding_idx}) uniform ConstantInput const_input;
    {new_line.join(var_mapping_lines)}
    """
    # TODO messed up indentation...
    return code_construct


# imgui stuff
def update_gui():
    # todo: look at exmaples nad largely copy nad paste, will be called in the draw_frame function I think.

    pass


def gui(constants: list[ShaderConstant], constants_data: UniformArray):
    ig.new_frame()
    ig.set_next_window_pos((0, 0), ig.Cond_.appearing)
    ig.set_next_window_size((400, 0), ig.Cond_.appearing)
    ig.begin("Shader constants", None)

    ig.text('in-dev imgui overlay\n')
    if ig.is_item_hovered():
        ig.set_tooltip("TODO")

    # create the sliders?
    for const in constants:
        if const.shader_dtype == "float":
            _, constants_data[const.name] = ig.slider_float(const.name, constants_data[const.name], 0, const.value*2.0)
        elif const.shader_dtype == "int":
            _, constants_data[const.name] = ig.slider_int(const.name, constants_data[const.name], 0, const.value*2)
            # TODO: improve min/max for negatives

    ig.end()
    ig.end_frame()
    ig.render()
    return ig.get_draw_data()

def get_backend(device, canvas, render_texture_format):
    """
    copied from backend example, held here to avoid clutter in the main class
    """

    # init imgui backend
    ig.create_context()
    imgui_backend = ImguiWgpuBackend(device, render_texture_format)
    imgui_backend.io.display_size = canvas.get_logical_size()
    imgui_backend.io.display_framebuffer_scale = (
        canvas.get_pixel_ratio(),
        canvas.get_pixel_ratio(),
    )
    return imgui_backend
