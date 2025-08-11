import re
from imgui_bundle import imgui as ig

from .utils import UniformArray
from wgpu.utils.imgui import ImguiWgpuBackend
# from wgpu_shadertoy.passes import RenderPass #circular import-.-
from dataclasses import dataclass


# could imgui become just another RenderPass after Image? I got to understand backend vs renderer first.
# make become part of .passes??
# todo: raise error if imgui isn't installed (only if this module is required?)


@dataclass
class ShaderConstant:
    # renderpass_pass: str #maybe this is a RenderPass pointer? likely redundant
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

def parse_constants(code:str) -> list[ShaderConstant]:
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
    define_pattern = re.compile(r"#\s*define\s+(\w+)\s+(-?[\d.]+)") #for numerical literals right now.
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
                # renderpass_pass="image",  # TODO: shouldn't be names.
                line_number=li,
                original_line=line.strip(),
                name=name,
                value=value,
                shader_dtype=dtype
            )
            # todo: remove lines here? (comment out better)
            constants.append(constant)
            print(f"In line {li} found constant: {name} with value: {value} of dtype {dtype}") # maybe name renderpass too?

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
    # return a UniformArray object too (cycling import?) also needs device handed down.
    # (does this need to be a class to update the values?)
    return data

# TODO mark private?
def construct_imports(constants: list[ShaderConstant], constant_binding_idx: int) -> str:
    # codegen the import block for this uniform (including binding? - which number?)
    # could be part of the UniformArray class maybe?
    # to be pasted near the top of the fragment shader code.
    # alternatively: insert these in the ShadertoyInputs uniform?
    # better yet: use push constants
    # TODO: can you even import a uniform struct and then have these available as global?
    # maybe I got to add them back in as #define name = constant.name or something

    if not constants:
        return ""

    var_init_lines = []
    var_mapping_lines = []
    for const in constants:
        var_init_lines.append(f"{const.shader_dtype} {const.name};")
        var_mapping_lines.append(f"# define {const.name} const_input{constant_binding_idx}.{const.name}")

    new_line = "\n" # pytest was complaining about having blackslash in an f-string
    code_construct = f"""
    uniform struct ConstantInput{constant_binding_idx} {{
        {new_line.join(var_init_lines)}
    }};
    layout(binding = {constant_binding_idx}) uniform ConstantInput{constant_binding_idx} const_input{constant_binding_idx};
    {new_line.join(var_mapping_lines)}
    """
    # the identifier name includes the digit so common doesn't cause redefinition!
    # TODO messed up indentation... textwrap.dedent?
    return code_construct

def replace_constants(code: str, constants: list[ShaderConstant], constant_binding_idx: int) -> str:
    """
    comment out existing constants and redefine them with uniform struct
    """
    code_lines = code.splitlines()
    for const in constants:
        # comment out existing constants
        code_lines[const.line_number] = f"// {code_lines[const.line_number]}"

    constant_headers = construct_imports(constants, constant_binding_idx)
    code_lines.insert(0, constant_headers)

    return "\n".join(code_lines)


# imgui stuff
def update_gui():
    # todo: look at exmaples nad largely copy nad paste, will be called in the draw_frame function I think.

    pass


def gui(renderpasses: list["RenderPass"]):
    ig.new_frame()
    ig.set_next_window_pos((0, 0), ig.Cond_.appearing)
    ig.set_next_window_size((0, 0), ig.Cond_.appearing) # auto size not wide enough with text :/
    ig.begin("Shader constants", None)
    ig.text('in-dev imgui overlay\n')

    if ig.is_item_hovered():
        ig.set_tooltip("TODO")

    # maybe we should have a global main or utils.get_main()?
    main = renderpasses[0].main

    # TODO: avoid duplication, maybe common should be a renderpass instance (at least a little bit) - or we iterate through constants lists
    if main._common_constants:
        if ig.collapsing_header("Common Constants", flags=ig.TreeNodeFlags_.default_open):
            for const in main._common_constants:
                if const.shader_dtype == "float":
                    _, main._common_constants_data[const.name] = ig.slider_float(f"{const.name}", main._common_constants_data[const.name], -const.value, const.value*2.0)
                elif const.shader_dtype == "int":
                    _, main._common_constants_data[const.name] = ig.slider_int(f"{const.name}", main._common_constants_data[const.name], -const.value, const.value*2)
                if ig.is_item_hovered() and ig.is_mouse_clicked(ig.MouseButton_.right):
                    main._common_constants_data[const.name] = const.value
                if ig.is_item_hovered():
                    ig.set_tooltip("Right click to reset")

    for rp in renderpasses: # TODO: most likely add common here?
        constants = rp._constants
        constants_data = rp._constants_data
        if ig.collapsing_header(f"{rp} Constants", flags=ig.TreeNodeFlags_.default_open):
            if hasattr(rp, "texture_front"): # isinstance(rp, BufferRenderPass)
                # make this another toggle? or a whole 2nd UI?
                front_view = rp.texture_front.create_view()
                front_ref = rp.main._imgui_backend.register_texture(front_view)
                scale = 0.25  # TODO dynamic zoom via width?
                # TODO: can we force a background? do we need to request additional view formats? -> ig.image_with_bg?
                buf_img = ig.image(front_ref, (front_view.size[0]*scale, front_view.size[1]*scale), uv0=(0,1), uv1=(1,0))

            # create the sliders?
            for const in constants:
                if const.shader_dtype == "float":
                    _, constants_data[const.name] = ig.slider_float(f"{const.name}", constants_data[const.name], -const.value, const.value*2.0)
                elif const.shader_dtype == "int":
                    _, constants_data[const.name] = ig.slider_int(f"{const.name}", constants_data[const.name], -const.value, const.value*2)
                    # TODO: improve min/max for negatives maybe infinite range with scaling?
                # right click to reset?
                if ig.is_item_hovered() and ig.is_mouse_clicked(ig.MouseButton_.right):
                    constants_data[const.name] = const.value
                    # print(f"Reset {const.name} to {const.value} from {constants_data[const.name]}")
                if ig.is_item_hovered():
                    ig.set_tooltip("Right click to reset")

    # TODO: control the size of these headers to make the window as small as possible after they are collapsed!
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
