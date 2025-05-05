import re
from .utils import UniformArray

# could imgui become just another RenderPass after Image? I got to understand backend vs renderer first.
# make become part of .passes??
# todo: raise error if imgui isn't installed (only if this module is required?)


def parse_constants(code, common_code):
    # todo:
    # WGSL variants??
    # re/tree-sitter/loops and functions?
    # parse and collect constants from shadercode (including common pass?)
    # get information about the line, the type and it's initial value
    # make up a range (maybe just the order of magnitude + 1 as max and 0 as min (what about negative values?))
    # what is the return type? (line(int), type(str), value(float/tuple/int?)) maybe proper dataclasss for once
    
    # mataches the macro: #define NAME VALUE
    define_pattern = re.compile(r"#define\s+(\w+)\s+(.+)")

    constants = []
    for li, line in enumerate(code.splitlines()):
        match = define_pattern.match(line.rstrip())
        if match:
            name, value = match.groups()
            if "." in value: #value.isdecimal?
                # TODO: wgsl needs to be more specific (f32 for example?) - but there is no preprocessor anyways...
                dtype = "f" #default float (32bit)
                value = float(value)
            elif value.isdecimal: # value.isalum?
                dtype = "I" # "big I (32bit)"
                value = int(value)
            else:
                # TODO complexer types?
                print(f"can't parse type for constant {name} with value {value}, skipping")
                continue
            # todo: remove lines here? (comment out better)
            constants.append((li, name, value, dtype)) # what about line to remove?
            print(f"In line {li} found constant: {name} with value: {value} of dtype {dtype}")

    return constants

def make_uniform(constants):
    arr_data = []
    for constant in constants:
        _, name, value, dtype = constant
        arr_data.append(tuple([name, dtype, 1]))
    data = UniformArray(*arr_data)
    
    # todo:
    # figure out order due to padding/alignment: https://www.w3.org/TR/WGSL/#alignment-and-size
    # return a UniformArray object (cycling import?)
    # (does this need to be a class to update the values?)
    return data

def make_import(uniform) -> str:
    # codegen the import block for this uniform (including binding? - which number?)
    # could be part of the UniformArray class maybe?
    # to be pasted near the top of the fragment shader code.
    # alternatively: insert these in the ShadertoyInputs uniform?
    # better yet: use push constants
    # TODO: can you even import a uniform struct and then have these available as global?
    # maybe I got to add them back in as #define name = constant.name or something

    pass


# imgui stuff
def update_gui():
    # todo: look at exmaples nad largely copy nad paste, will be called in the draw_frame function I think.

    pass