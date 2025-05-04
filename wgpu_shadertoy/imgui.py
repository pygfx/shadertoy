import re

# could imgui become just another RenderPass after Image? I got to understand backend vs renderer first.

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
    for line in code.splitlines():
        match = define_pattern.match(line.rstrip())
        if match:
            constants.append(match.groups())
            name, value = match.groups()
            print(f"Found constant: {name} with value: {value}")

    return constants

def make_uniform(constants):
    # todo:
    # figure out order due to padding/alignment: https://www.w3.org/TR/WGSL/#alignment-and-size
    # return a UniformArray object (cycling import?)
    # (does this need to be a class to update the values?)
    pass

# imgui stuff
def update_gui():
    # todo: look at exmaples nad largely copy nad paste, will be called in the draw_frame function I think.

    pass