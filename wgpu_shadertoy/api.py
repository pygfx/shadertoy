import json
import os

import numpy as np
import requests
from PIL import Image


# TODO: write function that gives a good error message
def get_api_key():
    key = os.environ.get("SHADERTOY_KEY", None)
    if key is None:
        raise Exception(
            "SHADERTOY_KEY environment variable not set, please set it to your Shadertoy API key to use API features. Follow the instructions at https://www.shadertoy.com/howto#q2"
        )
    return key


API_KEY = get_api_key()

HEADERS = {"user-agent": "https://github.com/pygfx/shadertoy script"}


def get_shadertoy_by_id(shader_id) -> dict:
    """
    Fetches a shader from Shadertoy.com by its ID and returns the JSON data as dict.
    """
    url = f"https://www.shadertoy.com/api/v1/shaders/{shader_id}"
    response = requests.get(url, params={"key": API_KEY}, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(
            f"Failed to load shader at https://www.shadertoy.com/view/{shader_id}"
        )
    if "Error" in response.json():
        raise Exception(
            f"Shadertoy API error: {response.json()['Error']}, perhaps the shader isn't set to `public+api`"
        )
    return response.json()


# TODO: consider caching media locally?
def download_media_channels(inputs):
    from . import ShadertoyChannel  # lazy import to avoid circular imports?

    """
    Downloads media (currently just Textures) from Shadertoy.com and returns a list of `ShadertoyChannel` to be directly used for `inputs`.
    """
    media_url = "https://www.shadertoy.com"
    channels = {}
    for inp in inputs:
        if inp["ctype"] != "texture":
            continue  # we currently can't handle stuff that isn't textures for input...
        response = requests.get(media_url + inp["src"], headers=HEADERS, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to load media {media_url + inp['src']}")
        img = Image.open(response.raw).convert("RGBA")
        img_data = np.array(img)
        channel = ShadertoyChannel(
            img_data, kind="texture", wrap=inp["sampler"]["wrap"]
        )
        channels[inp["channel"]] = channel
    return list(channels.values())


def build_shader_from_data(cls, shader_data, **kwargs):
    """
    Builds a `Shadertoy` instance from a JSON-like dict of Shadertoy.com shader data.
    """
    if not isinstance(shader_data, dict):
        raise Exception("shader_data must be a dict")
    main_image_code = ""
    common_code = ""
    inputs = []
    for r_pass in shader_data["Shader"]["renderpass"]:
        if r_pass["type"] == "image":
            main_image_code = r_pass["code"]
            if r_pass["inputs"] is not []:
                inputs = download_media_channels(r_pass["inputs"])
        elif r_pass["type"] == "common":
            common_code = r_pass["code"]
        else:
            # TODO should be a warning and not verbose!
            print(
                f"renderpass of type {r_pass['type']} not yet supported, will be ommitted"
            )
    title = f'{shader_data["Shader"]["info"]["name"]} by {shader_data["Shader"]["info"]["username"]}'
    shader = cls(
        main_image_code,
        common=common_code,
        shader_type="glsl",
        inputs=inputs,
        title=title,
        **kwargs,
    )
    return shader


class APIMixin:
    @classmethod
    def from_id(cls, shader_id, **kwargs):
        shader_data = get_shadertoy_by_id(shader_id)
        shader = build_shader_from_data(cls, shader_data, **kwargs)
        return shader

    # TODO consider caching locally to temp, .cache or custom?
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        shader = build_shader_from_data(cls, data)
        return shader
