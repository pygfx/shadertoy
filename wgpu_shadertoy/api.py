import json
import os
import sys

import requests
from PIL import Image
from wgpu import logger

from .inputs import ShadertoyChannel

HEADERS = {"user-agent": "https://github.com/pygfx/shadertoy script"}


def _get_api_key() -> str:
    key = os.environ.get("SHADERTOY_KEY", None)
    if key is None:
        raise ValueError(
            "SHADERTOY_KEY environment variable not set, please set it to your Shadertoy API key to use API features. Follow the instructions at https://www.shadertoy.com/howto#q2"
        )
    test_url = "https://www.shadertoy.com/api/v1/shaders/query/test"
    test_response = requests.get(test_url, params={"key": key}, headers=HEADERS)
    if test_response.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"Failed to use ShaderToy API with key: {test_response.status_code}"
        )
    test_response = test_response.json()
    if "Error" in test_response:
        raise ValueError(
            f"Failed to use ShaderToy API with key: {test_response['Error']}"
        )
    return key


def _get_cache_dir(subdir="media") -> os.PathLike:
    """
    returns the OS appropriate cache directory
    """
    if sys.platform.startswith("win"):
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], "shadertoy")
    elif sys.platform.startswith("darwin"):
        cache_dir = os.path.join(os.environ["HOME"], "Library", "Caches", "shadertoy")
    else:
        if "XDG_CACHE_HOME" in os.environ:
            cache_dir = os.path.join(os.environ["XDG_CACHE_HOME"], "shadertoy")
        else:
            cache_dir = os.path.join(os.environ["HOME"], ".cache", "shadertoy")
    cache_dir = os.path.join(cache_dir, subdir)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create cache directory at {cache_dir}, due to {e}")
    return cache_dir


def _download_media_channels(inputs: list, use_cache=True):
    """
    Downloads media (currently just textures) from Shadertoy.com and returns a list of `ShadertoyChannel` to be directly used for `inputs`.
    Requires internet connection (API key not required).
    """
    media_url = "https://www.shadertoy.com"
    channels = {}
    cache_dir = _get_cache_dir("media")
    for inp in inputs:
        if inp["ctype"] != "texture":
            continue  # TODO: support other media types

        cache_path = os.path.join(cache_dir, inp["src"].split("/")[-1])
        if use_cache and os.path.exists(cache_path):
            img = Image.open(cache_path)
        else:
            response = requests.get(
                media_url + inp["src"], headers=HEADERS, stream=True
            )
            if response.status_code != 200:
                raise requests.exceptions.HTTPError(
                    f"Failed to load media {media_url + inp['src']} with status code {response.status_code}"
                )
            img = Image.open(response.raw)
            if use_cache:
                img.save(cache_path)

        channel = ShadertoyChannel(img, kind="texture", **inp["sampler"])
        channels[inp["channel"]] = channel
    return list(channels.values())


def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shadertoy_from_id(id_or_url) -> dict:
    """
    Fetches a shader from Shadertoy.com by its ID (or url) and returns the JSON data as dict.
    """
    if "/" in id_or_url:
        shader_id = id_or_url.rstrip("/").split("/")[-1]
    else:
        shader_id = id_or_url
    url = f"https://www.shadertoy.com/api/v1/shaders/{shader_id}"
    response = requests.get(url, params={"key": _get_api_key()}, headers=HEADERS)
    if response.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"Failed to load shader at https://www.shadertoy.com/view/{shader_id} with status code {response.status_code}"
        )
    shader_data = response.json()
    if "Error" in shader_data:
        raise RuntimeError(
            f"Shadertoy API error: {shader_data['Error']} for https://www.shadertoy.com/view/{shader_id}, perhaps the shader isn't set to `public+api`"
        )
    return shader_data


def shader_args_from_json(dict_or_path, **kwargs) -> dict:
    """
    Builds the args for a `Shadertoy` instance from a JSON-like dict of Shadertoy.com shader data.
    """
    if isinstance(dict_or_path, (str, os.PathLike)):
        shader_data = _load_json(dict_or_path)
    else:
        shader_data = dict_or_path
    use_cache = kwargs.pop("use_cache", True)

    if not isinstance(shader_data, dict):
        raise TypeError("shader_data must be a dict")
    main_image_code = ""
    common_code = ""
    inputs = []
    if "Shader" not in shader_data:
        raise ValueError(
            "shader_data must have a 'Shader' key, following Shadertoy export format."
        )
    for r_pass in shader_data["Shader"]["renderpass"]:
        if r_pass["type"] == "image":
            main_image_code = r_pass["code"]
            if r_pass["inputs"] is not []:
                inputs = _download_media_channels(r_pass["inputs"], use_cache=use_cache)
        elif r_pass["type"] == "common":
            common_code = r_pass["code"]
        else:
            # TODO should be a warning and not verbose!
            logger.warn(
                f"renderpass of type {r_pass['type']} not yet supported, will be omitted."
            )
    title = f'{shader_data["Shader"]["info"]["name"]} by {shader_data["Shader"]["info"]["username"]}'

    shader_args = {
        "shader_code": main_image_code,
        "common": common_code,
        "shader_type": "glsl",
        "inputs": inputs,
        "title": title,
        **kwargs,
    }
    return shader_args
