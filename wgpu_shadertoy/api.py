import json
import os

import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from wgpu import logger

from .inputs import ShadertoyChannel

HEADERS = {"user-agent": "https://github.com/pygfx/shadertoy script"}

OPTIONS = Options()
OPTIONS.add_argument("--headless=new")
OPTIONS.add_argument("user-agent=a python script to scrape non API shaders")


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


def _scrape_shader(id_or_url: str) -> dict:
    """
    Scrape shaders from Shadertoy.com even if they are not public or API accessible.
    """
    driver = webdriver.Chrome(options=OPTIONS)
    if "/" in id_or_url:
        shader_id = id_or_url.rstrip("/").split("/")[-1]
    else:
        shader_id = id_or_url
    url = f"https://www.shadertoy.com/view/{shader_id}"
    driver.get(url)
    if driver.title == "Error - Shadertoy BETA":
        raise RuntimeError(
            f"Failed to load shader at {url}, perhaps it's public or deleted."
        )
    # Wait until gShaderToy.mIsRendering is true so we get all the outputs we require
    WebDriverWait(driver, 3).until(
        lambda driver: driver.execute_script("return gShaderToy.mIsRendering;")
    )

    extraction_script = """
    json_data = gShaderToy.Save();
    return json_data;
    """
    json_data = driver.execute_script(extraction_script)
    driver.quit()

    # constrcut a dict exactly like the API return would provide it
    shader_data = {
        "Shader": {
            "info": json_data["info"],
            "ver": json_data["ver"],
            "renderpass": json_data["renderpass"],
        }
    }
    del shader_data["Shader"]["info"]["usePreview"]
    for rp in shader_data["Shader"]["renderpass"]:
        for inp in rp["inputs"]:
            inp["src"] = inp.pop("filepath")
            inp["ctype"] = inp.pop("type")

    return shader_data


def _download_media_channels(inputs: list):
    """
    Downloads media (currently just textures) from Shadertoy.com and returns a list of `ShadertoyChannel` to be directly used for `inputs`.
    Requires internet connection (API key not required).
    """
    media_url = "https://www.shadertoy.com"
    channels = {}
    for inp in inputs:
        if inp["ctype"] != "texture":
            continue  # TODO: support other media types
        response = requests.get(media_url + inp["src"], headers=HEADERS, stream=True)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"Failed to load media {media_url + inp['src']} with status code {response.status_code}"
            )
        img = Image.open(response.raw)
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
        print(f"Shadertoy {url} not retieved by API, trying to scrapt it ...")
        return _scrape_shader(id_or_url)
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
                inputs = _download_media_channels(r_pass["inputs"])
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
