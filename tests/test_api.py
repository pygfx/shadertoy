import pytest
from testutils import can_use_wgpu_lib

from wgpu_shadertoy.api import _get_api_key, shader_args_from_json, shadertoy_from_id

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)

try:
    API_KEY = _get_api_key()
except Exception as e:
    pytest.skip("Skipping API tests: " + str(e), allow_module_level=True)


# coverage for shadertoy_from_id(id_or_url)
def test_from_id_with_invalid_id():
    with pytest.raises(RuntimeError):
        shadertoy_from_id("invalid_id")


# coverage for shader_args_from_json(dict_or_path, **kwargs)
def test_from_json_with_invalid_path():
    with pytest.raises(FileNotFoundError):
        shader_args_from_json("/invalid/path")


def test_from_json_with_invalid_type():
    with pytest.raises(TypeError):
        shader_args_from_json(123)
