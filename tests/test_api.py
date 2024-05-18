import pytest
from testutils import can_use_wgpu_lib

from wgpu_shadertoy.api import _get_api_key, shader_args_from_json, shadertoy_from_id

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


@pytest.fixture
def api_available():
    """
    Skip tests some tests if no API is unavailable.
    """
    try:
        return _get_api_key()
    except Exception as e:
        pytest.skip("Skipping API tests: " + str(e))


# coverage for shadertoy_from_id(id_or_url)
def test_from_id_with_invalid_id(api_available):
    with pytest.raises(RuntimeError):
        shadertoy_from_id("invalid_id")


def test_from_id_with_valid_id(api_available):
    # shadertoy source: https://www.shadertoy.com/view/mtyGWy by kishimisu
    data = shadertoy_from_id("mtyGWy")
    assert "Shader" in data
    assert data["Shader"]["info"]["id"] == "mtyGWy"
    assert data["Shader"]["info"]["username"] == "kishimisu"


def test_shadertoy_from_id(api_available):
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    # shadertoy source: https://www.shadertoy.com/view/l3fXWN by Vipitis
    shader = Shadertoy.from_id("l3fXWN")

    assert shader.title == '"API test for CI" by jakel101'
    assert shader.shader_type == "glsl"
    assert shader.shader_code.startswith("//Confirm API working!")
    assert shader.common.startswith("//Common pass loaded!")
    assert (
        shader.image.channels[0].sampler_settings["address_mode_u"] == "clamp-to-edge"
    )
    assert shader.image.channels[0].data.shape == (32, 256, 4)
    assert shader.image.channels[0].texture_size == (256, 32, 1)


def test_shadertoy_from_id_without_cache(api_available):
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    # shadertoy source: https://www.shadertoy.com/view/l3fXWN by Vipitis
    shader = Shadertoy.from_id("l3fXWN", use_cache=False)

    assert shader.title == '"API test for CI" by jakel101'
    assert shader.shader_type == "glsl"
    assert shader.shader_code.startswith("//Confirm API working!")
    assert shader.common.startswith("//Common pass loaded!")
    assert shader.image.channels != []


# coverage for shader_args_from_json(dict_or_path, **kwargs)
def test_from_json_with_invalid_path():
    with pytest.raises(FileNotFoundError):
        shader_args_from_json("/invalid/path")


def test_from_json_with_invalid_type():
    with pytest.raises(TypeError):
        shader_args_from_json(123)
