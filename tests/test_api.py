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


def test_buffers_from_api(api_available):
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy, ShadertoyChannelBuffer

    # this is a Shadertoy we don't control - so it could change. perhaps we need a fork that is stable.
    # shadertoy source: https://www.shadertoy.com/view/4X33D2 by brisingre
    shader = Shadertoy.from_id("4X33D2")

    assert shader.title == '"Common Code (API Test)" by brisingre'
    assert "" not in shader.buffers.values()
    assert len(shader.image._input_headers) > 0
    assert type(shader.buffers["a"].channels[0]) == ShadertoyChannelBuffer
    assert shader.buffers["a"].channels[0].channel_idx == 0
    assert shader.buffers["a"].channels[0].buffer_idx == "a"
    assert shader.buffers["a"].channels[0].renderpass == shader.buffers["a"]
    assert type(shader.buffers["b"].channels[0]) == ShadertoyChannelBuffer
    assert shader.buffers["b"].channels[0].buffer_idx == "b"
    assert shader.buffers["b"].channels[0].renderpass == shader.buffers["b"]
    assert type(shader.buffers["c"].channels[0]) == ShadertoyChannelBuffer
    assert shader.buffers["c"].channels[0].buffer_idx == "c"
    assert shader.buffers["c"].channels[0].renderpass == shader.buffers["c"]
    assert type(shader.buffers["d"].channels[0]) == ShadertoyChannelBuffer
    assert shader.buffers["d"].channels[0].buffer_idx == "d"
    assert shader.buffers["d"].channels[0].renderpass == shader.buffers["d"]


# coverage for shader_args_from_json(dict_or_path, **kwargs)
def test_from_json_with_invalid_path():
    with pytest.raises(FileNotFoundError):
        shader_args_from_json("/invalid/path")


def test_from_json_with_invalid_type():
    with pytest.raises(TypeError):
        shader_args_from_json(123)
