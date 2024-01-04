import re
import platform

from setuptools import find_packages, setup
from wheel.bdist_wheel import get_platform, bdist_wheel as _bdist_wheel


NAME = "shadertoy"
SUMMARY = "Shadertoy implementation based on wgpu-py"

with open(f"{NAME}/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)



setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*"]
    ),
    python_requires=">=3.8.0",
    license="BSD 2-Clause",
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jan Kels",
    author_email="Jan.Kels@hhu.de",
    url="https://github.com/pygfx/shadertoy",
)
