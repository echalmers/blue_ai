from setuptools import setup

setup(
    name="blue_ai_envs",
    version="0.0.1",
    install_requires=["minigrid", "imageio", "gymnasium"],
    include_package_data=True,
    package_data={"": ["*.png"]},
)
