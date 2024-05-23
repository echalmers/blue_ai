from setuptools import setup

setup(
    version="0.0.1",
    name="blue_ai_envs",
    install_requires=[
        "imageio",
        "matplotlib",
        "minigrid",
        "pandas",
        "scipy",
        "seaborn",
        "torch",
        "tqdm",
    ],
    include_package_data=True,
    package_data={"": ["*.png"]},
    packages=["blue_ai"],
)
