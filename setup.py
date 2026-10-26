from setuptools import setup, find_packages

setup(
    name="dotguard",
    version="1.1.0",
    packages=find_packages(),
    py_modules=["cli"],
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
    ],
    entry_points={
        "console_scripts": [
            "dotguard=cli:main",
        ],
    },
)
