import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="sippy",
    version="0.2.0",
    author="Giuseppe Armenise",
    description="Systems Identification Package for Python - Modern Architecture",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/CPCLAB-UNIPI/sippy",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.13",

    install_requires=(
        "numpy",
        "scipy",
        "control>=0.10.2",
        "slycot>=0.7.0",
    ),
    classifiers=(
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ),
)
