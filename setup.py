from setuptools import setup, find_packages, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

#with open("requirements.txt", "r") as fh:
#    requirements = fh.read()
#    requirements = requirements.split()

setup(
    name="pyrl",
    version="0.0.1",
    author="Filipo Studzinski Perotto",
    author_email="filipo.perotto@onera.fr",
    description="ACAS-Xu Environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/fsperotto/acas",
    #license = 'MIT',
    package_dir = {"src":"src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_data = {'images':['src/acas/img/*.png', 'src/acas/img/*.jpg', 'src/acas/img/*.bmp'],
    #                'nn':['nn/*.onnx', 'nn/*.nnet'],
    #                'lut':['lut/*.npz']},
    include_package_data=True,    #then see MANIFEST.in
    python_requires='>=3.7',
    #install_requires = requirements,
    #tests_require = [],
)