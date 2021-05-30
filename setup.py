import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="remix",
    version="0.0.1",
    author="Mateo Espinosa Zarlenga",
    author_email="me466@cam.ac.uk",
    description="Rule Extraction Methods for Interactive eXplainability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mateoespinosa/remix",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.9',
    install_requires=[
        "dill==0.3.3",
        "flexx==0.8.1",
        "Keras-Preprocessing==1.1.2",
        "numpy<1.19.0",
        "pandas==0.25.3",
        "prettytable==1.0.1",
        "pscript==0.7.5",
        "PyYAML==5.3.1",
        "rpy2==3.3.6",
        "scikit-learn==0.23.2",
        "scipy>=1.2.0",
        "sklearn==0.0",
        "tensorboard-plugin-wit>=1.7.0",
        "tensorboard>=2.3.0",
        "tensorflow-estimator>=2.3.0",
        "tensorflow>=2.3.1",
        "tqdm==4.51.0",
    ],
)
