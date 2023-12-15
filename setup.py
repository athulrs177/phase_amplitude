from setuptools import setup, find_packages

setup(
    name='phase_amplitude_diagram',
    version='0.1.0',
    url="https://github.com/athulrs177/phase_amplitude_diagram",
    author="Athul Rasheeda Satheesh",
    author_email="athul.satheesh@kit.edu",
    packages=["phase_amplitude_diagram"],
    install_requires=[
        'numpy',
        'xarray',
        'pandas',
        'matplotlib'
    ],
)
