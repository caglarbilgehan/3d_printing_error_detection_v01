from setuptools import setup, find_packages

setup(
    name="printer_monitor",
    version="0.1.0",
    description="3D Printer Web Monitoring System",
    packages=find_packages(),
    install_requires=[
        'flask>=2.0.0',
        'opencv-python>=4.5.0',
        'numpy>=1.19.0',
    ],
    python_requires='>=3.7',
)
