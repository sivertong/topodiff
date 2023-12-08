from setuptools import setup

setup(
    name="topodiff",
    py_modules=["topodiff"],
    # 注意 此处使用了scikit取代了已废弃的sklearn
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "matplotlib", "scikit-learn", "solidspy", "opencv-python"],
)
