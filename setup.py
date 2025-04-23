import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "hezarfenai",
    version = "0.1.3",
    author = "Ömer Asaf Karasu",
    author_email = "omer.karasu@niafix.com",
    description = "A Teknofest project named HezarfenAI Official Python Package",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
            "nltk>=3.9",
            "scikit-learn>=1.0",
            "pandas",
            "TurkishStemmer",
            "numpy",
            "scipy",
            "joblib"
        ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)
