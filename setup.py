import json
from setuptools import setup, find_packages

with open('metainfo.json') as file:
    metainfo = json.load(file)

setup(
    name='cluster_based_sisso',
    version='1.0',
    author=', '.join(metainfo['authors']),
    author_email=metainfo['email'],
    url=metainfo['url'],
    description=metainfo['title'],
    long_description=metainfo['description'],
    packages=find_packages(),
    install_requires=['sissopp', 'tensorflow','numpy','pandas','scipy','scikit-learn','seaborn','toml'],
)
