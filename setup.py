from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT: str = '-e .'

def get_requirements(filename: str) -> List[str]:
    with open(filename, 'r') as file:
        return [req.replace('\n', '') for req in file.readlines() if req != HYPHEN_E_DOT]
    
setup(
    name='health_ins',
    version='0.0.1',
    author='Tonzai',
    author_email='hoseatonzaifavour@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    
)