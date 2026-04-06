from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Esta función retorna la lista de requerimientos filtrando '-e .'
    """
    requirements = []
    try:
        with open(file_path, 'r') as file_obj:
            # Leemos y quitamos los saltos de línea
            requirements = [req.replace("\n", "") for req in file_obj.readlines()]

            # Filtramos el disparador del modo editable si existe
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_path}")
        
    return requirements

setup(
    name='proyect',
    version='0.0.1',
    author='Ignacio',
    author_email='ignacionicolastoyos@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
