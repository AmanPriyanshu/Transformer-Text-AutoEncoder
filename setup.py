from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='Transformer_Text_AutoEncoder',
    version='0.0.5',    
    description='Transformer Text AutoEncoder: An autoencoder is a type of artificial neural network used to learn efficient encodings of unlabeled data, the same is employed for textual data employing pre-trained models from the hugging-face library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AmanPriyanshu/Transformer-Text-AutoEncoder',
    author='Aman Priyanshu',
    author_email='amanpriyanshusms2001@gmail.com',
    license='BSD 2-clause',
    packages=['Transformer_Text_AutoEncoder'],
    install_requires=['tqdm>=4',
                      'nltk>=3',
                      'numpy',
                      'transformers',
                      'torch',                    
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Operating System :: Microsoft :: Windows :: Windows 8',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)