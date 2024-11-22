from setuptools import setup, find_packages


install_requires = [
    'pytest>=6.0',  # Test dependency, but included here to auto-install
    'torch',        # Test dependency
    'numpy',        # Test dependency
    'jaxtyping',    # Test dependency
    'einops',       # Test dependency
    'fancy_einsum', # Test dependency
    'plotly==5.19.0',
    'timm',         # Test dependency
    'transformers', # Test dependency
    'scikit-learn', # Test dependency
    'datasets',
    'line_profiler',
    'wandb',
    'matplotlib',
    'kaleido',
    'open-clip-torch',
    'tinyimagenet',
    'torchattacks',
    'jupyter',
    'goodfire',
    'accelerate',
]

setup(
    name='apart_hack',
    version='0.1.0',
    author='Ed',
    author_email='ed@gmail.com',
    description='Apart Hackathon',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={
},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    keywords='vision-transformer, clip, multimodal, machine-learning, mechanistic interpretability',
    zip_safe=False,
    extras_require={
        'sae': ['sae-lens==2.1.3'],
        'arrow': ['pyarrow']  # to use: pip install -e .[sae] # as of 2.1.3, windows will require pip install sae-lens==2.1.3 --no-dependencies followed by manually installing needed packages
    },
)
