# py-mm

Markov models for categorical sequential data in Python. The joint probabiity 

<img src="https://latex.codecogs.com/svg.latex?\space p(x_1, x_2,\ldots,x_N)=p(x_1)p(x_2|x_1)\ldots p(x_{M}|x_{M-1},\ldots,x_1)\prod_{n=M+1}^N p(x_n|x_{n-1},\ldots,x_{n-M})" title=" x=\frac{-1\pm\sqrt{b^2-4ac}}{2a}" />

for an ordered sequence  <img src="https://latex.codecogs.com/svg.latex?\space (x_1,x_2,\ldots,x_N)" title=" x=\frac{-1\pm\sqrt{b^2-4ac}}{2a}" /> of the categorical variable <img src="https://latex.codecogs.com/svg.latex?\space x" title=" x" /> is composed of M+1 conditional distributions of order <img src="https://latex.codecogs.com/svg.latex?\space m=[0, M]" title=" m" />. Each component <img src="https://latex.codecogs.com/svg.latex?\space p(x_n|x_{n-1},\ldots,x_{n-m})" title=" m" /> is represented by a rank  <img src="https://latex.codecogs.com/svg.latex?\space m+1" title=" x" /> tensor of transition state probabilities. These are inferred under maximum likelihood estimation of the data.


```python
import numpy as np
from pymm.models import MarkovModel


# number of states for categorical variable
K = 2

# order of Markov model
M = 1

# generator for synthetic data
generator = MarkovModel(K=K, M=M, random_init=True)

# create artificial sequential dataset
X = [generator.sample(10) for _ in range(1000)]

# Markov model to infer joint distribution for sequential data
model = MarkovModel(K=K, M=M)

# infer conditional transition state probabilities
%timeit -r 1 -n 1 model.fit(X)
```

### Install
To install, download or git clone the full repository and then run 

``
python setup.py install
``

from the repository root in your chosen python environment. PyPi release to follow.

### Status
This repository is pre-release and is in active development. Please check for regular updates and switch to the PyPi release when available.
