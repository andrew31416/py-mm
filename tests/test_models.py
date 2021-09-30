import numpy as np
from sys import path
from typing import List
path.append('.')


from src.models import MarkovModel


def test_logprob_MarkovModel():
    """
    Check cumulative log probability over models is correct
    """
    def _test(M: int):
        # p(x1,x2,x3,x4,x5)=p(x5|x4,x3)p(x4|x3,x2)p(x3|x2,x1)p(x2|x1)p(x1)
        model = MarkovModel(K=2, M=M, random_init=True)

        # sequence
        x = np.asarray([0, 1, 0, 1, 1], dtype=int)

        if M==0:
            true = sum([model.models[0].log_prob(_x) for _x in x])
        elif M==1:
            # ln p(x1)
            true = model.models[0].log_prob(x[0])
            # ln p(x2|x1)
            true += model.models[1].log_prob(x[1], x[0])
            # ln p(x3|x2)
            true += model.models[1].log_prob(x[2], x[1])
            # ln p(x4|x3)
            true += model.models[1].log_prob(x[3], x[2])
            # ln p(x5|x4)
            true += model.models[1].log_prob(x[4], x[3])
        elif M==2:
            # ln p(x1)
            true = model.models[0].log_prob(x[0])
            # += ln p(x2|x1)
            true += model.models[1].log_prob(x[1], x[0])
            # += ln p(x3|x2,x1)
            true += model.models[2].log_prob(x[2], x[1], x[0])
            # += ln p(x4|x3,x2)
            true += model.models[2].log_prob(x[3], x[2], x[1])
            # += ln p(x5|x4,x3)
            true += model.models[2].log_prob(x[4], x[3], x[2])
        else:
            raise ValueError(f'M = {M} unsupported')
            
        model_val = model.log_prob(x)
        assert np.isclose(true, model_val), 'log joint probability incorrect'
        
    _test(M=0)
    _test(M=1)
    _test(M=2)
      
  
def _test_sample(M: int,
                 xtrain: np.ndarray,
                 seeds: List[List],
                 xtests: List[np.ndarray]
                 ):
    model = MarkovModel(K=2, M=M)
    model.fit(xtrain)
    
    for ss in range(len(seeds)):
        xsample = model.sample(N=xtests[ss].shape[0], x=seeds[ss])
        assert np.allclose(xsample, xtests[ss])
    
def test_sample():
    """
    Check constrained optimisation by sampling from
    models trained on simple sequences.
    """
    
    _test_sample(1,
                 np.asarray([0, 1, 0, 1, 0]),
                 [[0], [1]],
                 [np.asarray([0, 1, 0, 1, 0]), np.asarray([1, 0, 1, 0])]
                 )
                 
    _test_sample(2,
                 np.asarray([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
                 [[1, 0], [1, 0, 0]],
                 [np.asarray([1, 0, 0, 1]), np.asarray([1, 0, 0, 1, 0])]
                 )
  
 