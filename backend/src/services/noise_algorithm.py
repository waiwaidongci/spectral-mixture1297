import math
import numpy as np
from .unmix_algorithm import fit_simplex

def conditional_refits(A,y,noise_sigma=.005,repeats=80,seed=2026,locked=None,locked_weights=None):
    if type(noise_sigma) not in (int,float) or not 0<=noise_sigma<=.2 or not math.isfinite(noise_sigma):raise ValueError("sigma必须是[0,.2]内有限反射率单位，0表示无扰动")
    if type(repeats) is not int or not 20<=repeats<=120:raise ValueError("条件重复须为20–120次")
    locked=[] if locked is None else list(locked)
    locked_weights=[] if locked_weights is None else list(locked_weights)
    point=fit_simplex(A,y,locked,locked_weights);rng=np.random.default_rng(seed);observed=np.asarray(y,dtype=float)
    weights=[];rss=[]
    # The locked weights are held at their weighed values in every replicate;
    # only the free weights are re-estimated under the added noise.
    for noise in rng.normal(0,noise_sigma,size=(repeats,len(observed))):
        fitted=fit_simplex(A,observed+noise,locked,locked_weights);weights.append(fitted['weights']);rss.append(fitted['rss'])
    samples=np.asarray(weights)
    locked_indices=sorted(locked)
    interval_note="2.5/97.5%分位范围条件于固定端元与给定sigma；对观测谱加独立Gaussian扰动重拟合，不覆盖端元误差，也不保证真配比覆盖。"
    if locked_indices:interval_note+="锁定端元份额在全部条件重拟合中保持称量值不变，其分位区间退化于该值。"
    return {"point_fit":point,"replicate_weights":weights,"replicate_rss":rss,"lower":np.quantile(samples,.025,axis=0,method='linear').tolist(),"median":np.quantile(samples,.5,axis=0,method='linear').tolist(),"upper":np.quantile(samples,.975,axis=0,method='linear').tolist(),"noise_sigma":noise_sigma,"repeats":repeats,"degenerate_noise":noise_sigma==0,"interval_note":interval_note,"locked_endmember_indices":locked_indices}
