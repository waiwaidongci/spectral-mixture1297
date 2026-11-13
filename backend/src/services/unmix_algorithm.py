"""Global simplex-constrained least squares by exhaustive nonempty supports (2..6 endmembers).

A subset of endmember weights may be pinned to analyst-supplied values (``locked``).
Pinned weights leave the unknown vector, and the remaining mass ``1-sum(locked)`` is
split among the free endmembers. Identifiability is judged on those remaining degrees
of freedom instead of the full augmented endmember matrix; with every endmember
locked there are no free parameters, yet reconstruction and residuals are still
reported from the locked values.
"""
from itertools import combinations
import numpy as np
CONDITION_LIMIT=1e6
FEASIBILITY_TOL=1e-10
LOCK_SUM_TOL=1e-9

def _as_matrix(A):
    try:A=np.asarray(A,dtype=float)
    except (TypeError,ValueError,OverflowError):raise ValueError("端元矩阵必须为有限数字")
    if A.ndim!=2 or not 2<=A.shape[1]<=6 or not 2<=A.shape[0]<=64:raise ValueError("矩阵要求2–64波段、2–6端元，列为端元")
    if not np.isfinite(A).all() or np.max(np.abs(A))>1000:raise ValueError("端元谱包含非有限或超范围数值")
    return A

def _value_condition(matrix):
    singular=np.linalg.svd(matrix,compute_uv=False)
    rank=int(np.sum(singular>max(matrix.shape)*np.finfo(float).eps*singular[0]))
    raw=float(singular[0]/singular[-1]) if singular[-1]>0 else float('inf')
    return rank,(raw if np.isfinite(raw) else None),raw

def matrix_diagnostics(A):
    A=_as_matrix(A)
    augmented=np.vstack([A,np.ones(A.shape[1])])
    rank,condition,raw=_value_condition(augmented)
    supported=rank==A.shape[1] and raw<=CONDITION_LIMIT
    reason="增广[A;1]满列秩，支持固定端元下的唯一配比解释" if supported else "增广[A;1]秩亏，配比可能非唯一，拒绝唯一成分解释" if rank<A.shape[1] else "增广[A;1]病态，条件数超过1e6，拒绝不稳定配比解释"
    return A,{"augmented_rank":rank,"condition_number":condition,"condition_limit":CONDITION_LIMIT,"supported":supported,"reason":reason}

def _lock_indices(A,locked):
    A=_as_matrix(A);m=A.shape[1]
    if locked is None:return []
    if not isinstance(locked,(list,tuple)):raise ValueError("锁定端元须为列表")
    indices=[]
    for key in locked:
        if type(key) is not int:raise ValueError("锁定端元编号无效")
        if not 0<=key<m:raise ValueError("锁定端元不属于所选光谱库")
        if key in indices:raise ValueError("同一端元不可重复锁定")
        indices.append(key)
    return indices

def parse_locks(A,locked=None,locked_weights=None):
    """Validate a lock request; returns ``(locked_index_list, locked_vector)``."""
    A=_as_matrix(A);m=A.shape[1]
    indices=_lock_indices(A,locked)
    if locked_weights is None:locked_weights=[]
    if not isinstance(locked_weights,(list,tuple)):raise ValueError("锁定配比须为数值列表")
    if len(indices)!=len(locked_weights):raise ValueError("锁定端元与锁定配比数量不一致")
    vector=np.zeros(m)
    for index,value in zip(indices,locked_weights):
        if type(value) not in (int,float):raise ValueError("锁定配比必须为有限数字")
        number=float(value)
        if not np.isfinite(number) or not 0<=number<=1:raise ValueError("锁定配比必须位于0–1之间")
        vector[index]=number
    total=float(vector.sum())
    free_count=m-len(indices)
    if total>1+LOCK_SUM_TOL:raise ValueError("锁定配比合计超过1，剩余份额无法分摊")
    if free_count>=1 and total>1-LOCK_SUM_TOL:raise ValueError("锁定配比合计已占满1，未锁定颜料无可分配剩余份额")
    if free_count==0 and abs(total-1)>LOCK_SUM_TOL:raise ValueError("全部端元锁定时锁定配比合计须为1，才能给出重建与残差")
    return indices,vector

def _free_identifiability(A_free,free_count):
    """Identifiability of the free sub-problem on the remaining simplex."""
    details={"free_endmember_count":int(free_count),"augmented_rank":0,"condition_number":None,"condition_limit":CONDITION_LIMIT,"supported":True,"identifiable":True}
    if free_count==0:
        details.update(augmented_rank=0,basis="locked-only",
            reason="全部端元已锁定，剩余自由度为零；按锁定值直接重建，不估计自由配比")
        return True,details
    if free_count==1:
        # One free weight is fixed by the sum constraint alone; no augmented rank needed.
        details.update(augmented_rank=1,condition_number=1.,basis="simplex-complement",
            reason="仅余1个自由端元，其份额由和约束唯一确定，无需增广满列秩")
        return True,details
    augmented=np.vstack([A_free,np.ones(free_count)])
    rank,condition,raw=_value_condition(augmented)
    supported=rank==free_count and raw<=CONDITION_LIMIT
    reason="自由端元增广[A_S;1]满列秩，可在剩余自由度上唯一解释配比" if supported else "自由端元增广[A_S;1]秩亏，剩余自由度上配比不唯一，拒绝唯一成分解释" if rank<free_count else "自由端元增广[A_S;1]病态，条件数超过1e6，拒绝不稳定配比解释"
    details.update(augmented_rank=rank,condition_number=condition,identifiable=supported,supported=supported,basis="augmented-free",reason=reason)
    return supported,details

def identifiability(A,locked=None):
    """Judge identifiability from the free (unlocked) degrees of freedom."""
    A=_as_matrix(A);indices=_lock_indices(A,locked)
    free=[i for i in range(A.shape[1]) if i not in indices]
    return _free_identifiability(A[:,free],len(free))

def _fit(A,y,index_map,lock_vector):
    m=A.shape[1];free=[i for i in range(m) if i not in index_map];free_count=len(free)
    supported,free_diagnostics=_free_identifiability(A[:,free],free_count)
    if not supported:raise ValueError(free_diagnostics["reason"])
    locked_flags=[i for i in range(m) if i in index_map]
    if free_count==0:
        weights=lock_vector.copy();reconstructed=A@weights;residual=y-reconstructed;rss=float(np.sum(residual**2))
        return {"weights":weights.tolist(),"reconstructed":reconstructed.tolist(),"residual":residual.tolist(),"rss":rss,"rmse":float(np.sqrt(rss/len(y))),"active_count":int(np.sum(weights>1e-8)),"equality_error":abs(float(weights.sum())-1),"active_stationarity_error":0.,"inactive_slack_min":None,"kkt_tolerance":0.,"supports_solved":0,"locked":True,"locked_endmember_indices":locked_flags,"free_identifiability":free_diagnostics,"diagnostics":free_diagnostics}
    remainder=1-float(lock_vector.sum());target=y-A@lock_vector;best=None;tested=0
    for size in range(1,free_count+1):
        for support in combinations(free,size):
            S=A[:,support]
            # Equality-constrained KKT solution via an orthonormal null space.
            # Solve only for the free mass against the residual target left after
            # subtracting the locked endmembers' weighed contribution.
            # Avoid forming A.T@A, which squares the condition number.
            origin=np.full(size,remainder/size)
            if size==1:local=origin
            else:
                _,_,vh=np.linalg.svd(np.ones((1,size)),full_matrices=True)
                basis=vh[1:].T
                try:coordinates=np.linalg.lstsq(S@basis,target-S@origin,rcond=None)[0]
                except np.linalg.LinAlgError:continue
                local=origin+basis@coordinates
            tested+=1;weights=lock_vector.copy();weights[list(support)]=local
            if weights.min() < -FEASIBILITY_TOL or abs(weights.sum()-1)>1e-8:continue
            if not index_map:
                # Unlocked path unchanged from the pre-lock algorithm (bit-for-bit).
                weights=np.maximum(weights,0.);weights/=weights.sum()
            else:
                # Clip and renormalise only the free coordinates; the weighed locks stay
                # bit-exact so every replicate reports the analyst's values unchanged.
                free_clipped=np.maximum(weights[list(support)],0.);free_sum=float(free_clipped.sum())
                weights[list(support)]=free_clipped*(remainder/free_sum) if free_sum>0 else remainder/size
            reconstructed=A@weights;rss=float(np.sum((reconstructed-y)**2))
            if best is None or rss<best['rss']:best={"weights":weights,"rss":rss,"support":support}
    if best is None:raise ValueError("没有数值可行的非负和为1解")
    weights=best['weights'];reconstructed=A@weights
    multiplier=float(-np.mean(2*A[:,best['support']].T@(reconstructed-y)))
    gradient=2*A.T@(reconstructed-y)+multiplier
    free_mask=np.zeros(m,dtype=bool);free_mask[free]=True;active_normal=weights>1e-8
    active=active_normal&free_mask
    scale=1+np.linalg.norm(A)**2+np.linalg.norm(A)*np.linalg.norm(y);tol=1e-8*scale
    active_error=float(np.max(np.abs(gradient[active]))) if active.any() else 0.
    inactive_min=float(np.min(gradient[free_mask&~active_normal])) if (free_mask&~active_normal).any() else None
    # Locked weights are deliberate constraints, not estimated stationarity points;
    # only the free coordinates are required to satisfy the KKT conditions.
    if active_error>tol or (inactive_min is not None and inactive_min < -tol):raise ValueError("完整KKT最优性校验未通过；不输出不可靠配比")
    residual=y-reconstructed
    return {"weights":weights.tolist(),"reconstructed":reconstructed.tolist(),"residual":residual.tolist(),"rss":float(best['rss']),"rmse":float(np.sqrt(best['rss']/len(y))),"active_count":int(np.sum(weights>1e-8)),"equality_error":abs(float(weights.sum())-1),"active_stationarity_error":active_error,"inactive_slack_min":inactive_min,"kkt_tolerance":float(tol),"supports_solved":tested,"locked":bool(index_map),"locked_endmember_indices":locked_flags,"free_identifiability":free_diagnostics,"diagnostics":free_diagnostics}

def fit_simplex(A,y,locked=None,locked_weights=None):
    A=_as_matrix(A)
    try:y=np.asarray(y,dtype=float)
    except (TypeError,ValueError,OverflowError):raise ValueError("混合谱必须为有限数字")
    if y.shape!=(A.shape[0],) or not np.isfinite(y).all() or np.max(np.abs(y))>1000:raise ValueError("混合谱长度不匹配或包含非有限/超范围数值")
    indices,lock_vector=parse_locks(A,locked,locked_weights)
    return _fit(A,y,set(indices),lock_vector)
