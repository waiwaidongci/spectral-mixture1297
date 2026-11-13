import numpy as np
from .unmix_algorithm import identifiability,parse_locks
from .noise_algorithm import conditional_refits
from ..utils.fingerprint import fingerprint
ALGORITHM_VERSION="simplex-nullspace-kkt-gaussian-v3-locked"
BASE_ALGORITHM_VERSION="simplex-nullspace-kkt-gaussian-v2"

def _ordered_members(snapshot):return sorted(snapshot['endmembers'],key=lambda e:e['code'])

def resolve_locks(snapshot,locked):
    """Map stored lock records ({endmember_id, weight, ...}) to column indices.

    ``locked`` may be ``None`` or an empty list for legacy/unlocked runs. An
    endmember id that is not part of the frozen library rejects the computation.
    """
    members=_ordered_members(snapshot)
    by_id={e['id']:i for i,e in enumerate(members)}
    locked=list(locked or [])
    indices=[];weights=[];records=[]
    for entry in locked:
        if isinstance(entry,dict):
            endmember_id=entry.get('endmember_id');weight=entry.get('weight')
        else:raise ValueError("锁定配比记录格式无效")
        if type(endmember_id) is not int or endmember_id not in by_id:raise ValueError("锁定端元不属于所选光谱库")
        if by_id[endmember_id] in indices:raise ValueError("同一端元不可重复锁定")
        try:number=float(weight)
        except (TypeError,ValueError,OverflowError):raise ValueError("锁定配比必须为有限数字")
        if not np.isfinite(number) or not 0<=number<=1:raise ValueError("锁定配比必须位于0–1之间")
        indices.append(by_id[endmember_id]);weights.append(number)
        records.append({"endmember_id":endmember_id,"code":members[by_id[endmember_id]]['code'],"name":members[by_id[endmember_id]]['name'],"weight":number})
    return indices,weights,records

def analyze_unmix(snapshot,noise_sigma=.005,repeats=80,seed=2026,locked=None,lock_basis=""):
    library=snapshot['library'];members=_ordered_members(snapshot);samples=sorted(snapshot['samples'],key=lambda s:s['code'])
    if not 1<=len(samples)<=6 or len(samples)*repeats>480:raise ValueError("选择1–6个样本，样本数×重复数不得超过480")
    if len({s['code'] for s in samples})!=len(samples):raise ValueError("混合样本编号重复")
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError("根种子须为0–2147483647整数")
    lock_basis=lock_basis or ""
    if not isinstance(lock_basis,str):raise ValueError("称量依据须为文本")
    if locked and not lock_basis.strip():raise ValueError("锁定配比须注明称量依据")
    wavelengths=np.asarray(library['wavelengths'],dtype=float)
    if wavelengths.ndim!=1 or not np.isfinite(wavelengths).all() or np.any(np.diff(wavelengths)<=0):raise ValueError("波长须有限且严格递增")
    A=np.array([e['spectrum'] for e in members],dtype=float).T
    if len(wavelengths)!=A.shape[0]:raise ValueError("波长与端元波段数不匹配")
    lock_indices,lock_values,lock_records=resolve_locks(snapshot,locked)
    parse_locks(A,lock_indices,lock_values)
    supported,free_diagnostics=identifiability(A,lock_indices)
    if not supported:raise ValueError(free_diagnostics['reason'])
    streams=np.random.SeedSequence(seed).spawn(len(samples));results=[]
    for sample,child in zip(samples,streams):
        refits=conditional_refits(A,sample['observed'],noise_sigma,repeats,child,lock_indices,lock_values)
        results.append({"sample_id":sample['id'],"code":sample['code'],"name":sample['name'],"observed":sample['observed'],"spawn_key":list(child.spawn_key),"high_residual":refits['point_fit']['rmse']>max(3*noise_sigma,1e-8),**refits})
    canonical={"library":library,"endmembers":members,"samples":samples}
    method=ALGORITHM_VERSION if lock_records else BASE_ALGORITHM_VERSION
    # Unlocked runs reproduce the pre-lock fingerprint exactly (no lock keys added),
    # so a re-run of legacy data keeps the same SHA256.
    lock_payload={"lock_basis":lock_basis.strip(),"locked_weights":lock_records} if lock_records else {}
    return {"algorithm_version":method,"library":library,"endmembers":members,"samples":results,"diagnostics":free_diagnostics,"locked":bool(lock_records),"lock_basis":lock_basis.strip() if lock_records else "","locked_weights":lock_records,"noise_sigma":noise_sigma,"repeats":repeats,"seed":seed,"input_snapshot":canonical,"input_hash":fingerprint({"method":method,"snapshot":canonical,**lock_payload,"noise_sigma":noise_sigma,"repeats":repeats,"seed":seed}),"assumptions":"固定校准端元的线性混合；权重非负且和为1。称量已锁定的端元份额在点估计与全部条件重拟合中保持锁定值，仅其余端元分摊剩余份额。噪声在各波段独立同方差Gaussian，sigma由用户给定；高残差只提示失配，不是噪声或成分真实性检验。","quantile_note":"分位范围是条件扰动情景，不是配比真值的覆盖保证；锁定端元不参与条件重估。","feasibility_tolerance":1e-10,"condition_limit":1e6}
