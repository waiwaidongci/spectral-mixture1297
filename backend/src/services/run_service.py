import csv,io
import numpy as np
from redis import Redis
from fastapi import HTTPException
from ..models import SpectralLibrary,Endmember,MixtureSample,UnmixRun,ReplicationNote
from ..config.settings import REDIS_URL
from ..utils.serialize import record
from .unmix_algorithm import identifiability
from .audit_service import log
queue=Redis.from_url(REDIS_URL,decode_responses=True)
def list_runs(db):return [record(r) for r in db.query(UnmixRun).order_by(UnmixRun.id.desc())]
def get_run(db,rid):
    row=db.get(UnmixRun,rid)
    if not row:raise HTTPException(404,"解混运行不存在")
    return row

def create_snapshot(db,library,sample_ids):
    members=db.query(Endmember).filter_by(library_id=library.id).order_by(Endmember.code).all()
    samples=db.query(MixtureSample).filter(MixtureSample.id.in_(sample_ids),MixtureSample.library_id==library.id).order_by(MixtureSample.code).all()
    if len(samples)!=len(sample_ids):raise HTTPException(422,"样本重复、不存在或不属于所选光谱库")
    return {"library":record(library),"endmembers":[record(e) for e in members],"samples":[record(s) for s in samples]}

def resolve_locks(snapshot,requested):
    """Turn requested locks into canonical records against the frozen library."""
    members=sorted(snapshot['endmembers'],key=lambda e:e['code'])
    by_id={e['id']:e for e in members}
    records=[];indices=[]
    for entry in requested:
        endmember_id=entry.endmember_id if hasattr(entry,'endmember_id') else entry.get('endmember_id')
        weight=entry.weight if hasattr(entry,'weight') else entry.get('weight')
        if endmember_id not in by_id:raise HTTPException(422,"锁定端元不属于所选光谱库")
        index=[e['id'] for e in members].index(endmember_id)
        if index in indices:raise HTTPException(422,"同一端元不可重复锁定")
        try:number=float(weight)
        except (TypeError,ValueError,OverflowError):raise HTTPException(422,"锁定配比必须为有限数字")
        if not np.isfinite(number) or not 0<=number<=1:raise HTTPException(422,"锁定配比必须位于0–1之间")
        indices.append(index);records.append({"endmember_id":endmember_id,"code":by_id[endmember_id]['code'],"name":by_id[endmember_id]['name'],"weight":number})
    total=sum(r['weight'] for r in records)
    if total>1+1e-9:raise HTTPException(422,"锁定配比合计超过1，剩余份额无法分摊")
    if len(records)<len(members) and total>1-1e-9:raise HTTPException(422,"锁定配比合计已占满1，未锁定颜料无可分配剩余份额")
    if len(records)==len(members) and abs(total-1)>1e-9:raise HTTPException(422,"全部端元锁定时锁定配比合计须为1，才能给出重建与残差")
    return records,indices

def create_run(db,data,user):
    library=db.get(SpectralLibrary,data.library_id)
    if not library:raise HTTPException(404,"光谱库不存在")
    if len(set(data.sample_ids))!=len(data.sample_ids) or len(data.sample_ids)*data.repeats>480:raise HTTPException(422,"样本不可重复，样本数×重复数不得超过480")
    snapshot=create_snapshot(db,library,data.sample_ids)
    try:
        A=np.array([e['spectrum'] for e in snapshot['endmembers']]).T
        records,indices=resolve_locks(snapshot,data.locked_weights)
        basis=data.lock_basis.strip()
        if records and not basis:raise ValueError("锁定配比须注明称量依据")
        supported,details=identifiability(A,indices)
        if not supported:raise ValueError(details['reason'])
    except ValueError as e:raise HTTPException(422,str(e))
    payload=data.model_dump();payload.pop('locked_weights');payload.pop('lock_basis')
    row=UnmixRun(**payload,locked_weights=records,lock_basis=basis,input_snapshot=snapshot,owner=user.name)
    db.add(row);db.flush();log(db,user.email,"unmix.enqueue",row.id);db.commit()
    try:queue.rpush("spectral:jobs",row.id)
    except Exception:
        row.status="failed";row.error="队列不可用，请重提";db.commit();raise HTTPException(503,row.error)
    return record(row)

def export_run(db,rid):
    row=get_run(db,rid)
    if row.status!="completed" or not db.query(ReplicationNote).filter_by(run_id=rid,status="approved").first():raise HTTPException(403,"仅获批完成研究可导出")
    locks={r['code']:r['weight'] for r in (row.locked_weights or [])}
    basis=row.lock_basis or ""
    out=io.StringIO();w=csv.writer(out);w.writerow(['sample','endmember','weight','locked','lock_basis','conditional_q025','conditional_q975','rss','rmse','noise_sigma','repeats','root_seed','input_hash'])
    for s in row.result['samples']:
        for i,e in enumerate(row.result['endmembers']):
            if e['code'] in locks:
                weight=locks[e['code']];lower=upper=weight;locked='yes'
            else:
                weight=s['point_fit']['weights'][i];lower=s['lower'][i];upper=s['upper'][i];locked='no'
            w.writerow([s['code'],e['code'],weight,locked,basis,lower,upper,s['point_fit']['rss'],s['point_fit']['rmse'],row.noise_sigma,row.repeats,row.seed,row.result['input_hash']])
    return out.getvalue()
