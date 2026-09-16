#!/usr/bin/env python3
"""Minimal GGUF reader: dumps metadata KVs and tensor table."""
import struct, sys, json

GGUF_MAGIC = 0x46554747
# value types
UINT8,INT8,UINT16,INT16,UINT32,INT32,FLOAT32,BOOL,STRING,ARRAY,UINT64,INT64,FLOAT64 = range(13)
FMT = {UINT8:'<B',INT8:'<b',UINT16:'<H',INT16:'<h',UINT32:'<I',INT32:'<i',
       FLOAT32:'<f',BOOL:'<?',UINT64:'<Q',INT64:'<q',FLOAT64:'<d'}
SZ  = {UINT8:1,INT8:1,UINT16:2,INT16:2,UINT32:4,INT32:4,FLOAT32:4,BOOL:1,UINT64:8,INT64:8,FLOAT64:8}

GGML_TYPE_NAME = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
 10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',
 18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',
 26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16',34:'TQ1_0',35:'TQ2_0'}
# (type -> (block_elems, block_bytes))
BLK = {0:(1,4),1:(1,2),30:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),
 10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),20:(32,20),23:(256,136),
 34:(256,54),35:(256,66)}

class R:
    def __init__(s,b): s.b=b; s.o=0
    def raw(s,n): v=s.b[s.o:s.o+n]; s.o+=n; return v
    def scalar(s,t): v=struct.unpack_from(FMT[t],s.b,s.o)[0]; s.o+=SZ[t]; return v
    def string(s):
        n=s.scalar(UINT64); return s.raw(n).decode('utf-8',errors='replace')
    def value(s,t):
        if t==STRING: return s.string()
        if t==ARRAY:
            et=s.scalar(UINT32); n=s.scalar(UINT64)
            return [s.value(et) for _ in range(n)]
        return s.scalar(t)

def main(path):
    b=open(path,'rb').read()
    r=R(b)
    magic=r.scalar(UINT32); ver=r.scalar(UINT32)
    assert magic==GGUF_MAGIC, hex(magic)
    n_tensors=r.scalar(UINT64); n_kv=r.scalar(UINT64)
    print(f"gguf v{ver}  tensors={n_tensors}  kv={n_kv}")
    kv={}
    for _ in range(n_kv):
        k=r.string(); t=r.scalar(UINT32); kv[k]=r.value(t)
    print("\n=== METADATA ===")
    for k,v in kv.items():
        if isinstance(v,list):
            print(f"  {k}: [len={len(v)}] {v[:6]}{' ...' if len(v)>6 else ''}")
        else:
            print(f"  {k}: {v!r}")
    tensors=[]
    for _ in range(n_tensors):
        name=r.string(); nd=r.scalar(UINT32)
        dims=[r.scalar(UINT64) for _ in range(nd)]
        tt=r.scalar(UINT32); off=r.scalar(UINT64)
        tensors.append((name,dims,tt,off))
    align=kv.get('general.alignment',32)
    data_start=(r.o+align-1)//align*align
    print(f"\n=== TENSORS === (alignment={align}, data_start={data_start})")
    tot=0; bytype={}
    for name,dims,tt,off in tensors:
        n=1
        for d in dims: n*=d
        be,bb=BLK[tt]; nbytes=n//be*bb
        tot+=nbytes; bytype[GGML_TYPE_NAME[tt]]=bytype.get(GGML_TYPE_NAME[tt],0)+nbytes
        print(f"  {name:44s} {str(dims):22s} {GGML_TYPE_NAME[tt]:6s} off={off:<10d} {nbytes/1e6:8.3f} MB")
    print(f"\ntotal tensor bytes: {tot/1e6:.2f} MB  file={len(b)/1e6:.2f} MB")
    print("by type:", {k:f"{v/1e6:.2f}MB" for k,v in sorted(bytype.items())})

if __name__=='__main__': main(sys.argv[1])
