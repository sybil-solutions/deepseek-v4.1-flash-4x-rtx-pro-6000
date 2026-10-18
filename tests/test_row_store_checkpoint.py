"""Actual-checkpoint parity: row store lookups vs independent reads, both Engram layers, all TP ranks.

Usage: python3 tests/test_row_store_checkpoint.py [MODEL_PATH] [LIBRARY]
Each rank's store is opened with the production per-store budget (DSV41_CACHE_GIB / (2 x TP)) and its
ownership range; random owned rows (with repeats), unowned rows and range edges are looked up cold, then
again warm, and every byte is compared with plain buffered reads of the safetensors shard.
"""
import ctypes as C
import json
import os
from pathlib import Path
import random
import struct
import sys
import time

model = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('MODEL_PATH', '/models/DeepSeek-V4.1-Flash'))
library = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[1]/'adapter/librow_store.so')
lib = C.CDLL(library)
U, P = C.c_uint64, C.c_void_p
lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
lib.row_store_open.restype = P
lib.row_store_range.argtypes = [P, U, U]
lib.row_store_close.argtypes = [P]
lib.row_store_stats.argtypes = [P, C.POINTER(U)]

class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]

lib.row_store_lookup.argtypes = [C.POINTER(Work)]
TP, SAMPLES = 4, 2048
budget = int(float(os.getenv('DSV41_CACHE_GIB', '64')) * 2**30) // (2 * TP)
index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
summary = []
for layer in (1, 14):
    prefix = f'layers.{layer}.engram.embed.'
    filename = model/index[prefix + 'weight']
    with filename.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
    w, s = header[prefix + 'weight'], header[prefix + 'scale']
    rows = w['shape'][0]
    woff, soff = 8 + length + w['data_offsets'][0], 8 + length + s['data_offsets'][0]
    fd = os.open(filename, os.O_RDONLY)
    for rank in range(TP):
        lo, hi = rows * rank // TP, rows * (rank + 1) // TP
        store = lib.row_store_open(str(filename).encode(), rows, woff, soff, budget)
        assert store, filename
        lib.row_store_range(store, lo, hi)
        r = random.Random(1000 * layer + rank)
        owned = [r.randrange(lo, hi) for _ in range(SAMPLES)]
        ids = [lo, hi - 1] + owned + owned[:SAMPLES // 8] + [r.randrange(rows) for _ in range(SAMPLES // 8)]
        if lo:
            ids.append(lo - 1)
        if hi < rows:
            ids.append(hi)
        expect_w = b''.join(os.pread(fd, 256, woff + i * 256) if lo <= i < hi else bytes(256) for i in ids)
        expect_s = b''.join(os.pread(fd, 8, soff + i * 8) if lo <= i < hi else bytes(8) for i in ids)
        indices = (C.c_int64 * len(ids))(*ids)
        times = []
        for _ in ('cold', 'warm'):
            out_w, out_s = C.create_string_buffer(len(ids) * 256), C.create_string_buffer(len(ids) * 8)
            work = Work(store, C.addressof(indices), C.addressof(out_w), C.addressof(out_s), len(ids))
            t = time.perf_counter()
            lib.row_store_lookup(C.byref(work))
            times.append(time.perf_counter() - t)
            assert out_w.raw == expect_w and out_s.raw == expect_s, (layer, rank)
        stats = (U * 4)()
        lib.row_store_stats(store, stats)
        assert stats[3] <= budget
        summary.append(dict(layer=layer, rank=rank, rows=[lo, hi], checked=2 * len(ids), hits=stats[0],
                            misses=stats[1], reads=stats[2], cold_ms=round(times[0] * 1e3, 1),
                            warm_ms=round(times[1] * 1e3, 2)))
        print(json.dumps(summary[-1]), flush=True)
        lib.row_store_close(store)
    os.close(fd)
print(json.dumps(dict(library=Path(library).name, budget=budget, io_depth=os.getenv('DSV41_ENGRAM_IO_DEPTH', '64'),
                      checked=sum(x['checked'] for x in summary), passed=True)))
