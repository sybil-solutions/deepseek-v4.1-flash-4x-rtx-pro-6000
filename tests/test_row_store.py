"""CPU parity, eviction, ownership, read depth and simultaneous callers on an unaligned fixture."""
import concurrent.futures
import ctypes as C
import json
import os
from pathlib import Path
import random
import tempfile

lib = C.CDLL(str(Path(__file__).resolve().parents[1]/'adapter/librow_store.so'))
U = C.c_uint64
P = C.c_void_p
lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
lib.row_store_open.restype = P
lib.row_store_range.argtypes = [P, U, U]
lib.row_store_close.argtypes = [P]
lib.row_store_stats.argtypes = [P, C.POINTER(U)]

class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]

lib.row_store_lookup.argtypes = [C.POINTER(Work)]
rows = 4099
rng = random.Random(413)
weights, scales = rng.randbytes(rows * 256), rng.randbytes(rows * 8)
offset = 777
cases = [(depth, '1', budget, (0, rows)) for depth in ('1', '64') for budget in (0, 272 * 17, 272 * rows)]
cases += [(depth, keep, 272 * rows, (1000, 3500)) for depth in ('1', '64') for keep in ('1', '0')]
cases += [('64', '1', 272 * 17, (1000, 3500))]
with tempfile.NamedTemporaryFile() as f:
    f.write(bytes(offset) + weights + scales)
    f.flush()
    for depth, keep, budget, (lo, hi) in cases:
        os.environ['DSV41_ENGRAM_IO_DEPTH'], os.environ['DSV41_ENGRAM_RESIDENT_SCALES'] = depth, keep
        store = lib.row_store_open(f.name.encode(), rows, offset, offset + len(weights), budget)
        assert store
        if (lo, hi) != (0, rows):
            lib.row_store_range(store, lo, hi)
        def check(seed):
            r = random.Random(seed)
            ids = [0, rows - 1, 15, 16, 17, 0, 17, lo, hi - 1] + [r.randrange(rows) for _ in range(1000)]
            indices = (C.c_int64 * len(ids))(*ids)
            w, s = C.create_string_buffer(len(ids) * 256), C.create_string_buffer(len(ids) * 8)
            work = Work(store, C.addressof(indices), C.addressof(w), C.addressof(s), len(ids))
            lib.row_store_lookup(C.byref(work))
            assert w.raw == b''.join(weights[i*256:(i+1)*256] if lo <= i < hi else bytes(256) for i in ids)
            assert s.raw == b''.join(scales[i*8:(i+1)*8] if lo <= i < hi else bytes(8) for i in ids)
            return sum(lo <= i < hi for i in ids)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            checked = sum(pool.map(check, range(12)))
        stats = (U * 4)()
        lib.row_store_stats(store, stats)
        assert stats[0] + stats[1] == checked
        assert stats[3] <= budget
        print(json.dumps(dict(io_depth=int(depth), resident_scales=keep == '1', budget=budget, owned=[lo, hi],
                              checked=checked, hits=stats[0], misses=stats[1], reads=stats[2], passed=True)))
        lib.row_store_close(store)
