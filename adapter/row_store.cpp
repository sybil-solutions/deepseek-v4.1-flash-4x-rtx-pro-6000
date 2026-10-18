// Exact immutable FP8 row retrieval; bounded direct-mapped RAM cache.
// No CUDA calls in the callback: suitable for cudaLaunchHostFunc graph nodes.
// A lookup's cache misses are deduplicated and read in parallel with Linux native AIO on the O_DIRECT
// descriptor (DSV41_ENGRAM_IO_DEPTH reads in flight, default 64; 1 reads one at a time). When the owned
// rows' scales fit in half the budget they stay resident inside it (DSV41_ENGRAM_RESIDENT_SCALES=0
// disables), so a miss costs one read instead of two.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <mutex>
#include <numeric>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

struct Store {
  int fd;
  uint64_t rows, weight_offset, scale_offset, slots, row_lo, row_hi, budget;
  uint8_t *cache;
  uint64_t *keys;
  uint8_t *resident = nullptr;
  size_t resident_size = 0;
  uint8_t *scale_map = nullptr, *scales = nullptr; // resident scales of [row_lo,row_hi)
  size_t scale_map_size = 0;
  aio_context_t aio = 0;
  int depth = 1;
  uint8_t *io_buffers = nullptr; // depth x 8 KiB, one per read in flight
  std::mutex locks[256];
  std::mutex io; // one miss batch at a time per store: the AIO context and buffers are shared
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0};
};
struct Work {
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;
  uint64_t count;
};
struct Read { // one aligned direct read that delivers `bytes` at `dest`
  uint64_t base;
  uint32_t length, delta, bytes;
  uint8_t *dest;
};

static void fail(const char *reason) {
  std::fprintf(stderr, "Engram retrieval failed: %s (errno=%d)\n", reason, errno);
  std::abort(); // Never allow a generation to continue with missing/stale rows.
}

static void read_bytes(Store *s, uint64_t offset, uint8_t *out, size_t length) {
  if (s->resident) {
    std::memcpy(out, s->resident + offset, length);
    return;
  }
  alignas(4096) uint8_t page[8192];
  const uint64_t base = offset & ~uint64_t(4095);
  const size_t delta = offset - base;
  const size_t requested = ((delta + length + 4095) / 4096) * 4096;
  ssize_t got;
  do { got = pread(s->fd, page, requested, base); } while (got < 0 && errno == EINTR);
  if (got < 0 || size_t(got) < delta + length) fail("short or failed direct read");
  std::memcpy(out, page + delta, length);
  s->reads.fetch_add(1, std::memory_order_relaxed);
}

static Read make_read(uint64_t offset, uint32_t bytes, uint8_t *dest) {
  const uint64_t base = offset & ~uint64_t(4095);
  const uint32_t delta = uint32_t(offset - base);
  return {base, (delta + bytes + 4095) / 4096 * 4096, delta, bytes, dest};
}

static void run_reads(Store *s, const std::vector<Read> &reads) {
  if (s->depth <= 1) {
    for (const Read &r : reads) read_bytes(s, r.base + r.delta, r.dest, r.bytes);
    return;
  }
  const size_t depth = size_t(s->depth);
  std::vector<iocb> cbs(depth);
  std::vector<iocb *> queue;
  queue.reserve(depth);
  std::vector<io_event> events(depth);
  std::vector<size_t> read_of(depth), free_slots(depth);
  std::iota(free_slots.begin(), free_slots.end(), size_t(0));
  size_t next = 0, inflight = 0;
  while (next < reads.size() || inflight) {
    queue.clear();
    while (inflight + queue.size() < depth && next < reads.size()) {
      const size_t slot = free_slots.back();
      free_slots.pop_back();
      iocb &cb = cbs[slot];
      std::memset(&cb, 0, sizeof cb);
      cb.aio_fildes = uint32_t(s->fd);
      cb.aio_lio_opcode = IOCB_CMD_PREAD;
      cb.aio_buf = uint64_t(uintptr_t(s->io_buffers + slot * 8192));
      cb.aio_nbytes = reads[next].length;
      cb.aio_offset = int64_t(reads[next].base);
      cb.aio_data = slot;
      read_of[slot] = next++;
      queue.push_back(&cb);
    }
    size_t sent = 0;
    while (sent < queue.size()) {
      const long r = syscall(SYS_io_submit, s->aio, long(queue.size() - sent), queue.data() + sent);
      if (r > 0) { sent += size_t(r); continue; }
      if (r < 0 && errno == EINTR) continue;
      if (r < 0 && errno == EAGAIN && inflight + sent) break; // reap, then resubmit the rest
      fail("direct read submission");
    }
    if (sent < queue.size()) { // unsent reads are a suffix of consecutive indices
      next = read_of[queue[sent]->aio_data];
      for (size_t q = sent; q < queue.size(); ++q) free_slots.push_back(size_t(queue[q]->aio_data));
    }
    inflight += sent;
    long got;
    do { got = syscall(SYS_io_getevents, s->aio, 1L, long(depth), events.data(), nullptr); }
    while (got < 0 && errno == EINTR);
    if (got <= 0) fail("direct read completion");
    for (long e = 0; e < got; ++e) {
      const size_t slot = size_t(events[e].data);
      const Read &r = reads[read_of[slot]];
      if (events[e].res < 0) errno = int(-events[e].res);
      if (events[e].res < int64_t(r.delta + r.bytes)) fail("short or failed direct read");
      std::memcpy(r.dest, s->io_buffers + slot * 8192 + r.delta, r.bytes);
      free_slots.push_back(slot);
    }
    inflight -= size_t(got);
    s->reads.fetch_add(uint64_t(got), std::memory_order_relaxed);
  }
}

static void allocate_cache(Store *s, uint64_t budget) {
  s->slots = budget / (264 + sizeof(uint64_t));
  s->cache = nullptr; s->keys = nullptr;
  if (s->slots) {
    s->cache = static_cast<uint8_t *>(mmap(nullptr, s->slots * 264,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    s->keys = static_cast<uint64_t *>(mmap(nullptr, s->slots * sizeof(uint64_t),
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (s->cache == MAP_FAILED || s->keys == MAP_FAILED) fail("cache allocation");
  }
}

static void free_cache(Store *s) {
  if (s->slots) {
    munmap(s->cache, s->slots * 264);
    munmap(s->keys, s->slots * sizeof(uint64_t));
  }
  s->slots = 0; s->cache = nullptr; s->keys = nullptr;
}

static void load_scales(Store *s) { // the owned scale rows, read once with large direct reads
  const uint64_t offset = s->scale_offset + s->row_lo * 8, length = (s->row_hi - s->row_lo) * 8;
  const uint64_t base = offset & ~uint64_t(4095), delta = offset - base;
  s->scale_map_size = (delta + length + 4095) / 4096 * 4096;
  s->scale_map = static_cast<uint8_t *>(mmap(nullptr, s->scale_map_size,
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (s->scale_map == MAP_FAILED) fail("scale allocation");
  uint64_t done = 0;
  while (done < delta + length) {
    const size_t chunk = size_t(std::min<uint64_t>(s->scale_map_size - done, uint64_t(64) << 20));
    const ssize_t got = pread(s->fd, s->scale_map + done, chunk, off_t(base + done));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) fail("short or failed scale read");
    done += uint64_t(got);
  }
  s->scales = s->scale_map + delta;
}

extern "C" Store *row_store_open(const char *path, uint64_t rows,
                                 uint64_t woff, uint64_t soff, uint64_t budget) {
  auto *s = new Store;
  const char *mode = std::getenv("OFFLOAD_MODE");
  const bool ram = mode && std::strcmp(mode, "ram") == 0;
  s->fd = open(path, O_RDONLY | O_CLOEXEC | (ram ? 0 : O_DIRECT));
  if (s->fd < 0) { delete s; return nullptr; }
  struct stat statbuf;
  if (fstat(s->fd, &statbuf) || woff > uint64_t(statbuf.st_size) ||
      soff > uint64_t(statbuf.st_size) || rows > (uint64_t(statbuf.st_size)-woff)/256 ||
      rows > (uint64_t(statbuf.st_size)-soff)/8) fail("invalid table extent");
  if (ram) {
    s->resident_size = statbuf.st_size;
    s->resident = static_cast<uint8_t *>(mmap(nullptr,s->resident_size,
        PROT_READ,MAP_SHARED | MAP_POPULATE,s->fd,0));
    if (s->resident == MAP_FAILED) fail("resident mapping");
    if (mlock(s->resident,s->resident_size)) fail("RAM mode requires memlock capability and sufficient RAM");
    budget = 0;
  }
  s->rows = rows; s->weight_offset = woff; s->scale_offset = soff;
  s->row_lo = 0; s->row_hi = rows; s->budget = budget;
  allocate_cache(s, budget);
  const char *depth = std::getenv("DSV41_ENGRAM_IO_DEPTH");
  const int want = std::max(1, std::min(1024, depth && *depth ? std::atoi(depth) : 64));
  if (!ram && want > 1) {
    if (syscall(SYS_io_setup, long(want), &s->aio) == 0) {
      s->io_buffers = static_cast<uint8_t *>(mmap(nullptr, size_t(want) * 8192,
          PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      if (s->io_buffers == MAP_FAILED) fail("read buffer allocation");
      s->depth = want;
    } else {
      std::fprintf(stderr, "Engram row store: native AIO unavailable (errno=%d); reading misses one at a time\n", errno);
      s->aio = 0;
    }
  }
  return s;
}

extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  std::vector<std::pair<uint64_t, uint64_t>> pending; // (row ID, position) of each cache miss
  for (uint64_t i = 0; i < work->count; ++i) {
    const int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) >= s->rows) fail("row ID out of bounds");
    uint8_t *weight = work->weights + i * 256, *scale = work->scales + i * 8;
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(weight, 0, 256);
      std::memset(scale, 0, 8);
      continue;
    }
    if (s->resident) {
      read_bytes(s, s->weight_offset + uint64_t(id) * 256, weight, 256);
      read_bytes(s, s->scale_offset + uint64_t(id) * 8, scale, 8);
      ++s->misses;
      continue;
    }
    if (s->slots) {
      const uint64_t slot = uint64_t(id) % s->slots;
      std::lock_guard<std::mutex> guard(s->locks[slot % 256]);
      if (s->keys[slot] == uint64_t(id) + 1) {
        std::memcpy(weight, s->cache + slot * 264, 256);
        std::memcpy(scale, s->cache + slot * 264 + 256, 8);
        ++s->hits;
        continue;
      }
    }
    pending.emplace_back(uint64_t(id), i);
  }
  if (pending.empty()) return;
  // One read per distinct row; repeats within the batch are served from it.
  std::sort(pending.begin(), pending.end());
  std::vector<uint64_t> distinct;
  for (size_t k = 0; k < pending.size(); ++k)
    if (!k || pending[k].first != pending[k - 1].first) distinct.push_back(pending[k].first);
  s->misses += distinct.size();
  s->hits += pending.size() - distinct.size();
  std::vector<uint8_t> rows(distinct.size() * 264);
  std::vector<Read> reads;
  reads.reserve(distinct.size() * (s->scales ? 1 : 2));
  for (size_t u = 0; u < distinct.size(); ++u) {
    uint8_t *row = rows.data() + u * 264;
    reads.push_back(make_read(s->weight_offset + distinct[u] * 256, 256, row));
    if (s->scales) std::memcpy(row + 256, s->scales + (distinct[u] - s->row_lo) * 8, 8);
    else reads.push_back(make_read(s->scale_offset + distinct[u] * 8, 8, row + 256));
  }
  {
    std::lock_guard<std::mutex> io(s->io);
    run_reads(s, reads);
  }
  for (size_t k = 0, u = 0; k < pending.size(); ++k) {
    if (k && pending[k].first != pending[k - 1].first) ++u;
    std::memcpy(work->weights + pending[k].second * 256, rows.data() + u * 264, 256);
    std::memcpy(work->scales + pending[k].second * 8, rows.data() + u * 264 + 256, 8);
  }
  if (s->slots) {
    for (size_t u = 0; u < distinct.size(); ++u) {
      const uint64_t slot = distinct[u] % s->slots;
      std::lock_guard<std::mutex> guard(s->locks[slot % 256]);
      std::memcpy(s->cache + slot * 264, rows.data() + u * 264, 264);
      s->keys[slot] = distinct[u] + 1;
    }
  }
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load(); out[1] = s->misses.load(); out[2] = s->reads.load();
  out[3] = s->slots * 272 + (s->scales ? (s->row_hi - s->row_lo) * 8 : 0);
}
extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) fail("invalid row ownership range");
  if (s->scale_map) { munmap(s->scale_map, s->scale_map_size); s->scale_map = s->scales = nullptr; }
  s->row_lo = lo; s->row_hi = hi;
  if (s->resident || s->hits.load() + s->misses.load()) return; // a cache already in use keeps its size
  const char *keep = std::getenv("DSV41_ENGRAM_RESIDENT_SCALES");
  const uint64_t scale_bytes = (hi - lo) * 8;
  const bool resident_scales = hi > lo && scale_bytes <= s->budget / 2 && !(keep && std::strcmp(keep, "0") == 0);
  if (resident_scales) load_scales(s);
  free_cache(s);
  allocate_cache(s, s->budget - (resident_scales ? scale_bytes : 0));
  std::fprintf(stderr, "Engram row store: rows [%llu,%llu) cache %llu slots, resident scales %llu MiB, "
               "%d reads in flight\n", (unsigned long long)lo, (unsigned long long)hi,
               (unsigned long long)s->slots, (unsigned long long)(resident_scales ? scale_bytes >> 20 : 0), s->depth);
}
extern "C" void row_store_close(Store *s) {
  if (s->resident) munmap(s->resident,s->resident_size);
  free_cache(s);
  if (s->scale_map) munmap(s->scale_map, s->scale_map_size);
  if (s->io_buffers) munmap(s->io_buffers, size_t(s->depth) * 8192);
  if (s->aio) syscall(SYS_io_destroy, s->aio);
  close(s->fd); delete s;
}
