# DeepSeek-V4.1-Flash on 4 × RTX PRO 6000

Run DeepSeek-V4.1-Flash on one Linux machine with **4 × 96 GB RTX PRO 6000 Blackwell GPUs, 128 GB DDR5, and local NVMe**. Compute weights and model-native compressed KV stay on the GPUs. Native FP8 Engram rows use a bounded DDR5 cache, with exact parallel NVMe reads on misses. **DSpark is enabled. No second machine is required.**

This repository contains an end-to-end Docker launcher and measured inference results. It is still experimental: successful loading, measured speed, populated context capacity, and feature correctness are separate milestones.

## 1. Current status

| Item | Verified result or current limit |
|---|---|
| Model precision | Published mixed precision: FP4 routed experts, FP8/BF16 components, FP8 Engram. No EXL3 conversion or pruning. |
| Local NVMe inference | Running and completing real requests with DSpark block 5. |
| Eight simultaneous slots | Shared generation demonstrated in the current sweep. The admission-delay fix keeps the eighth slot usable. |
| 400k input | Earlier native runs retrieved three random codes correctly from 400k-token inputs. |
| Populated GPU KV | Earlier native NVMe test reached **2,853,120 tokens with seven active requests**. |
| 4M populated GPU KV | **Passed:** eight distinct 500k inputs completed 8,192 outputs each; peak 4,063,744 populated tokens, no prefix reuse or retractions. [Receipt](results/capacity-4000000.json). |
| Text, JSON, tools | Fresh arithmetic, schema-constrained JSON and an actual tool-call/result round trip passed. Broader quality testing remains pending. |
| Vision | Single-image inference passed at the checkpoint's full **1,024 image tokens/image**. Multi-image reliability is unresolved. |
| Video | Six-frame temporal-color test failed. Do not treat video as supported by this release. |
| Audio | No native audio workflow is qualified; no auxiliary audio model is included. |
| Local Studio | Short controller-managed chats passed. Updated 400k proxy validation and final candidate promotion are pending. |
| Reliability | Final representative one-hour soak and rollback exercise remain pending. |

**Optimization focus:** the single-machine NVMe + DDR5 setup. Remote retrieval experiments have stopped. Historical results remain below for transparency.

## 2. Start the service

Requirements: Linux x86-64, a CUDA 13-compatible NVIDIA driver, Docker Compose, NVIDIA Container Toolkit, four available 96 GB RTX PRO 6000 GPUs, and about 510 GB of checkpoint storage plus container/cache headroom. NVMe mode requires a filesystem supporting direct I/O.

```bash
git clone https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000.git
cd deepseek-v4.1-flash-4x-rtx-pro-6000
mkdir -p models state
docker compose up --build -d
docker compose logs -f
```

To reuse an existing checkpoint:

```bash
MODEL_DIRECTORY=/absolute/path/to/DeepSeek-V4.1-Flash-1017 docker compose up --build -d
```

The launcher downloads the pinned files, verifies SHA256/Git-blob hashes, compiles the adapter, starts TP4/EP4 inference, and checks arithmetic, JSON, a tool round trip and a native image before reporting ready. Model files, verification state and kernel caches persist. It does not delete other models or stop other containers.

The API binds to `127.0.0.1:8010`; `BIND_ADDRESS` can select a specific interface. The generated key is in `state/api-key` with mode 0600. An explicit `API_KEY` can instead be supplied through a private `.env` file.

```bash
API_KEY=$(cat state/api-key)
curl http://127.0.0.1:8010/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 19 + 23?"}],"chat_template_kwargs":{"thinking":false}}'
```

## 3. Configuration

| Setting | Default | Current speed-sweep candidate |
|---|---:|---:|
| `OFFLOAD_MODE` | `nvme` | `nvme` |
| `DSV41_CACHE_GIB` | 64 | 64 |
| `DSV41_ENGRAM_IO_DEPTH` | 64 | 64 |
| `DSV41_ENGRAM_RESIDENT_SCALES` | 1 | 1 |
| `CONTEXT_LENGTH` | 409600 | 524288 |
| `MEMORY_FRACTION` | 0.85 | 0.95 |
| `MAX_RUNNING_REQUESTS` | 8 | 8 |
| `MAX_TOTAL_TOKENS` | Runtime-selected | 4200000 |
| DSpark block size | 5 | 5 |
| Prefill chunk size | 2048 | 2048 |
| `min_free_slots_delay` | 1 | 1 |

The executable [measured Compose override](compose.measured.yaml) reproduces the current native NVMe/DDR5 allocation and scheduling configuration:

```bash
docker compose -f compose.yaml -f compose.measured.yaml up --build -d
```

This selects the measured configuration without changing the conservative default. DSpark block 5, bounded replay and the eight-slot admission fix are implemented in `boot.py`; the native row cache is implemented in `adapter/row_store.cpp`. The override configures 4.2M allocated tokens. The 4M populated-capacity gate passed; broader quality acceptance remains pending. B12x io_uring is not enabled by this override.

The current candidate has **not finished long-context acceptance**. To reproduce its allocation settings after reviewing that limitation:

```bash
MEMORY_FRACTION=0.95 MAX_TOTAL_TOKENS=4200000 \
CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=8 \
docker compose up --build -d
```

A 64 GiB cache budget is bounded capacity, not a promise that every byte is populated or locked. The native Engram tables occupy about 189 GiB, so they cannot reside entirely in this 128 GB host. Cache misses retrieve the original row and scale bytes; model hashing, gating, projections and tensor-parallel reduction are preserved.

Each lookup deduplicates its misses and reads them in parallel with Linux native AIO on the `O_DIRECT` descriptor, with up to `DSV41_ENGRAM_IO_DEPTH` reads in flight per lookup (`1` reads one row at a time). Docker's default seccomp profile permits native AIO, so this needs no syscall policy change and no extra package, unlike `io_uring`. When a rank's scale rows fit in half its cache budget (about 0.72 GiB of the 8 GiB per layer per rank at the default 64 GiB), they are read once at start and held inside that budget, so a miss costs one 4 KiB read instead of two. `DSV41_ENGRAM_RESIDENT_SCALES=0` keeps them on NVMe.

`OFFLOAD_MODE=ram` is an optional **unqualified full-model mode for larger-memory hosts**. It prefaults and locks the native tables, requires roughly 189 GiB plus at least 24 GiB available headroom, and aborts if `mlock` fails. Shared mappings avoid four physical copies. It is not the recommended mode for a 128 GB host.

The launcher accepts context lengths from 400,000 through the model's published 1,048,576 limit. Acceptance applies only to measured configurations. Decode/verification CUDA graphs are enabled; this runtime disables prefill CUDA graphs. The container does not change fans, power limits or CPU policy.

## 4. Current local speed sweep

**All 45 cases completed successfully.** Every request completed without reported errors; each case had a positive shared decode interval. [Machine-readable results](results/local-nvme-dspark-summary.json) · [Configuration](results/local-nvme-dspark-config.json).

Each request uses 8,192 forced output tokens. Inputs contain repeated synthetic reference text with unique prefixes. These are performance tests, **not quality scores**. The same four GPUs were capped at 275 W each, connected over PCIe without NVLink.

- **Prefill:** effective input tok/s through the last request's first token, including queueing.
- **Total decode:** actual delivered tokens across all streams in one shared decode interval.
- **Decode/request:** median request rate within that same interval. Its product with concurrency need not equal the total.
- **TTFT:** median time to first token. “No overlap” means a simultaneous decode rate was not measurable.

| Input tokens/request | C | Prefill tok/s | **Total decode tok/s** | Decode/request tok/s | TTFT s | Shared decode s |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1,943.0 | **200.9** | 200.9 | 0.26 | 40.78 |
| 512 | 2 | 4,036.4 | **347.0** | 173.5 | 0.25 | 47.15 |
| 512 | 4 | 5,251.7 | **517.0** | 130.5 | 0.39 | 60.94 |
| 512 | 6 | 4,363.0 | **644.0** | 109.0 | 0.69 | 74.18 |
| 512 | 8 | 5,016.9 | **713.5** | 88.8 | 0.81 | 87.59 |
| 2,048 | 1 | 7,010.3 | **226.8** | 226.8 | 0.29 | 36.11 |
| 2,048 | 2 | 6,720.3 | **376.4** | 188.2 | 0.61 | 41.75 |
| 2,048 | 4 | 6,736.3 | **545.1** | 135.4 | 1.06 | 56.13 |
| 2,048 | 6 | 6,747.9 | **654.3** | 111.4 | 1.36 | 72.04 |
| 2,048 | 8 | 6,772.6 | **729.1** | 91.0 | 1.66 | 86.06 |
| 8,192 | 1 | 7,337.0 | **196.4** | 196.4 | 1.12 | 41.71 |
| 8,192 | 2 | 7,251.6 | **357.1** | 178.5 | 1.84 | 45.26 |
| 8,192 | 4 | 7,292.8 | **538.8** | 135.0 | 3.11 | 57.56 |
| 8,192 | 6 | 7,311.7 | **674.9** | 114.5 | 4.21 | 69.99 |
| 8,192 | 8 | 7,311.2 | **704.5** | 89.9 | 5.32 | 84.97 |
| 32,768 | 1 | 7,438.7 | **211.5** | 211.5 | 4.40 | 38.72 |
| 32,768 | 2 | 7,432.0 | **370.8** | 185.4 | 6.77 | 43.31 |
| 32,768 | 4 | 7,406.6 | **526.6** | 134.4 | 11.36 | 57.51 |
| 32,768 | 6 | 7,478.2 | **686.1** | 113.8 | 15.62 | 67.68 |
| 32,768 | 8 | 7,420.9 | **747.7** | 95.0 | 20.20 | 82.67 |
| 65,536 | 1 | 7,346.3 | **231.6** | 231.6 | 8.92 | 35.37 |
| 65,536 | 2 | 7,320.8 | **366.4** | 183.2 | 13.59 | 43.36 |
| 65,536 | 4 | 7,313.4 | **520.2** | 135.2 | 22.70 | 58.29 |
| 65,536 | 6 | 7,330.6 | **677.8** | 112.5 | 31.59 | 68.18 |
| 65,536 | 8 | 7,335.8 | **712.3** | 89.2 | 40.49 | 83.64 |
| 131,072 | 1 | 7,064.6 | **203.4** | 203.4 | 18.55 | 40.28 |
| 131,072 | 2 | 7,115.8 | **372.2** | 186.1 | 27.74 | 42.97 |
| 131,072 | 4 | 7,095.7 | **585.6** | 148.4 | 46.51 | 54.73 |
| 131,072 | 6 | 7,082.0 | **646.0** | 108.8 | 64.93 | 71.12 |
| 131,072 | 8 | 7,089.5 | **736.3** | 91.3 | 83.39 | 83.89 |
| 200,000 | 1 | 6,802.6 | **202.2** | 202.2 | 29.40 | 40.52 |
| 200,000 | 2 | 6,824.1 | **374.2** | 187.1 | 44.12 | 43.30 |
| 200,000 | 4 | 6,835.0 | **553.2** | 138.7 | 73.71 | 57.31 |
| 200,000 | 6 | 6,847.9 | **606.2** | 105.5 | 102.70 | 72.68 |
| 200,000 | 8 | 6,864.8 | **723.6** | 93.2 | 131.55 | 86.18 |
| 400,000 | 1 | 6,081.4 | **220.2** | 220.2 | 65.77 | 37.20 |
| 400,000 | 2 | 6,162.6 | **347.6** | 173.8 | 97.56 | 47.08 |
| 400,000 | 4 | 6,165.5 | **506.1** | 126.9 | 162.73 | 63.58 |
| 400,000 | 6 | 6,173.8 | **596.4** | 101.2 | 227.11 | 77.36 |
| 400,000 | 8 | 6,156.5 | **652.5** | 82.3 | 292.54 | 95.00 |
| 500,000 | 1 | 5,844.5 | **212.4** | 212.4 | 85.55 | 38.57 |
| 500,000 | 2 | 5,789.2 | **293.8** | 146.9 | 129.93 | 53.40 |
| 500,000 | 4 | 5,854.1 | **467.3** | 116.6 | 213.94 | 67.25 |
| 500,000 | 6 | 5,843.2 | **578.2** | 97.6 | 300.29 | 81.91 |
| 500,000 | 8 | 5,834.8 | **599.7** | 75.1 | 386.09 | 105.58 |

The complete grid is 512 / 2,048 / 8,192 / 32,768 / 65,536 / 131,072 / 200,000 / 400,000 / 500,000 input tokens at C1 / C2 / C4 / C6 / C8. The 4M gate requires eight distinct 500k inputs, shared continued generation, no prefix reuse or retractions, and matching runtime occupancy.

<details>
<summary>Current burst latency and DSpark acceptance</summary>

These are client-observed gaps between positive output bursts, not individual-token latency. Percentiles pool gaps across requests; acceptance is the median engine-reported rate across completed requests. [JSON with sample counts](results/local-nvme-burst-latency.json).

| Input | C | Burst gap p50 ms | p95 ms | p99 ms | Gap samples | DSpark acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 23.59 | 29.00 | 29.33 | 1657 | 78.9% |
| 512 | 2 | 28.05 | 36.83 | 39.18 | 3209 | 82.2% |
| 512 | 4 | 37.66 | 46.47 | 50.13 | 6552 | 81.0% |
| 512 | 6 | 46.54 | 55.15 | 60.10 | 9744 | 82.6% |
| 512 | 8 | 55.14 | 66.60 | 72.29 | 13100 | 79.2% |
| 2,048 | 1 | 22.96 | 25.51 | 27.62 | 1556 | 85.3% |
| 2,048 | 2 | 27.04 | 31.69 | 33.48 | 3131 | 84.8% |
| 2,048 | 4 | 36.85 | 43.03 | 46.73 | 6381 | 82.2% |
| 2,048 | 6 | 44.74 | 52.89 | 57.77 | 9864 | 81.5% |
| 2,048 | 8 | 54.41 | 65.06 | 72.02 | 13027 | 80.9% |
| 8,192 | 1 | 23.16 | 28.09 | 29.07 | 1742 | 74.1% |
| 8,192 | 2 | 28.21 | 32.96 | 34.93 | 3183 | 83.0% |
| 8,192 | 4 | 37.07 | 42.93 | 45.84 | 6467 | 80.9% |
| 8,192 | 6 | 45.20 | 53.11 | 56.21 | 9628 | 84.3% |
| 8,192 | 8 | 54.85 | 65.05 | 69.49 | 13290 | 80.2% |
| 32,768 | 1 | 23.14 | 26.46 | 28.53 | 1643 | 79.7% |
| 32,768 | 2 | 28.31 | 31.78 | 33.89 | 3103 | 85.7% |
| 32,768 | 4 | 37.61 | 43.53 | 45.98 | 6527 | 81.4% |
| 32,768 | 6 | 44.67 | 50.81 | 54.39 | 9507 | 83.0% |
| 32,768 | 8 | 54.37 | 61.83 | 66.51 | 12798 | 84.3% |
| 65,536 | 1 | 23.29 | 24.73 | 27.00 | 1511 | 88.4% |
| 65,536 | 2 | 28.81 | 32.02 | 33.51 | 3083 | 86.4% |

</details>

Host telemetry is being recorded. Swappiness was 180, with some inference-process pages already swapped after loading; short samples did not show growing inference swap. This does not establish a causal bottleneck. Cache behavior, host memory, attention workspace and B12x are candidates for controlled local tuning after the baseline sweep.

`CHUNKED_PREFILL_SIZE` is configurable through Docker Compose; the default remains **2,048**. Larger values are experimental until long-context, eight-request and vision tests pass. A previous 4,096-token chunk configuration failed near 399k input, so configuration availability is not a stability claim.

The next storage candidate reuses the [upstream B12x reader](https://github.com/local-inference-lab/b12x/blob/01ac763bef7503d934c4c2874bfd005799ca24a1/b12x/loader/_ple_reader.c). Its configurable native byte reader can represent the checkpoint's 256-byte FP8 rows plus 8-byte E8M0 scale rows. The high-level PLE wrapper has different quantization and graph-capture contracts, so it cannot simply replace DeepSeek's embedding module. Integration must retain native row/scaling semantics, the DDR5 cache and DSpark graph replay, then pass parity and matched inference sweeps.

### Cold Engram rows: parallel direct reads

The sweep above repeats synthetic reference text, so after the first chunk nearly every Engram row is a DDR5 cache hit. New text behaves differently. Each token looks up 48 rows (2 layers × 3 n-gram orders × 8 hash heads). About half of a new code prompt's rows, and nearly all of a random-word prompt's, are not cached yet, and a larger cache cannot help with rows that have never been read. Each miss used to be two serial 4 KiB direct reads inside the CUDA host callback, so the four ranks kept four reads in flight (about 67k reads/s) while the GPUs waited. The same NVMe sustains 0.8–1.3M random 4 KiB reads/s with 16–64 reads in flight per process.

Misses are now deduplicated and read in parallel, and scale rows stay resident (see [Configuration](#3-configuration)). Measured on 4 × RTX PRO 6000 at 350 W with a Samsung 990 PRO 2TB and 1,048,576 context. Each arm was freshly restarted and given the same request sequence. [Receipt](results/engram-parallel-reads.json).

| Prompt | Engram rows | Before | After | Speedup |
|---|---|---:|---:|---:|
| 25.7k tokens of new code | cold | 999 tok/s | 3,925 tok/s | 3.9× |
| 24.8k tokens of different code | cold | 1,039 tok/s | 5,205 tok/s | 5.0× |
| 36.6k random words | cold | 682 tok/s | 5,125 tok/s | 7.5× |
| 100k tokens of new code | cold | 1,052 tok/s | 7,269 tok/s | 6.9× |
| The same four prompts again | warm | 8,562–9,306 tok/s | 8,897–9,672 tok/s | 1.0× |

- **GPU utilization:** during cold prefill it rose from 0–25% to 44–72%. The first cold prompt after a restart is the slowest, because new rows also land in untouched cache pages.
- **Byte parity:** 11 fixture cases (including `OFFLOAD_MODE=ram`), plus 41,016 checkpoint rows across both Engram layers and all four TP ranks, cold and warm.
- **Lookup speed:** a single-process cold lookup of about 2,100 missing rows per store took 200–254 ms before and 8–9 ms after.
- **Decode:** in a three-prompt greedy check, single-stream decode was also faster (prose 72 → 95 tok/s). The engine is not bitwise deterministic run to run, so treat decode as indicative.

## 5. Experiment history

### Capacity and runtime experiments

| Experiment | Result | Decision / limit |
|---|---|---|
| Native NVMe, six × 400k | 2,406,912 populated KV; 501 total decode tok/s; 11.69 s shared decode | Passed that capacity; earlier 0.85 configuration |
| Native NVMe, six × 350k | 2,143,232 populated KV; 75.64 s shared generation | Passed that capacity; separate earlier workload |
| Native NVMe, seven × 400k, 8,192 output | **2,853,120 populated KV; 6,353 prefill tok/s; 721 total decode tok/s; 104/request; 74.48 s shared** | Passed; [receipt](results/capacity-2800000.json) |
| Uncapped 0.93 memory fraction | Allocated 7.62M logical slots; first generation failed on temporary attention-buffer allocation | Failed; reserved slots were not usable capacity |
| 0.93, 4.2M cap, ten-slot remote trial | Nine requests overlapped; tenth queued; at least 3,656,704 populated tokens observed | Did not prove 4M; not the selected local setup |
| Default DSpark admission delay | Left the final request slot idle until another request completed | Set `--min-free-slots-delay 1`; current local C8 overlap is measured above |
| 0.95, 4.2M cap, eight local slots | Eight × 500k; 4,063,744 populated tokens; 599.7 total decode tok/s; 105.58 s shared generation | Capacity passed; [receipt](results/capacity-4000000.json). Long-context answer quality remains unqualified. |
| Larger 4096 prefill chunk / 0.88 trial | Failed near 399k input despite successful loading | Keep 2048 prefill chunks for this baseline |
| B12x attention kernels | Standalone correctness/shape tests passed | Not yet accepted in full-model serving; default still uses the documented SM120 compatibility path |
| Latest B12x NVMe reader | Upstream `01ac763` adds bounded `io_uring`, fixed buffers, page deduplication and coalesced reads | Priority local candidate; no full-model speed result yet |
| B12x reader prerequisites in current image | `io_uring_setup` returned EPERM; `liburing` development package is absent | Candidate needs the dependency and a scoped container syscall policy before testing |
| B12x candidate prerequisite probe | `io_uring_setup` succeeds with only three additional syscalls allowed; candidate image built with `liburing` | Running baseline unchanged |
| B12x native reader bridge | **12,288 byte-parity checks passed** at queue depths 1/8/64, four TP ranks, native 256-byte rows plus 8-byte scales | [Receipt](results/b12x-uring-bridge-parity.json); synthetic CPU fixtures only, no graph/model acceptance yet |
| B12x bounded DDR5 cache adapter | **36,864 native-layout byte checks passed**, including cache collisions, warm hits, TP ownership and concurrent callers | [Receipt](results/b12x-uring-cache-parity.json); CPU fixtures, no model throughput claim |
| Assembled B12x adapter initialization | Native bytes and warm-cache behavior passed through the patched adapter constructor | [Receipt](results/b12x-uring-adapter-init.json); GPU callback replay passed below; full inference remains pending |
| Actual-checkpoint B12x reader | 8,192 row checks passed across both Engram layers and all four TP ranks | [Receipt](results/b12x-uring-checkpoint-parity.json); original FP8 values and E8M0 scales versus independent reads |
| B12x GPU callback and graph replay | 320 graph replays and 994,560 synthetic row checks passed | [Receipt](results/b12x-uring-graph-replay.json); initial attempt hit OOM beside the occupied baseline, isolated rerun passed. Full-model quality and speed remain pending |
| Parallel native-AIO misses with resident scales | New-text cold prefill 3.9–7.5× faster; warm unchanged; 41,016 checkpoint rows byte-identical | Default (`DSV41_ENGRAM_IO_DEPTH=64`); no io_uring, extra package or seccomp change; [receipt](results/engram-parallel-reads.json) |
| Native fully resident DDR5 | About 189 GiB needed before runtime headroom | Does not fit 128 GB host; not tested as a fabricated “RAM-only” result |

### Historical remote comparison — stopped

These three cases used the same DSpark5 / 0.95 / 4.2M-cap / eight-slot / 8,192-output settings as the local sweep. They are single-wave synthetic measurements, not a complete storage comparison. Remote storage is not part of the current deployment. [Raw summaries](results/remote-historical-summary.json).

| Input | C | Remote prefill tok/s | Remote total decode tok/s | Local prefill tok/s | Local total decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 2,324.8 | 116.8 | 1,943.0 | 200.9 |
| 512 | 2 | 2,122.6 | 209.5 | 4,036.4 | 347.0 |
| 512 | 4 | 3,117.6 | 306.8 | 5,251.7 | 517.0 |

The historical remote client also underwent a TCP_NODELAY microbenchmark. These are lookup latencies, **not inference throughput**, and are not part of the selected local path.

| Rows in lookup batch | Before median ms | After median ms |
|---:|---:|---:|
| 24 | 42.05 | 0.5615 |
| 120 | 43.93 | 0.8323 |
| 1024 | 11.82 | 3.448 |

### Experimental compressed Engram — not deployed

| Check | Observed result | What it establishes |
|---|---|---|
| Full native table calibration | Layer maxima 11 and 24; no invalid blocks | Quantizer calibration, not model quality |
| NVFP4 packed format | 128 value bytes + 16 scale bytes per row; about 103 GiB total | Storage estimate before runtime headroom |
| Reference packing | Exact byte match on 8,192 sampled rows | Converter sample parity |
| Sample reconstruction | Relative RMSE 9.48% / 9.66%; mean cosine about 0.9955 | Tensor error only; no task-quality acceptance |
| CPU packed-row retrieval | 16,384 checks, including 8,192 locked-RAM checks | Lookup and TP ownership parity |
| GPU dequantization | Exact BF16 comparison; 200 CUDA-graph replays on RTX 3090 | Kernel-only result, not Blackwell/model acceptance |
| Full conversion | First table completed; second stopped when focus returned to local native optimization | Incomplete candidate; original checkpoint preserved |
| Combined CUDA host-callback test | Did not execute because the test could not locate its CUDA runtime library | No combined-path acceptance |
| Full-model compressed inference | Not run | No speed or quality claim |

### Earlier speed baseline

The following 40 waves used the earlier allocation/admission settings and shorter output budget (usually 1,024 tokens). **Do not treat differences from the current 8,192-token sweep as an isolated optimization gain.** Requested C8 often lacked eight-stream overlap. Every wave completed, but absent overlap is not a simultaneous throughput result. [Summaries](results/summary.json) · [Shared-window evidence](results/common-window.json).

<details>
<summary>All 40 earlier measurements</summary>

| Input | C | Prefill tok/s | Total decode tok/s | Decode/request tok/s | Shared decode s |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1,070.0 | 152.3 | 152.3 | 6.72 |
| 512 | 2 | 1,848.8 | 259.4 | 129.7 | 7.60 |
| 512 | 4 | 2,976.9 | 409.6 | 102.8 | 9.82 |
| 512 | 6 | 3,216.8 | 564.4 | 94.7 | 10.69 |
| 512 | 8 | 3,477.0 | 582.0 | 72.9 | 13.12 |
| 2,048 | 1 | 5,875.1 | 160.3 | 160.3 | 6.38 |
| 2,048 | 2 | 6,205.0 | 323.8 | 161.9 | 6.25 |
| 2,048 | 4 | 6,227.1 | 491.5 | 121.3 | 7.95 |
| 2,048 | 6 | 6,359.5 | 594.5 | 100.8 | 10.02 |
| 2,048 | 8 | 1,169.1 | Not measured | Not measured | 0.00 |
| 8,192 | 1 | 7,457.5 | 157.9 | 157.9 | 6.48 |
| 8,192 | 2 | 7,614.8 | 312.9 | 156.4 | 6.43 |
| 8,192 | 4 | 7,500.6 | 449.2 | 112.4 | 9.03 |
| 8,192 | 6 | 7,580.9 | 646.4 | 107.7 | 9.10 |
| 8,192 | 8 | 3,166.7 | Not measured | Not measured | 0.00 |
| 32,768 | 1 | 7,816.6 | 187.4 | 187.4 | 5.46 |
| 32,768 | 2 | 7,771.2 | 321.5 | 160.7 | 6.24 |
| 32,768 | 4 | 7,704.3 | 500.8 | 123.6 | 7.67 |
| 32,768 | 6 | 7,614.2 | 589.7 | 98.6 | 9.82 |
| 32,768 | 8 | 5,947.3 | Not measured | Not measured | 0.00 |
| 65,536 | 1 | 7,447.3 | 169.2 | 169.2 | 6.05 |
| 65,536 | 2 | 7,449.8 | 309.9 | 154.9 | 6.54 |
| 65,536 | 4 | 7,427.8 | 502.3 | 125.8 | 7.86 |
| 65,536 | 6 | 7,402.7 | 565.6 | 96.0 | 10.17 |
| 65,536 | 8 | 6,462.1 | Not measured | Not measured | 0.00 |
| 131,072 | 1 | 7,151.6 | 174.1 | 174.1 | 5.88 |
| 131,072 | 2 | 7,133.3 | 305.9 | 153.0 | 6.60 |
| 131,072 | 4 | 7,137.9 | 492.3 | 120.9 | 7.72 |
| 131,072 | 6 | 7,128.3 | 617.8 | 102.4 | 9.44 |
| 131,072 | 8 | 6,663.4 | Not measured | Not measured | 0.00 |
| 200,000 | 1 | 6,860.9 | 190.7 | 190.7 | 5.36 |
| 200,000 | 2 | 6,906.8 | 309.9 | 154.9 | 6.46 |
| 200,000 | 4 | 6,861.2 | 473.6 | 119.1 | 8.41 |
| 200,000 | 6 | 6,901.2 | 586.8 | 97.8 | 10.24 |
| 200,000 | 8 | 6,549.4 | Not measured | Not measured | 0.00 |
| 400,000 | 1 | 6,128.5 | 166.7 | 166.7 | 6.14 |
| 400,000 | 2 | 6,171.3 | 248.7 | 124.4 | 8.13 |
| 400,000 | 4 | 6,168.7 | 425.0 | 106.3 | 9.54 |
| 400,000 | 6 | 6,178.2 | 500.6 | 83.4 | 11.69 |
| 400,000 | 8 | 6,045.8 | Not measured | Not measured | 0.00 |

</details>

Earlier short windows counted emitted tokens 129–641 independently for each request. These are not aggregate simultaneous rates.

| Requested C | Earlier individual decode tok/s |
|---:|---:|
| 1 | 162.6 |
| 2 | 135.1–148.0 |
| 4 | 112.6–126.4 |
| 8 | 83.6–96.6 |

## 6. Feature and controller acceptance

| Test | Result |
|---|---|
| 400k retrieval | Three random codes found; first token at 60.59 s in an earlier native run |
| Two-turn direct 400k chat | Both turns retrieved the codes and stopped naturally; 62.09 s then 1.92 s, with prefix reuse on continuation |
| Single 3024 × 588 image | Correct description; exactly 1,024 image tokens |
| Individual red/blue/green and three-image order tests | Passed |
| Six grouped duplicate-color images | Failed; alternating colors and explicitly numbered image variants passed narrower checks |
| Six-frame temporal-color video | Failed to recover the correct red → blue → green sequence |
| Arithmetic, JSON schema, tool round trip | Passed through a controller-managed Docker launch |
| Controller 400k chat with old 409600 recipe | Rejected before inference: heuristic estimated 466,663 against a 368,640 soft ceiling |
| Controller with proposed 524288 context | Source analysis indicates the earlier fixture fits; live retest pending |

Local Studio's exact display name is **`Deepseek-v4.1-Flash`**; its served model ID is **`deepseek-v4.1-flash`**. The tested controller required recipe port and `SERVER_PORT=8000`; default Compose uses 8010. A healthy direct API does not prove controller or visible UI acceptance. Keep recipe context, launch context and proxy port consistent.

## 7. Reproduce benchmarks and stop

```bash
docker compose exec deepseek python3 /opt/dsv41/boot.py smoke
docker compose exec -e PREFILL_SIZES=512,2048,8192,32768,65536,131072,200000,400000,500000 \
  -e CONCURRENCIES=1,2,4,6,8 -e OUTPUT_TOKENS=8192 -e IGNORE_EOS=1 \
  deepseek python3 /opt/dsv41/benchmarks/matrix.py
docker compose down
```

Each matrix saves raw token events, JSON summaries and `TABLE.md` under `state/matrix-<timestamp>`. `benchmarks/common_window.py` can recompute total decode from saved events. Preserve the exact launch manifest, prompts, output budget and cache state when comparing candidates. The public snapshots above do not imply all pending cases completed.

CPU adapter checks:

```bash
docker compose run --rm --entrypoint python3 deepseek /opt/dsv41/tests/test_row_store.py
docker compose run --rm -e OFFLOAD_MODE=ram --entrypoint python3 deepseek /opt/dsv41/tests/test_row_store.py
docker compose run --rm --entrypoint python3 deepseek /opt/dsv41/tests/test_row_store_checkpoint.py
```

The checkpoint check compares cold and warm lookups for both Engram layers and all four TP ranks against plain reads of the shards.

Shutdown preserves checkpoint, state and kernel caches. Restore an earlier service using its saved container configuration; a complete rollback exercise for the final candidate is still pending.

## 8. Provenance

| Component | Pinned source / attribution |
|---|---|
| Model | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Config, tensor index, tokenizer and inference code are identical to the earlier [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`; all 48 weight shards changed. Measurements below that predate 2026-10-17 used the earlier revision. |
| SGLang base | `lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5` |
| SM120 compatibility | Existing DeepGEMM planner initialization; sparse-prefill sources split into supported 64-token pages with independent scratch buffers |
| Related work | [B12x](https://github.com/local-inference-lab/b12x), [DeepSpark](https://github.com/brandonmmusic-max/deepspark), [earlier DS4 SM120 recipe](https://github.com/jacklarmer/deepseek-v4-flash-0731-sm120) |

Weights are downloaded at runtime and retain their upstream license. They are not included in the container. The adapted SGLang file retains Apache-2.0 licensing; see [NOTICE](NOTICE) for attribution.

## Study protocol and upstream skills

The [study protocol](STUDY-PROTOCOL.md) records pinned Local Inference Lab sources, installed skills, bounded experiment budgets, correctness gates and the evidence audit. Performance winners require repeatable matched measurements and quality acceptance.
