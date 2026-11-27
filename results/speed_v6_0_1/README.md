# Speed: base Gemma 4 12B vs MoSV v6.0.1

Setups: dual (Vulkan0 + Vulkan1, layer split), then single (Vulkan1); RX 9070 XT. One model at a time. llama-server flags `-ngl 99 -sm layer -fa on -c 8192 -np 1 --jinja --no-webui -ctk q8_0 -ctv q8_0`. Sampling `{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0, "top_n_sigma": -1.0, "mirostat": 0, "seed": 7}`. Thinking off, no prompt cache, warm-up not counted. Fixed runs: 512 new tokens each (end ignored), order base, MoSV, MoSV, base. Raw logs: `*_server.log`, `*_bench.log`. Every request: `speed.json`. Answers: `outputs.md`.

Natural answers cut at the 2048-token limit: **0 of 40** (longest: 1060 tokens); every other answer ended on its own.

## dual card(s)

### Fixed length: 512 tokens each, 2 sessions (same work for both)

| Model | Requests | Generation tok/s | Prompt tok/s | Seconds per answer |
|---|---|---|---|---|
| base | 20 | 50.17 | 269.8 | 10.40 |
| mosv | 20 | 47.32 | 242.3 | 11.11 |

MoSV generation speed vs base: **-5.7%**.

### Natural answers (each model stops where it wants)

| Model | Avg answer tokens | Generation tok/s | Avg seconds per answer |
|---|---|---|---|
| base | 763 | 46.87 | 16.44 |
| mosv | 557 | 44.51 | 12.71 |

### llama-bench (5 repeats)

| Model | Test | tok/s | ± sd |
|---|---|---|---|
| base | pp512 | 1974.06 | 20.03 |
| base | tg128 | 53.11 | 0.58 |
| mosv | pp512 | 1934.86 | 26.77 |
| mosv | tg128 | 53.41 | 1.01 |

### Per prompt, fixed length (gen tok/s, mean of 2 sessions)

| # | Kind | Base | MoSV | Diff |
|---|---|---|---|---|
| 0 | teaching | 48.29 | 46.42 | -3.9% |
| 1 | teaching | 52.08 | 46.95 | -9.8% |
| 2 | debugging | 51.36 | 49.77 | -3.1% |
| 3 | debugging | 50.55 | 50.65 | +0.2% |
| 4 | story | 48.46 | 49.63 | +2.4% |
| 5 | story | 49.58 | 45.88 | -7.5% |
| 6 | everyday | 50.29 | 47.41 | -5.7% |
| 7 | explanation | 50.55 | 45.61 | -9.8% |
| 8 | engineering | 50.25 | 45.67 | -9.1% |
| 9 | email | 50.28 | 45.19 | -10.1% |

### Per prompt, natural answers

| # | Kind | Base tokens | MoSV tokens | Base s | MoSV s |
|---|---|---|---|---|---|
| 0 | teaching | 940 | 268 | 20.0 | 6.4 |
| 1 | teaching | 925 | 502 | 18.9 | 11.5 |
| 2 | debugging | 996 | 783 | 20.5 | 17.8 |
| 3 | debugging | 873 | 903 | 18.0 | 20.6 |
| 4 | story | 740 | 468 | 15.6 | 10.7 |
| 5 | story | 1000 | 623 | 20.8 | 13.8 |
| 6 | everyday | 670 | 619 | 15.0 | 14.0 |
| 7 | explanation | 448 | 584 | 9.9 | 13.1 |
| 8 | engineering | 699 | 705 | 17.6 | 16.1 |
| 9 | email | 343 | 118 | 8.0 | 3.0 |

## single card(s)

### Fixed length: 512 tokens each, 2 sessions (same work for both)

| Model | Requests | Generation tok/s | Prompt tok/s | Seconds per answer |
|---|---|---|---|---|
| base | 20 | 55.37 | 282.0 | 9.47 |
| mosv | 20 | 54.83 | 277.6 | 9.59 |

MoSV generation speed vs base: **-1.0%**.

### Natural answers (each model stops where it wants)

| Model | Avg answer tokens | Generation tok/s | Avg seconds per answer |
|---|---|---|---|
| base | 783 | 57.33 | 13.85 |
| mosv | 539 | 57.59 | 9.60 |

### llama-bench (5 repeats)

| Model | Test | tok/s | ± sd |
|---|---|---|---|
| base | pp512 | 1973.26 | 13.31 |
| base | tg128 | 58.12 | 1.22 |
| mosv | pp512 | 1975.46 | 10.96 |
| mosv | tg128 | 60.00 | 2.45 |

### Per prompt, fixed length (gen tok/s, mean of 2 sessions)

| # | Kind | Base | MoSV | Diff |
|---|---|---|---|---|
| 0 | teaching | 53.03 | 58.64 | +10.6% |
| 1 | teaching | 55.53 | 55.85 | +0.6% |
| 2 | debugging | 55.94 | 56.71 | +1.4% |
| 3 | debugging | 56.14 | 56.52 | +0.7% |
| 4 | story | 55.07 | 56.16 | +2.0% |
| 5 | story | 56.24 | 49.52 | -11.9% |
| 6 | everyday | 55.72 | 52.99 | -4.9% |
| 7 | explanation | 54.85 | 53.14 | -3.1% |
| 8 | engineering | 55.60 | 52.84 | -5.0% |
| 9 | email | 55.62 | 55.96 | +0.6% |

### Per prompt, natural answers

| # | Kind | Base tokens | MoSV tokens | Base s | MoSV s |
|---|---|---|---|---|---|
| 0 | teaching | 1011 | 223 | 17.7 | 4.0 |
| 1 | teaching | 966 | 513 | 17.0 | 8.9 |
| 2 | debugging | 1060 | 806 | 18.7 | 14.7 |
| 3 | debugging | 852 | 707 | 15.1 | 12.4 |
| 4 | story | 819 | 490 | 14.6 | 8.7 |
| 5 | story | 963 | 663 | 17.0 | 11.8 |
| 6 | everyday | 708 | 735 | 12.4 | 13.4 |
| 7 | explanation | 469 | 521 | 8.5 | 9.1 |
| 8 | engineering | 602 | 609 | 10.8 | 10.7 |
| 9 | email | 380 | 122 | 6.7 | 2.3 |
