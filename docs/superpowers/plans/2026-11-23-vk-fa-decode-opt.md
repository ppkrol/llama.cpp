# Vulkan FA decode (q8_0 K / q4_0 V) - plan optymalizacji

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Przyspieszyc generowanie Inkling 2 Mini przy dlugim kontekscie przez lepsze wykorzystanie przepustowosci VRAM w kernelu flash attention dla 8 warstw z pelna uwaga (cel: ~458 -> ~650+ GB/s, czyli FA @131k z ~489 us do <= 340 us).

**Architecture:** Zmiany tylko w skalarnym shaderze FA (`flash_attn.comp`, sciezka MMQ dla K q8_0, dequant V q4_0) i w doborze parametrow (`get_fa_tuning_params_scalar`, split_k w `ggml_vk_flash_attn`). Kazdy krok mierzymy osobno mikrobenchmarkiem (test-backend-ops perf) i end-to-end (llama-bench tg @ d131072 / d262144). Zmiana zostaje tylko gdy daje zysk i przechodzi testy poprawnosci; inaczej revert.

**Tech Stack:** C++ (ggml-vulkan), GLSL compute + glslc, Vulkan (sterownik AMD Windows, RX 7900 XTX, RDNA3, wave32/64), test-backend-ops, llama-bench, PowerShell.

**Spec:** brak osobnego dokumentu; ustalenia z rozmowy (profil z `F:\llama-vulkan\diag\opt\tg_d128k_perf.err`) sa w sekcji "Stan wyjsciowy" ponizej.

## Stan wyjsciowy (zmierzony 2026-11-23, Q3_K_M, -ctk q8_0 -ctv q4_0, -ub 512)

- tg64 @ d131072 = 68.1 tok/s (~14.7 ms/token). Pelna uwaga: 8 x ~489 us = 3.9 ms/token.
- FA decode, ksztalt: `q(128,1,24,1) k(128,131328,8,1) v(128,131328,8,1) m(131328,1,24,1)` -> 489 us. Dane: K 142 MB + V 75 MB + maska 6 MB na warstwe = ~458 GB/s (karta ~960 GB/s).
- Warstwy SWA: `k(128,1536,8,1)` -> 40 x 28 us = 1.13 ms/token (~90 GB/s, ograniczone opoznieniem, nie przepustowoscia).
- INK_FA_SPLIT_MULT: 2 -> 68.1, 4 -> 65.7, 8 -> 63.4, 16 -> 62.5 tok/s (2 najlepsze).

Sciezka, ktora wykonuje sie przy decode (wyprowadzone z kodu, Task 1 to potwierdza logiem):
- `ggml_vk_flash_attn` (ggml-vulkan.cpp ~9326): GQA, `gqa_ratio = 3`, `N = 3`, maska z planem na kazda glowe (`mask_heads_gqa`) -> wymuszony `FA_SCALAR` (~9338).
- `get_fa_tuning_params_scalar` (~1209): subgroup 32, `row_split 1`, workgroup 128, `Br 4` (uzyte 3 wiersze), `Bc 32`, `D_split 8`, `shmem_staging 0` (AMD).
- `ggml_vk_fa_scalar_uses_mmq` (~1933) = true -> K przez `dotPacked4x8EXT` (`flash_attn_mmq_funcs.glsl`), V przez `dequantize4` q4_0 (`flash_attn_dequant.glsl`).
- split_k ~ shader_core_count * 2 / 8 grup -> 192 workgroupy = ~2 fale na SIMD.

Podejrzane przyczyny niskiej przepustowosci (do potwierdzenia pomiarami):
1. `Bc 32`: kazdy watek ma tylko 2 kolumny na iteracje -> malo odczytow w locie, duzo iteracji petli.
2. Maska per iteracja: ladowanie do `masksh` + 3 `barrier()` + `subgroupAll` + petla po `tmpsh` dla kazdych 32 kolumn.
3. Odczyty 16-bit: bloki q8_0 (34 B) i q4_0 (18 B) czytane przez `*_packed16` po `uint16`; kazdy watek osobno czyta skale bloku.
4. Niskie oblozenie (2 fale/SIMD) przy kernelu ograniczonym pamiecia.

## Global Constraints

- Branch: `vk-fa-decode-opt` (od `inkling2-vulkan` @ b37257220). Nie ruszac `inkling2-vulkan`.
- Osobny katalog builda: `F:\llama-inkling-fast-rebar\build-opt`. `build\` zostaje nietkniety, bo uzywaja go pliki `*-rebar*.bat` uzytkownika.
- Zmiany tylko dla sciezki: skalarny FA, `gqa_ratio > 1`, `N <= 8`, AMD (nie GCN). Inne sciezki i vendory bez zmian zachowania.
- Styl wg AGENTS.md: tylko ASCII w kodzie i komentarzach, komentarze 1-2 linie, prefiks `// local:` jak w reszcie forka.
- Kazdy commit dopiero po zgodzie uzytkownika; trailer `Assisted-by: Claude Opus 5.5` (nie `Co-authored-by`). Bez push bez osobnej zgody.
- Pomiary: tylko gdy nic innego nie uzywa GPU (SwarmUI zamkniete), te same flagi co `F:\llama-vulkan\diag\opt\opt1.ps1` (`-ngl 99 -fa 1 -ctk q8_0 -ctv q4_0 -t 5`, `GGML_VK_SUBALLOCATION_BLOCK_SIZE=1073741824`, `GGML_VK_DEVLOCAL_FIRST_MIB` = wagi + KV).
- Kryterium zatrzymania zmiany: FA @131k szybciej o >= 3% w perf i brak regresji tg @ d131072, eval FLASH_ATTN_EXT 100% OK. Inaczej revert tej zmiany.

## Review Focus

- KV nie bedacy wielokrotnoscia nowego `Bc` (np. 1536 + 17 tokenow w SWA) -> sciezka unaligned musi dawac wynik zgodny z CPU. Test: eval z `kv = 1553` (Task 2).
- Maska z `-inf` w calym bloku (okno SWA, `mask_heads = -24`) -> blok musi byc pominiety, wynik bez NaN. Test: eval `mask_heads = -24` w kazdym tasku.
- split_k > 1 z innym `Bc` -> `split_kv` dalej wielokrotnoscia `alignment`; wynik po `split_k_reduce` zgodny z CPU. Test: eval przy kv 8192 i 32768 (Task 2).
- Sinki (`sinks = true`) i `nb = 3` (spekulacja / kilka tokenow) dalej poprawne. Test: istniejace przypadki eval z linii ~11390 test-backend-ops.cpp.
- Prefill (N duze) nie moze sie zmienic: zmiany gated na `N <= 8`. Test: pp512 @ d65536 w llama-bench bez regresji (Task 5).

---

### Task 1: Build testowy, przypadki perf, pomiar bazowy

**Files:**
- Modify: `tests/test-backend-ops.cpp` (w `make_test_cases_perf`, obok przypadkow FA ~13560-13700)
- Create: `F:\llama-vulkan\diag\faopt\bench.ps1` (poza repo)

**Interfaces:**
- Produces: `bench.ps1 -Id <nazwa> [-Env @{...}] [-Tg]` -> dopisuje linie do `F:\llama-vulkan\diag\faopt\results.csv`: `id, case, us_per_run, GBps` dla perf FA oraz `id, tg64@d131072, tg64@d262144` gdy `-Tg`.
- Produces: przypadki perf o nazwie zawierajacej `mask_heads=24` i `type_K=q8_0,type_V=q4_0`.

- [ ] **Step 1:** Skonfigurowac build: `cmake -S . -B build-opt -G "Visual Studio 18 2026" -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON` (pozostale opcje skopiowac z `build\CMakeCache.txt`, poza katalogiem). Zbudowac `test-backend-ops` i `llama-bench` w Release.
- [ ] **Step 2:** Dodac w `make_test_cases_perf` petle `for kv in {1536, 32768, 131072, 262144}`: `test_flash_attn_ext(128, 128, 8, {3, 1}, kv, 1, true, true, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, {0, 1, 2, 3}, true, false, 0, 0, 24)` z komentarzem `// local: Inkling 2 decode, 24 Q heads over 8 K/V heads, one mask plane per head`.
- [ ] **Step 3:** Uruchomic `build-opt\bin\Release\test-backend-ops.exe -o FLASH_ATTN_EXT -b Vulkan0 -m perf -p mask_heads=24`. Oczekiwane: 4 linie; przy kv 131072 ~480-500 us (zgodne z profilem 489 us). Jesli roznica > 15%, zapisac i ustalic czemu (np. inny split_k) zanim pojdziemy dalej.
- [ ] **Step 4:** Uruchomic eval: `test-backend-ops.exe -o FLASH_ATTN_EXT -b Vulkan0`. Oczekiwane: wszystkie przypadki OK (to baza do porownan).
- [ ] **Step 5:** Napisac `bench.ps1` (wzor: `F:\llama-vulkan\diag\opt\opt1.ps1`): perf FA z kroku 3 -> parsowanie `us/run` i liczenie GB/s z rozmiaru K+V+maski; z `-Tg` dodatkowo `llama-bench -p 0 -n 64 -d 131072` i `-d 262144` na modelu Q3 (`Abiray-...Q3_K_M-fork.gguf`).
- [ ] **Step 6:** `bench.ps1 -Id base -Tg`. Zapisac wynik bazowy w tym pliku (sekcja Wyniki na dole).
- [ ] **Step 7:** Commit (po zgodzie): `tests : Inkling 2 decode FA perf cases`.

### Task 2: Strojenie parametrow decode (Bc, workgroup, split_k)

**Files:**
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`get_fa_tuning_params_scalar` ~1209-1292, split_k ~9545-9578)
- Test: `tests/test-backend-ops.cpp` (eval, sekcja ~11390)

**Interfaces:**
- Consumes: `bench.ps1` z Task 1.
- Produces: tymczasowe zmienne `INK_FA_DEC_BC`, `INK_FA_DEC_WG` (tylko na czas strojenia, usuniete przed commitem) oraz stale wartosci wpisane na sztywno dla AMD non-GCN, `n_rows <= 8`, `gqa`.

- [ ] **Step 1:** Dodac w eval (sekcja ~11390) przypadki `kv in {1553, 32768}` dla `nb = 1`, `mask_heads = 24` i `-24`, q8_0/q4_0. Uruchomic eval: OK na obecnym kodzie.
- [ ] **Step 2:** W `get_fa_tuning_params_scalar` dla AMD non-GCN i `n_rows <= 4` czytac `INK_FA_DEC_BC` (32/64/128) do `result.block_cols` oraz `INK_FA_DEC_WG` (64/128/256) do `result.workgroup_size`. `cols_per_thread = Bc / (WG / D_split)` musi byc >= 1.
- [ ] **Step 3:** Siatka: Bc {32, 64, 128} x WG {64, 128, 256} x `INK_FA_SPLIT_MULT` {1, 2, 4}: `bench.ps1 -Id bc<Bc>_wg<WG>_s<S> -Env @{...}` (bez `-Tg`). Wynik: tabela us dla kv 1536 / 32768 / 131072 / 262144.
- [ ] **Step 4:** Dla 2 najlepszych kombinacji: eval FLASH_ATTN_EXT (OK) i `bench.ps1 -Tg`. Zostaje kombinacja z najlepszym tg @ d131072 bez straty przy d262144.
- [ ] **Step 5:** Wpisac zwyciezce na sztywno (jesli inny niz obecny), usunac `INK_FA_DEC_*`, komentarz `// local:` z wynikiem pomiaru jak przy `INK_FA_SPLIT_MULT`. Jesli zadna kombinacja nie bije bazy o >= 3%: revert, zapisac wynik i przejsc dalej.
- [ ] **Step 6:** Commit (po zgodzie): `vulkan : tune scalar FA decode block size on AMD`.

### Task 3: Maska bez pamieci wspoldzielonej i barier w petli decode

**Files:**
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp:194-245, 457-465`

**Interfaces:**
- Consumes: parametry z Task 2 (Bc, WG).
- Produces: nic nowego dla innych taskow; zachowanie bez zmian poza ta sciezka.

- [ ] **Step 1:** Warunek nowej sciezki (uniform, staly w dispatchu): `MASK_ENABLE && SHMEM_STAGING == 0 && !USE_SPARSE && !USE_MASK_OPT && p.gqa_ratio > 1`. Poza nim kod bez zmian.
- [ ] **Step 2:** W nowej sciezce kazdy watek czyta z `data_m` tylko wartosci dla swoich `(tile_row(r), c * cols_per_iter + col_tid)` do rejestrow `ACC_TYPE mreg[rows_per_thread][cols_per_thread]` (wiersze `>= p.gqa_ratio` -> 0, poza KV -> 0). Pominiecie bloku: `subgroupAll(max <= NEG_FLT_MAX_OVER_2)` per subgroup, bez `barrier()` i bez `tmpsh`. W petli nie ma innych barier przy `SHMEM_STAGING == 0`, wiec rozbieznosc miedzy subgroupami jest dozwolona; shuffle po `D_split` zostaje w obrebie subgroupa.
- [ ] **Step 3:** Dodawanie maski (linie ~457-465) bierze `mreg` zamiast `masksh` w nowej sciezce.
- [ ] **Step 4:** Eval FLASH_ATTN_EXT: wszystkie OK, w tym `mask_heads = -24` (bloki calkiem `-inf`) i `kv = 1553`.
- [ ] **Step 5:** `bench.ps1 -Id mask_direct -Tg`. Oczekiwane: kv 131072 szybciej o >= 3%. Jesli nie: revert.
- [ ] **Step 6:** Commit (po zgodzie): `vulkan : load the FA decode mask in registers for grouped heads`.

### Task 4: Szersze odczyty K q8_0 i V q4_0 w decode

**Files:**
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_mmq_funcs.glsl` (odczyt K q8_0 dla MMQ)
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl` (odczyt V q4_0)
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` (petle K ~359-437 i V ~518-547)
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` tylko jesli trzeba nowego define/wariantu

**Interfaces:**
- Consumes: Task 2 i 3.
- Produces: helpery `fa_load_k_row_q8_0(uint kcol, uint k_offset)` i `fa_load_v_row_q4_0(uint vcol, uint v_offset)` uzywane tylko przy `HSK == 128 && D_split == 8` (inaczej stara sciezka).

- [ ] **Step 1:** Fakty do kodu: wiersz K (128 elementow q8_0) = 4 bloki x 34 B = 136 B, wiersz V (q4_0) = 4 x 18 B = 72 B; oba wielokrotnosci 8 B, wiec przy wyrownanym `k_offset`/`v_offset` wiersz da sie czytac jako `uvec2` (17 i 9 odczytow). Warunek wyrownania sprawdzic na hoscie (`(k->data offset | nbk1 | nbk2) % 8 == 0`) i przekazac jako uniform (np. bit w istniejacym polu flag push constants, bez nowego pola jesli sie da).
- [ ] **Step 2:** Dodac widok `layout (binding = 1) readonly buffer K_U64 { uvec2 data[]; }` (i dla V binding 2). 8 lane'ow jednej kolumny (`d_tid` 0..7) czyta wspolnie wiersz `uvec2` (lane l czyta indeksy l i l+8, lane 0 dodatkowo 16 dla K), a potrzebne bajty (skala + qs swojego fragmentu) wyciaga przez `subgroupShuffle` od wlasciciela.
- [ ] **Step 3:** K: z wyciagnietych bajtow zbudowac te same `k_quants[]` i `k_dm` co obecny `get_k_qs`/`get_k_scale`, reszta petli MMQ bez zmian. V: z bajtow q4_0 te same 4 wartosci co `FA_DEQUANT4_Q4_0`.
- [ ] **Step 4:** Eval FLASH_ATTN_EXT: wszystkie OK (wynik musi byc bit-w-bit lub w tolerancji jak przed zmiana).
- [ ] **Step 5:** `bench.ps1 -Id wide_loads -Tg`. Oczekiwane: kv 131072 FA <= ~400 us. Jesli < 3% zysku: revert.
- [ ] **Step 6:** Commit (po zgodzie): `vulkan : 8-byte row loads for q8_0/q4_0 FA decode`.

### Task 5: Walidacja end-to-end i dokumentacja

**Files:**
- Modify: `INKLING.md` (sekcja o predkosciach, tylko jesli jest zysk)
- Modify: ten plan (sekcja Wyniki)

- [ ] **Step 1:** llama-bench Q3 i Abiray Q4: `tg64 @ d0`, `@ d131072`, `@ d262144` oraz `pp512 @ d65536`, baza (`build\`) vs `build-opt\`. Oczekiwane: tg przy d0 bez zmian (+-1%), pp512 bez regresji, tg przy d131072/d262144 wyzsze.
- [ ] **Step 2:** Jakosc: `llama-perplexity` na krotkim tekscie (np. `wikitext-2-raw` 20 chunkow, `-c 4096`) dla bazy i `build-opt`; roznica PPL < 0.1%. Dodatkowo `llama-server` z `--temp 0` i ten sam prompt ~100k tokenow: odpowiedz identyczna lub sensowna (zmiany kolejnosci sumowania moga dac drobne roznice).
- [ ] **Step 3:** Zapisac tabele wynikow w sekcji Wyniki i (jesli zysk) w `INKLING.md`.
- [ ] **Step 4:** Commit (po zgodzie): `docs : FA decode speed on long context`.
- [ ] **Step 5:** Uzytkownik decyduje: merge do `inkling2-vulkan` i przebudowa `build\` (wtedy jego `.bat` dostaja zmiane) albo zostawienie na branchu.

## Opcjonalnie (po Task 5, osobna decyzja)

- Warstwy SWA (kv ~1536, 40 x 28 us/token = 1.13 ms): tu ogranicza opoznienie, nie przepustowosc. Sprawdzic `split_k = 1` i mniejszy workgroup dla `KV <= 4096` (siatka jak w Task 2 dla przypadku kv 1536). Potencjal do ~0.5 ms/token niezaleznie od dlugosci kontekstu.

## Wyniki

| id | FA us @1536 | @32768 | @131072 | @262144 | tg @d131072 | tg @d262144 |
| -- | --: | --: | --: | --: | --: | --: |
| base | 15.15 | 84.93 | 359.88 | 713.65 | 68.11 | 54.73 |

Base: test-backend-ops perf cases use the KV-cache layout (permute 0,2,1,3) and a dense mask (mask_dense=1, finite bias in every cell; the earlier lower numbers 307 us at 131072 came from masked-out blocks being skipped), mask_heads=24, q8_0/q4_0, nb=1: 624 GB/s at 131072 (K+V+mask bytes / time). In-model profile was 489 us at kv 131328, so the standalone case is still ~26% faster; judge changes by tg, not only by perf us.

### Task 2: grid Bc x WG x split_mult (decode, gqa 3, kv-cache layout, dense mask)

FA us per run (test-backend-ops perf), base = bc32_wg128_s2. tg re-measured the same session, base moved to 64.56 / 53.59 tok/s (machine state ~5% slower than the 68.11 / 54.73 of Task 1).

| Bc | WG | split | @1536 | @32768 | @131072 | @262144 |
| -- | -- | -- | --: | --: | --: | --: |
| 32 | 64 | 1 / 2 / 4 | 15.59 / 13.66 / 16.90 | 149.08 / 98.33 / 81.52 | 714.89 / 478.96 / 375.26 | 1407.86 / 949.34 / 740.93 |
| 32 | 128 | 1 / 2 / 4 | 12.59 / 13.35 / 18.51 | 111.12 / 85.43 / 80.47 | 485.75 / 359.24 / 389.30 | 958.22 / 711.54 / 788.28 |
| 32 | 256 | 1 / 2 / 4 | 17.09 / 25.79 / 43.27 | 112.88 / 122.16 / 140.75 | 480.09 / 503.23 / 528.85 | 938.99 / 977.82 / 1056.32 |
| 64 | 64 | 1 / 2 / 4 | 14.56 / 13.80 / 13.76 | 144.97 / 96.17 / 79.46 | 684.05 / 452.87 / 369.40 | 1353.25 / 892.94 / 732.40 |
| 64 | 128 | 1 / 2 / 4 | 12.16 / 13.34 / 13.33 | 97.86 / 79.63 / 74.89 | 460.02 / 353.23 / 363.82 | 909.92 / 697.40 / 741.82 |
| 64 | 256 | 1 / 2 / 4 | 15.94 / 24.92 / 24.85 | 87.10 / 98.10 / 116.11 | 418.91 / 451.95 / 476.76 | 825.69 / 877.77 / 1051.42 |
| 128 | 64 | 1 / 2 / 4 | 30.04 / 28.69 / 28.63 | 470.16 / 251.22 / 202.94 | 2049.49 / 1253.23 / 1028.45 | 3970.31 / 2494.45 / 1992.33 |
| 128 | 128 | 1 / 2 / 4 | 12.58 / 12.21 / 12.17 | 100.57 / 80.34 / 87.54 | 476.75 / 368.48 / 441.95 | 937.86 / 725.36 / 859.19 |
| 128 | 256 | 1 / 2 / 4 | 16.08 / 15.81 / 15.87 | 85.71 / 96.68 / 118.74 | 374.77 / 414.49 / 467.15 | 723.89 / 792.03 / 988.89 |

No combination was invalid (all 27 built and ran). Best perf vs base: bc64_wg128_s2 (-1.7% at 131072, -2.0% at 262144), bc64_wg128_s4 (-12% at 32768 only).

tg64 (llama-bench) for the two shortlisted combos and a base re-check:

| id | tg @d131072 | tg @d262144 |
| -- | --: | --: |
| bc32_wg128_s2 (base, re-check) | 64.56 | 53.59 |
| bc64_wg128_s2 | 64.42 | 50.88 |
| bc64_wg128_s4 | 64.23 | 52.51 |

Decision: no combination beats base by >= 3% (perf gains <= 2% at the long lengths, tg flat to worse). Code change reverted (INK_FA_DEC_* removed); only the new eval cases (kv 1553 / 32768, mask_heads +-24, both layouts) are kept.

### Task 3: mask in registers, no LDS and no barriers in the decode loop

Change: new path gated by a host flag (Flags bit 256, AMD non-GCN, FA_SCALAR) and `MASK_ENABLE && SHMEM_STAGING == 0 && !USE_SPARSE && !USE_MASK_OPT && p.gqa_ratio > 1`. Each lane read its own `rows_per_thread x cols_per_thread` mask values from `data_m` into `mreg` (rows >= gqa_ratio and cols past KV -> 0) and skipped a block per subgroup with `subgroupAll`; the `masksh` / `tmpsh` / 2 x `barrier()` code stayed for all other paths. Eval: 29/29 OK for `-p "mask_heads=-?24"` (kv 1553, mask_heads -24, both layouts, kv 32768 split_k).

Same session, base = HEAD rebuilt (t3_base), then the change (mask_direct):

| id | FA us @1536 | @32768 | @131072 | @262144 | tg @d131072 | tg @d262144 |
| -- | --: | --: | --: | --: | --: | --: |
| t3_base | 13.09 | 88.43 | 371.48 | 731.67 | 68.92 | 55.60 |
| mask_direct | 13.86 | 101.20 | 475.96 | 945.10 | 64.88 | 50.53 |

Decision: clear regression (FA +28% at 131072, tg -5.9% / -9.1%), so reverted. Likely cause (not profiled): each lane now issues 6 scattered 2-byte mask loads (the 8 D_split lanes of a column load the same values) instead of 1 coalesced load per thread, and the extra loads sit in front of the K reads every iteration; dropping the barriers did not make up for it. Patch kept outside the repo for reference.

### Task 4: 8-byte row loads for K q8_0 and V q4_0

Change: host flag bits 256 (K) / 512 (V) in Flags, set for AMD non-GCN, FA_SCALAR, gqa_ratio > 1, HSK = HSV = 128, D_split 8, no shmem staging, and 8-byte aligned rows (buffer offset, nb1, nb2, nb3). K q8_0 row (136 B = 17 uvec2): D_split lane l loads uvec2 2l and 2l+1 (lane 7 also 16), gets the next lane's first uvec2 with subgroupShuffle and the block scale from its even neighbor with subgroupShuffleXor (3 shuffles, 2-3 loads per lane instead of 9 u16 loads). V q4_0 row (72 B = 9 uvec2): lane l loads uvec2 l (lane 7 also 8), lanes are remapped to 16 consecutive elements (block l/2, low or high nibbles) with the output index changed to match, 2 xor shuffles + 2 shuffles per row (1-2 loads instead of 12 u16 loads). K and V rows of all columns are loaded at the start of the iteration. Eval: 29/29 OK for `-p "mask_heads=-?24"` (wide path active, flags 770), 35/35 OK for `-p "type_K=q8_0,type_V=q4_0"` (unaligned views fall back to the old path). Full `-o FLASH_ATTN_EXT` with the change: 5728/5728 OK.

Same session, base = HEAD binaries (t4_base), the change via a temporary env mask (K, V, both), base re-check at the end:

| id | FA us @1536 | @32768 | @131072 | @262144 | tg @d131072 | tg @d262144 |
| -- | --: | --: | --: | --: | --: | --: |
| t4_base | 13.96 | 83.78 | 358.02 | 706.20 | 68.37 | 55.45 |
| wide_loads (K+V) | 14.83 | 96.24 | 423.91 | 845.59 | 64.92 | 52.59 |
| wide_k | 13.25 | 84.57 | 383.87 | 758.57 | 68.25 | 55.38 |
| wide_v | 13.40 | 86.56 | 429.83 | 849.16 | 66.41 | 53.01 |
| t4_base2 | 13.09 | 84.26 | 371.45 | 736.67 | 69.11 | - |

Decision: no variant clears +3% tg (K only flat: -0.2% / -0.1%, V only -2.9% / -4.4%, both -5.0% / -5.2%; FA us +3% to +18% at 131072, the K-only figure depends on which base run it is compared to, the base drifted ~4%). Reverted. Scope of the result: this form (row loads hoisted to the top of the iteration plus shuffles to the consumer lanes) does not help; the hoist and the shuffles were not measured apart, so a just-in-time wide V load is not ruled out. Likely cause (not profiled): the shuffles (ds_bpermute) and selects add latency between the load and the dot, and the hoisted V rows hold more registers for the whole iteration. The patch gates on gqa_ratio > 1 but not N <= 8; tighten that if the idea is retried. Patch kept outside the repo (wide_loads.patch).

### Task 5: summary

Tasks 2-4 were all reverted, so the runtime code on this branch is the same as `inkling2-vulkan` (`git diff b37257220 HEAD -- ggml/ src/` is empty). The A/B in llama-bench and the perplexity check from Task 5 Steps 1-2 are therefore not needed, and INKLING.md is not changed. The branch only adds test cases (Inkling decode FA perf in the KV-cache layout with a dense mask, eval cases for kv 1553 / 32768 and mask_heads +-24) and this record.

What was learned for the decode FA kernel (scalar, gqa 3, q8_0 / q4_0, RDNA3, AMD Windows driver):
- In the model the kernel takes ~489 us at kv 131328 (~458 GB/s); the standalone perf case is ~360 us (~624 GB/s). The gap was not explained.
- Block size, workgroup size and split_k are already at the best point (Task 2).
- The loop is sensitive to load latency: anything that adds instructions or shuffles between the loads and the math made it slower (Tasks 3 and 4). Not tested: a just-in-time wide V load without the hoist, and software prefetch of the next block.
- tg drifts ~5% between sessions; only same-session comparisons are valid.
