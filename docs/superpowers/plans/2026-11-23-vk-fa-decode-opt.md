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
| base | 15.41 | 87.74 | 370.74 | 742.52 | 68.11 | 54.73 |

Base: test-backend-ops perf (mask_heads=24, q8_0/q4_0, nb=1) gives 605 GB/s at 131072 (K+V+mask bytes / time). In-model profile was 489 us at kv 131328, i.e. the standalone case is ~24% faster (see task-1 report); judge changes by tg, not only by perf us.
