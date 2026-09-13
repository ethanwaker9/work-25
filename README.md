# Folded Gaussian Sampling over NTRU Lattices

This repository contains implementation of our research "_Falcon’s Efficient Cousin: Optimal Trapdoor Quality with Linear Signing Memory_" and experimental evaluation for the folded trapdoor sampler,
the annular trapdoor generation matched to it, and the resulting hash and sign
signature scheme over NTRU lattices. 

| sampler | reference | quality |
| --- | --- | --- |
| `klein` | Klein / GPV nearest plane over `Z^{2n}` | `alpha_GPV` |
| `peikert` | Peikert round-off over the ring | `s_1(B)/sqrt(q)` |
| `folded` with depth `0` | Ducas--Prest hybrid sampler (Mitaka, Antrag) | `alpha_0` |
| `folded` with depth `l` | this work | `alpha_l` |
| `ffo` | fast Fourier orthogonalisation (Falcon) | `alpha_GPV` |

## Files and contents

```
saker/ring.py       power of two cyclotomic arithmetic, split/merge FFT, NTT
saker/ntru.py       NTRU equation solver (tower recursion with Babai reduction)
saker/keygen.py     discrete Gaussian, annular and folded trapdoor generation
saker/quality.py    folding hierarchy, quality functions, Gram-Schmidt profiles
saker/gauss.py      integer and continuous Gaussian samplers, operation counters
saker/samplers.py   the five samplers above, eager and lazy tree variants
saker/scheme.py     hash and sign signing and verification
saker/encoding.py   Golomb-Rice encoder and entropy optimal arithmetic coder
saker/security.py   CoreSVP estimates: forgery, key recovery, subfield attack
saker/params.py     parameter sets
demo.py             end to end key generation, signing and verification
experiments/        scripts
tests/              unit tests
```

## Running

```
python -m pytest tests -q
python demo.py 512          # key generation, signing, verification, encoding
cd experiments
python exp_folding.py        # folding identity and quality hierarchy: Tables 6 and 7
python exp_correctness.py    # output distribution of every sampler: Table 9
python exp_bench.py          # time, memory and operation counts: Tables 3 and 10
python exp_params.py         # parameter sets and security estimates: Table 1
python exp_keygen.py         # trapdoor generation cost: Table 8
python exp_sizes.py          # encoded signature sizes: Table 5
python make_figures.py       # Figures 1 and 3 to 7, as EPS and PDF
```
The scripts write a JSON file into `results/` and prints the corresponding
table to standard output; `make_figures.py` reads those files and writes into
`figures/`. Running the whole suite takes about twenty minutes on a laptop; the
dominant cost is the NTRU equation solver in dimension 1024. `python run_all.py` runs them in order. The values of randomness goes through `numpy.random.default_rng` with the seeds fixed in
the scripts, so repeated runs give identical numbers. Wall clock timings depend
on the machine; the operation counts (sequential sampling stages, continuous and
integer Gaussian samples, butterfly operations, stored words) do not.
