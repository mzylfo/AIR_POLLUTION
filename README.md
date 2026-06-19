# Generation of Air-Pollution Distribution Scenarios for the City of Seoul

**Bachelor's thesis — Matteo Zylfo, Università degli Studi di Milano-Bicocca**

This folder is the entry point for the thesis work: generating **synthetic scenarios of
air-pollutant distribution** over the city of **Seoul**, using a **β-Variational Autoencoder
(β-VAE)** that is trained to preserve the **spatial correlation** between monitoring zones.

> The heavy machinery (models, training, analysis) lives in the shared package
> [`../src/NeuroCorrelation/`](../src/NeuroCorrelation). This folder does **not** duplicate it:
> it provides the documentation, the dependency list, and a launcher
> ([`run_seoul.sh`](run_seoul.sh)) that drives the real pipeline with the correct
> Seoul configuration.

---

## 1. The problem

Seoul is modelled as a **graph of monitoring zones**:

- each **node** is a city zone with a pollutant time series (year **2019**);
- each **edge** encodes **spatial adjacency** between zones
  (`SEOUL_SLOT_GRAPH_n{16,25}__edges.csv`);
- two graph sizes are available: **16 nodes** and **25 nodes**.

The goal is **not forecasting**. The goal is **generative**: produce new, plausible spatial
snapshots ("scenarios") of how a pollutant is distributed across the city, such that the
synthetic data reproduces both the **marginal distribution** of each zone and the
**inter-zone correlation structure** of the real data.

Five pollutants are covered: **PM2.5, PM10, NO₂, O₃, SO₂**.

---

## 2. The model — β-VAE with a correlation-aware loss

The model is a **β-VAE** (experiment ids `bvae_0016_seoul`, `bvae_0025_seoul`):

| Setting | 16 nodes | 25 nodes |
|---------|----------|----------|
| Input dim (zones) | 16 | 25 |
| Latent dim | 6 | 8 |
| Epochs (VAE) | 100 | 100 |
| Batch size | 16 | 16 |
| Train split | 0.9 | 0.9 |
| Generated samples | 2375 | 2375 |

The core contribution is the **composite loss**: instead of letting correlation emerge on its
own, it is **injected as explicit loss terms**:

| Loss term | Weight | Purpose |
|-----------|--------|---------|
| MSE | 1.0 | reconstruction |
| KL divergence | 0.5 | latent regularization (the β term) |
| Variance | 0.5 | avoid variance collapse |
| Jensen–Shannon | 0.1 | distribution shape |
| Median (per batch) | 1.0 | centering |
| **Pearson correlation** | 0.01 | linear inter-zone correlation |
| **Spearman correlation** | 0.01 | rank inter-zone correlation |
| **Wasserstein** | 0.5 | distance between real / generated distributions |

Optionally, the weights can be tuned via **Bayesian optimization** (Gaussian Process,
scikit-optimize) by passing `--optimization yes`.

---

## 3. Validation

Each run produces, under the experiment folder:

- `correlation_comparison/` — real vs. generated correlation matrices;
- `datasplit/` — train/test samples and the copula-generated baseline;
- `maps_analysis_SEOUL/` + `scenarios_map/` — scenarios drawn on the **real Seoul map**
  ([`../src/NeuroCorrelation/Analysis/ScenariosMap.py`](../src/NeuroCorrelation/Analysis/ScenariosMap.py),
  via `folium`, `RdYlGn` colormap: green = clean, red = polluted);
- `summary/` — dataset/network settings and timing.

---

## 4. How to run

### Requirements

From the repository root, create an environment and install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r THESIS_SEOUL/requirements.txt
```

> Heavy dependencies: `torch`, `torch-geometric`, `folium`, `copulas`, `scikit-optimize`.
> A CUDA GPU is used automatically if available, otherwise it falls back to CPU.

### Launch a Seoul experiment

The launcher handles the one manual step of the original workflow — selecting the pollutant by
placing its CSV where the dataset definition expects it — and then calls the real pipeline:

```sh
# from anywhere; the script locates the repo root by itself
./THESIS_SEOUL/run_seoul.sh PM25 16 0 10
#                            │    │  │ └─ last seed (exclusive)
#                            │    │  └─── first seed
#                            │    └────── nodes: 16 or 25
#                            └─────────── pollutant: PM25 | PM10 | NO2 | O3 | SO2
```

Pollutant availability:

| Pollutant | 16 nodes | 25 nodes |
|-----------|:--------:|:--------:|
| PM2.5 (`PM25`) | ✓ | ✓ |
| PM10 (`PM10`)  | ✓ | ✓ |
| NO₂ (`NO2`)    | — | ✓ (unstable Wasserstein — see thesis limitations) |
| O₃ (`O3`)      | — | ✓ |
| SO₂ (`SO2`)    | — | ✓ |

The equivalent raw command (what the script ultimately runs, from the repo root) is:

```sh
python test.py --exp neuroD \
  --num_case bvae_0016_seoul \
  --experiment_name_suffix SEOUL_16 \
  --main_folder my_seoul_run \
  --repeation_b 0 --repeation_e 10 \
  --optimization no --load_model no --train_models yes \
  --time_slot B
```

Results are written under `data/neuroCorrelation_experiments/<main_folder>/`.

---

## 5. Notes & limitations

- **Small data regime** (16–25 nodes, ~2375 samples): results are statistically delicate and
  prone to overfitting; the low correlation weights (0.01) and the β term mitigate this.
- **NO₂ @ 25 nodes** showed Wasserstein-loss instability (the source file is literally named
  `..._NO2_wasserstein_problem.csv`).
- The model operates on **vectors**; the graph edges are mainly used for adjacency/mapping.
  Moving to a **Graph-Neural-Network VAE** is the natural next step (the `..._graph` cases in
  the repo are the starting point).
