# Dataset sources and provenance

The following records reproduce the provenance stated in **SI Table 7**. They document the supplied copies; no claim is made that the data were newly downloaded, regenerated from the cited work, or relicensed. Exact column schemas, numerical ranges and dimensional assignments appear in SI Tables 8–24 and in the app.

| Supplied file | N | Stated source | Preparation described in the SI |
|---|---:|---|---|
| dataset_mhd.csv | 483 | Glaws et al.; Yuan & Lozano-Durán | Published numerical duct-flow benchmark with dimensionless mean velocity |
| dataset_presGrad.csv | 144 | Dixit et al.; Yuan & Lozano-Durán | Compiled experimental/numerical observations with normalized variables |
| dataset_rayben.csv | 182 | Xie et al. | Published convection measurements compiled for dimensional learning |
| dataset_keyhole.csv | 90 | Xie et al. | Published high-speed X-ray observations of laser-induced keyhole depth |
| dataset_blasius.csv | 200 | Yuan & Lozano-Durán | Numerical integration of the Blasius equation |
| dataset_veloTrans.csv | 1,109 | Trettel & Larsson; Modesti & Pirozzoli; Yuan & Lozano-Durán | Mean profiles from four channel and two pipe DNS cases |
| dataset_goldenfeld.csv | 414 | Nikuradse; Goldenfeld; Sanfins et al. | Supplied dimensionless rough-pipe observations converted to randomized dimensional realizations |
| dataset_cess_channel.csv | 1,000 | Zhang et al. | Cess relation evaluated with stratified wall-normal sampling; no added noise |
| dataset_pipe.csv | 1,200 | Zhang et al. | Piecewise laminar/Colebrook relation, rejection sampling, no added noise |

The auxiliary `dataset_sand_grain.csv` has 118 rows. The source script associates it with Zhou et al., JCP 2024, but the nine-case paper does not give a complete data license/provenance entry for it. It is preserved as an auxiliary input and is not promoted as a tenth validated manuscript case. The script's laminar mean-velocity example references `dataset_laminar_mvp.csv`, which was not supplied; no substitute dataset is fabricated.

Useful primary references identified in the supplied SI include:

- Xie et al., *Data-driven discovery of dimensionless numbers and governing laws from scarce measurements*, Nature Communications 13, 7562 (2022), [doi:10.1038/s41467-022-35084-w](https://doi.org/10.1038/s41467-022-35084-w).
- Yuan & Lozano-Durán, *Dimensionless learning based on information*, Nature Communications 16, 9171 (2025), [doi:10.1038/s41467-025-64425-8](https://doi.org/10.1038/s41467-025-64425-8).
- Zhang et al., *Clustering dimensionless learning for multiple-physical-regime systems*, CMAME 420, 116728 (2024), [doi:10.1016/j.cma.2023.116728](https://doi.org/10.1016/j.cma.2023.116728).
- Goldenfeld, *Roughness-induced critical phenomena in a turbulent flow*, Physical Review Letters 96, 044503 (2006), [doi:10.1103/PhysRevLett.96.044503](https://doi.org/10.1103/PhysRevLett.96.044503).
- Glaws et al., *Dimension reduction in MHD power generation models: dimensional analysis and active subspaces* (2016), [arXiv:1609.01255](https://arxiv.org/abs/1609.01255).

The two supplied literature-grounded Markdown knowledge bases contain 247 dimensional physical quantities and 251 dimensionless relations. They retain entry-level source information and ambiguity notes. Their records are unmodified copies; lexical/semantic retrieval metadata and cached embeddings are separate runtime artifacts.

`data/manifest.json` records SHA-256 hashes of every original CSV and knowledge-base file. `docs/assets/provenance.json` identifies the selected original figure images and Streamlit preview, with their checksums. The public release distributes those images, not the full manuscript or supplementary document. Original artwork crops used by the workflow retain their source record in `docs/assets/workflow/original-artwork-provenance.json`. Author and third-party rights remain applicable.
