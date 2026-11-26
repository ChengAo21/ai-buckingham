# Case-study inputs

`datasets/` contains the ten case-study inputs, totaling 4,926 observations. File names use the case IDs. `manifest.json` records the file names, columns and SHA-256 checksums.

| Case ID | Rows | Source and preparation |
|---|---:|---|
| `mhd` | 483 | Glaws et al.; Yuan & Lozano-Durán; numerical duct-flow benchmark |
| `pressure_gradient` | 144 | Dixit et al.; Yuan & Lozano-Durán; wall friction under pressure gradients |
| `rayleigh_benard` | 182 | Xie et al.; Rayleigh-Bénard convection measurements |
| `laser_keyhole` | 90 | Xie et al.; laser-induced keyhole depth measurements |
| `blasius` | 200 | Yuan & Lozano-Durán; numerical integration of the Blasius equation |
| `sandgrain` | 104 | Cheng et al.; compiled experimental and numerical roughness data |
| `velocity_scaling` | 1,109 | Trettel & Larsson; Modesti & Pirozzoli; Yuan & Lozano-Durán; four channel and two pipe DNS profiles |
| `goldenfeld` | 414 | Nikuradse; Goldenfeld; Sanfins et al.; dimensionless observations converted to dimensional realizations |
| `cess_channel` | 1,000 | Zhang et al.; numerical evaluation of the Cess eddy-viscosity relation |
| `rough_pipe` | 1,200 | Zhang et al.; piecewise laminar/Colebrook relation |

Case prompts, numerical settings and reference dimensions are in `configs/cases/`. Ancillary columns identify data sources or series. `MKE` is the wall-normal integral of squared mean streamwise velocity. For sand-grain roughness parameterization, `ks/Ra` is the response and `Ra, kt, k_rms, Sk, Ku, ES, Po` are the seven predictors. The `Sk + 2` transformation is applied during the analysis.

Logarithmic regime analysis has 196 eligible Blasius and 1,103 eligible velocity observations after omitting zero responses. Cess and rough-pipe surrogate settings use 800 and 1,100 training observations respectively, separate from final collapse-validation splits.

Knowledge bases reside in `knowledge/` with entry-level citations. [Generation scripts](generation/README.md) reconstruct four numerical inputs. Use archived CSVs for comparisons with published numbers: solver and floating-point differences can affect regenerated final digits. Data retain the rights of their cited sources.
