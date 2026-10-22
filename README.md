# Code for "Sources of Arctic rapid-refreezing predictability: near-surface temperature, warm advection, and the onset forecast limit"

Code accompanying the manuscript under review at *The Cryosphere* (preprint egusphere-2026-6516).

The repository contains the complete pipeline used in the study: data preparation (ERA5 + NSIDC-0051),  
ConvLSTM training, the four verification levels (field scores, warning-level decomposition, Integrated  
Gradients attribution, and the regime diagnosis), and the scripts that generate every figure and table  
of the paper.

## Contents and correspondence to the manuscript

| Script                             | Role                                                                                                                     | Paper                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `step2_prepare_data_new.py`        | Sample construction: 15-channel tensors (SIC, 5 meteorological fields, 7 freezing-degree-day windows, land mask)         | Sect. 2, 3.1-3.4           |
| `convlstm.py`, `dataset.py`        | ConvLSTM encoder-decoder and dataset definitions                                                                         | Sect. 3.5                  |
| `step3_train.py`                   | Training of the full model and ablated configurations (single run each, per the single-run protocol stated in the paper) | Sect. 3.5                  |
| `step7_fdd_edge_analysis.py`       | Evaluation export: test predictions, transition pixels, regional edge statistics                                         | Sect. 3.6                  |
| `step4_eval_paired.py`             | Paired field-score evaluation (IIEE)                                                                                     | Sect. 4.1                  |
| `step8_freeze_onset_warning.py`    | Warning-level verification (missed freeze / false warning decomposition, Table 1)                                        | Sect. 4.2, Table 1         |
| `step9_miss_diagnosis.py`          | Temperature-anomaly regime diagnosis of missed versus warned onsets (Table 3)                                            | Sect. 4.4, Table 3         |
| `step9b_advection.py`              | Temperature-advection (-V.gradT) diagnostic (Sect. 3.6, 4.4)                                                             | Sect. 4.4                  |
| `step13_ig_maps.py`                | Integrated Gradients attribution: band means (Table 2) and attribution maps                                              | Sect. 4.3, Table 2, Fig. 7 |
| `step10_nep_shipping_synthesis.py` | Northeast Passage route-closure timeline (Fig. 9)                                                                        | Sect. 4.5, Fig. 9          |
| `region_defs.py`                   | Region masks (IHO S-23 sectors, Central Arctic, Remaining)                                                               | Sect. 2.3, Fig. 1          |
| `step14_domain_map.py`             | Figure 1 (study domain)                                                                                                  | Fig. 1                     |
| `step15_rsgp_definition_figure.py` | Figure 2 (objective RSGP definition)                                                                                     | Sect. 3.4, Fig. 2          |
| `step16_architecture_schematic.py` | Figure 3 (architecture schematic)                                                                                        | Sect. 3.5, Fig. 3          |
| `make_fig4_leadtime.py`            | Figure 4 (lead-time paired IIEE difference)                                                                              | Sect. 4.1, Fig. 4          |
| `make_fig6_missed_scatter.py`      | Figure 6 (excess missed area by start date)                                                                              | Sect. 4.2, Fig. 6          |
| `make_fig8_cases.py`               | Figure 8 (case-study warning maps)                                                                                       | Sect. 4.5, Fig. 8          |
| `step18_build_tables.py`           | Table assembly from the statistics JSONs                                                                                 | Tables 1-3                 |
| `config.yaml`                      | All data paths, channel definitions, and training hyperparameters                                                        | Sect. 3.5                  |

## Data sources

- Sea-ice concentration: NSIDC-0051, version 2 (NASA Team, 25 km Northern Hemisphere polar stereographic), <https://nsidc.org/data/NSIDC-0051>
- Atmospheric forcing: ERA5 (hourly, 0.25 deg), Copernicus Climate Data Store, <https://cds.climate.copernicus.eu>
- Pan-Arctic sea-ice extent for the RSGP definition: NSIDC Sea Ice Index, version 4, <https://nsidc.org/data/seaice_index/>

Raw data are publicly available from the sources above and are not redistributed here. Preprocessed  
samples (`npy/`, about 36 GB) and the trained-model test predictions are provided in the companion  
archive deposited with this code.

## Configuration

All paths, the 15-channel definition, and the training hyperparameters live in `config.yaml`.  
The NSIDC credential fields are left empty; fill them in only if you need to re-download raw NSIDC  
data (a local `credentials.yaml` or the environment variables `NSIDC_USERNAME` / `NSIDC_PASSWORD`  
are also supported).

## Environment

```bash
pip install -r requirements.txt
# training additionally requires a CUDA GPU; the preprocessed sample archive is about 36 GB
python step3_train.py            # train the full model
python step7_fdd_edge_analysis.py  # evaluation export
python step9_miss_diagnosis.py   # regime diagnosis (Table 3)
python step13_ig_maps.py         # Integrated Gradients (Table 2, Fig. 7)
```

## Notes

- Each configuration is trained once, following the single-run protocol stated in Sect. 3.5 of the    
  paper; small paired differences in field accuracy are therefore read as ties.
- The channel order is fixed by `step2_prepare_data_new.py` (`VAR_ORDER`): SIC (0), t2m/d2m/u10/v10/    
  sp_grad_x/sp_grad_y (1-6), fdd2-fdd20 (7-13), land_mask (14).
- Figures 2 and 3 are schematic/definition graphics produced by their respective scripts; Figure 5    
  (configuration ladder) is generated by `step7_fdd_edge_analysis.py`.

## License

MIT License (see `LICENSE`).
