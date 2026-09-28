# Flux-Informed Autophagy Signature + ChEMBL Assay Integration
## Gene Selection
Flux-informative genes selected: 25
| Rank | Gene | Cell lines | Contrast score | Priority |
|------|------|------------|----------------|----------|
| 1 | DDIT4 | 13 | 3.737 | Yes |
| 2 | BNIP3 | 9 | 2.789 | Yes |
| 3 | BNIP3L | 9 | 2.581 | Yes |
| 4 | BECN1 | 15 | 1.756 | Yes |
| 5 | NPC1 | 13 | 1.531 | No |
| 6 | WIPI1 | 9 | 1.271 | Yes |
| 7 | CTSL | 10 | 1.074 | No |
| 8 | TSC2 | 10 | 1.027 | Yes |
| 9 | ULK1 | 9 | 1.005 | Yes |
| 10 | ATG13 | 11 | 0.971 | Yes |
| 11 | GABARAPL1 | 12 | 0.957 | Yes |
| 12 | WIPI2 | 13 | 0.871 | Yes |
| 13 | EIF4EBP1 | 9 | 0.855 | No |
| 14 | GABARAPL2 | 10 | 0.851 | Yes |
| 15 | MAP1LC3B | 8 | 0.841 | Yes |
| 16 | ATG101 | 9 | 0.832 | No |
| 17 | LAMP1 | 7 | 0.811 | Yes |
| 18 | CTSB | 11 | 0.804 | Yes |
| 19 | CTSD | 10 | 0.803 | Yes |
| 20 | TSC1 | 11 | 0.799 | Yes |
| 21 | SQSTM1 | 6 | 0.797 | Yes |
| 22 | LAMP2 | 12 | 0.794 | Yes |
| 23 | RB1CC1 | 9 | 0.780 | Yes |
| 24 | ATG12 | 10 | 0.773 | Yes |
| 25 | ATG3 | 11 | 0.762 | No |

## Per-cell-line Signature Quality
| Cell line | n_act | n_inh | Activator quality | Inhibitor quality |
|-----------|-------|-------|-------------------|-------------------|
| 2 | 314 | 58 | 0.133 | 0.287 |
| 4 | 434 | 86 | 0.133 | 0.153 |
| 5 | 75 | 18 | 0.177 | 0.307 |
| 7 | 232 | 44 | 0.110 | 0.251 |
| 8 | 97 | 19 | 0.207 | 0.177 |
| 9 | 120 | 22 | 0.162 | 0.146 |
| 10 | 95 | 21 | 0.122 | 0.230 |
| 13 | 47 | 15 | 0.197 | 0.278 |
| 15 | 72 | 14 | 0.225 | 0.242 |
| 20 | 147 | 23 | 0.082 | 0.221 |
| 25 | 261 | 31 | 0.079 | 0.125 |
| 26 | 65 | 20 | 0.227 | 0.132 |
| 27 | 47 | 4 | 0.055 | 0.282 |
| 28 | 65 | 20 | 0.158 | 0.215 |
| 29 | 65 | 20 | 0.166 | -0.042 |
| 31 | 64 | 20 | 0.148 | 0.062 |
| 33 | 126 | 17 | 0.090 | 0.174 |
| 74 | 76 | 17 | 0.086 | 0.048 |

## Cross-validation
| Cell line | n_train | n_test | median_test_corr | median_random_corr | delta |
|-----------|---------|--------|------------------|--------------------|-------|
| 2.0 | 208.0 | 106.0 | 0.097 | 0.001 | 0.096 |
| 4.0 | 309.0 | 125.0 | 0.049 | -0.028 | 0.077 |
| 5.0 | 49.0 | 26.0 | 0.051 | -0.022 | 0.072 |
| 7.0 | 150.0 | 82.0 | 0.107 | -0.002 | 0.110 |
| 8.0 | 67.0 | 30.0 | 0.116 | 0.005 | 0.110 |
| 9.0 | 87.0 | 33.0 | 0.131 | -0.010 | 0.141 |
| 10.0 | 70.0 | 25.0 | 0.107 | 0.040 | 0.067 |
| 13.0 | 28.0 | 19.0 | 0.011 | 0.035 | -0.024 |
| 15.0 | 46.0 | 26.0 | 0.205 | 0.007 | 0.198 |
| 20.0 | 99.0 | 48.0 | 0.177 | 0.006 | 0.171 |
| 25.0 | 157.0 | 104.0 | -0.012 | 0.032 | -0.044 |
| 26.0 | 45.0 | 20.0 | 0.232 | 0.058 | 0.173 |
| 27.0 | 34.0 | 13.0 | -0.148 | -0.025 | -0.123 |
| 28.0 | 45.0 | 20.0 | 0.308 | 0.004 | 0.304 |
| 29.0 | 45.0 | 20.0 | 0.334 | 0.118 | 0.216 |
| 31.0 | 44.0 | 20.0 | 0.158 | -0.015 | 0.173 |
| 33.0 | 93.0 | 33.0 | 0.038 | 0.010 | 0.028 |
| 74.0 | 53.0 | 23.0 | 0.038 | -0.011 | 0.049 |

## Inverse Search HAMDB Enrichment
- Top 50: 0.0% HAMDB activators
- Top 100: 1.0% HAMDB activators
- Top 200: 2.5% HAMDB activators

## Top 20 Hits
| rank | canonical_smiles | contrastive_score | activator_corr_mean | inhibitor_corr_mean | pvalue | is_hamdb_activator | is_hamdb_inhibitor |
|---|---|---|---|---|---|---|---|
| 1 | Cc1ccc(cc1)S(=O)(=O)N[C@H]1CC[C@@H](CC(=O)N2CCc3ccccc3C2)O[C@@H]1CO | 1.0946153846153845 | 0.67 | -0.4246153846153846 | 0.01 | False | False |
| 2 | COc1cccc(c1)S(=O)(=O)N(C)C[C@H]2Oc3cc(ccc3S(=O)(=O)N(C[C@@H]2C)[C@@H](C)CO)C4=CCCCC4 | 1.0669230769230769 | 0.5215384615384615 | -0.5453846153846154 | 0.010833333333333334 | False | False |
| 3 | C[C@H](CO)N1C[C@@H](C)[C@H](CN(C)C(=O)NC2CCCCC2)OCc3ccccc3-c4ccccc4C1=O | 0.9715384615384615 | 0.6338461538461537 | -0.3376923076923077 | 0.01611111111111111 | False | False |
| 4 | CC(C)NC[C@H]1[C@H]([C@@H](CO)N1C(=O)NC(C)C)c1ccc(cc1)C1=CCCCC1 | 0.9430769230769231 | 0.6323076923076923 | -0.31076923076923074 | 0.017222222222222222 | False | False |
| 5 | C[C@H](CO)N1C[C@@H](C)[C@H](CN(C)S(=O)(=O)c2ccccc2F)Oc3cc(ccc3S1(=O)=O)c4ccncc4 | 0.8846153846153846 | 0.3984615384615384 | -0.48615384615384616 | 0.023333333333333334 | False | False |
| 6 | C[C@H](CO)N1C[C@@H](C)[C@H](CN(C)S(=O)(=O)c2ccc(C)cc2)Oc3cc(C#Cc4ccncc4)ccc3S1(=O)=O | 0.8846153846153846 | 0.5330769230769231 | -0.3515384615384615 | 0.023333333333333334 | False | False |
| 7 | CO[C@H]1CN(C)C(=O)c2cc(NC(=O)C3CCOCC3)ccc2OC[C@@H](C)N(Cc2cccnc2)C[C@@H]1C | 0.8815384615384614 | 0.26153846153846155 | -0.6199999999999999 | 0.023333333333333334 | False | False |
| 8 | CN(C)C(=O)C[C@@H]1C[C@H]2[C@H](Oc3ccc(NC(=O)CC4CC4)cc23)[C@H](CO)O1 | 0.8607692307692308 | 0.7484615384615385 | -0.11230769230769232 | 0.025277777777777777 | False | False |
| 9 | C[C@@H](CO)N1C[C@@H](C)[C@@H](CN(C)Cc2ccc(Cl)c(Cl)c2)Oc3ccc(NC(=O)CCCCCC(=O)Nc4ccccc4N)cc3CC1=O | 0.8581065453339022 | 0.5461149817526898 | -0.3119915635812124 | 0.025555555555555557 | False | False |
| 10 | OC[C@@H]1O[C@H](CC(=O)NCC2CC2)C=C[C@H]1NC(=O)c1ccncc1 | 0.85 | 0.3653846153846153 | -0.48461538461538467 | 0.02638888888888889 | False | False |
| 11 | OC[C@H]1[C@H]([C@H](C#N)N1c1nc(cs1)-c1ccccc1)c1ccc(cc1)C1=CCCC1 | 0.8492307692307692 | 0.2792307692307692 | -0.57 | 0.02666666666666667 | False | False |
| 12 | OC[C@H]1O[C@@H](CC(=O)NCc2cccnc2)C[C@H]3[C@@H]1Oc4ccc(NC(=O)c5ccc6OCOc6c5)cc34 | 0.8476923076923077 | 0.33076923076923076 | -0.5169230769230769 | 0.02666666666666667 | False | False |
| 13 | C[C@H](CO)N1C[C@H](C)[C@@H](CN(C)Cc2cncnc2)Oc3cc(C#CC4CCCC4)ccc3S1(=O)=O | 0.8438461538461539 | 0.4792307692307692 | -0.3646153846153847 | 0.026944444444444444 | False | False |
| 14 | C[C@@H](O)C#Cc1cnc2O[C@H](CN(C)CC3CCOCC3)[C@H](C)CN([C@H](C)CO)C(=O)c2c1 | 0.8346153846153846 | 0.3923076923076923 | -0.44230769230769235 | 0.02861111111111111 | False | False |
| 15 | C[C@H](CO)N1C[C@H](C)[C@@H](CN(C)C(=O)c2cc(on2)c3ccccc3)OCc4cnnn4CCCC1=O | 0.8344192856769337 | 0.26404382157611256 | -0.5703754641008211 | 0.02861111111111111 | False | False |
| 16 | OC[C@@H]1O[C@H](CC(=O)NCCc2ccccc2)C[C@H]3[C@@H]1Oc4ccc(NC(=O)c5ccc6OCOc6c5)cc34 | 0.833076923076923 | 0.15538461538461537 | -0.6776923076923077 | 0.02861111111111111 | False | False |
| 17 | CCCNC(=O)N(C)C[C@@H]1Oc2cc(C#CC(C)(C)O)ccc2S(=O)(=O)N(C[C@H]1C)[C@H](C)CO | 0.8223076923076923 | 0.5315384615384615 | -0.2907692307692308 | 0.029722222222222223 | False | False |
| 18 | C[C@H](CO)N1C[C@@H](C)[C@@H](CN(C)C(=O)CCN(C)C)OCc2ccccc2-c3ccccc3C1=O | 0.8192307692307693 | 0.44538461538461543 | -0.3738461538461539 | 0.029722222222222223 | False | False |
| 19 | C[C@@H](CO)N1C[C@@H](C)[C@H](CN(C)S(=O)(=O)c2ccc(C)cc2)Oc3cc(C#Cc4ccncc4)ccc3S1(=O)=O | 0.8192307692307692 | 0.5323076923076923 | -0.2869230769230769 | 0.029722222222222223 | False | False |
| 20 | CCCNC(=O)C[C@@H]1C[C@@H]2[C@@H](Oc3ccc(NC(=O)COC)cc23)[C@H](CO)O1 | 0.8184615384615385 | 0.46384615384615385 | -0.3546153846153846 | 0.029722222222222223 | False | False |

## ChEMBL Assay Integration
- Autophagy-related ChEMBL assays queried: 3602
- Top-200 hits with ChEMBL autophagy activity data: 5

### Top ChEMBL-annotated hits
| rank | canonical_smiles | contrastive_score | n_autophagy_assays | best_pchembl | chembl_activity_summary |
|---|---|---|---|---|---|
| 97 | Nc1cc(c(cn1)-c1cc(nc(n1)N1CCOCC1)N1CCOCC1)C(F)(F)F | 0.6461390784982936 | 7 | 7.72 | CHEMBL4019296:Ratio IC50=13.0None; CHEMBL4019293:IC50=94.0nM; CHEMBL4019288:Ratio=5.4None |
| 124 | COc1ccc(cc1CO)-c1ccc2c(nc(nc2n1)N1CCOC[C@@H]1C)N1CCOC[C@@H]1C | 0.6247456243493631 | 12 | 9.89 | CHEMBL2340652:Inhibition=None%; CHEMBL2346088:Inhibition=None%; CHEMBL2338827:IC50=27.0nM |
| 152 | Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@@H]2O)c(=O)n1 | 0.6048089720322025 | 3 | 7.13 | CHEMBL1614293:Potency=926.8nM; CHEMBL1613805:Potency=1309.2nM; CHEMBL1614009:Potency=73.6nM |
| 162 | CS(=O)(=O)N1CCN(Cc2cc3nc(nc(N4CCOCC4)c3s2)-c2cccc3[nH]ncc23)CC1 | 0.5995579322329911 | 12 | 7.32 | CHEMBL1103322:IC50=310.0nM; CHEMBL1109707:Ki=570.0nM; CHEMBL1925468:Ki=580.0nM |
| 171 | Oc1cccc(c1)-c1nc(N2CCOCC2)c2oc3ncccc3c2n1 | 0.592811438092864 | 23 | 8.28 | CHEMBL2219345:Residual Activity=49.0%; CHEMBL2219346:Residual Activity=35.0%; CHEMBL2219347:Residual Activity=-12.0% |

## Interpretation
HAMDB activator enrichment remains low even with the flux-informed gene set. This suggests that the transcriptional signal distinguishing HAMDB-curated autophagy activators from inhibitors is weak in L1000, or that the conserved stress response dominates. The ChEMBL annotations should be used as the primary orthogonal filter for prioritizing hits.
