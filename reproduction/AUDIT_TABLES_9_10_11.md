# Revision102 표 9·10·11 원본 실행 기록 대조 (2026-09-16)

공통 설정 기준(현재 STFR 정의): CATEPP_SFD_FRESH_NORM=minmax, CATEPP_SFD_NO_STALE=1(B 슬롯 없음), 200 epoch 상한, patience 15, EVAL_EVERY=3, 시드 20/21/22, 검증 Recall@20 선택. 재학습 없음(로그 파싱만).


## STFR 챔피언 (f, α) 선택 근거 — 12점 스윕 (seed 20, 30ep; Douban은 full-val 재평가값으로 선택)

- Amazon-VG/MF: n_grid=12, 선택 f=0.7 α=0.75 (val R@20 0.13006); 최종 로그 dial과 일치 여부는 table9/rank per-seed 로그명 참조
- Amazon-VG/LightGCN: n_grid=12, 선택 f=0.7 α=1.0 (val R@20 0.15867); 최종 로그 dial과 일치 여부는 table9/rank per-seed 로그명 참조
- Amazon-VG/SimGCL: n_grid=12, 선택 f=0.7 α=0.75 (val R@20 0.13328); 최종 로그 dial과 일치 여부는 table9/rank per-seed 로그명 참조
- Amazon-Movies/MF: n_grid=12, 선택 f=0.3 α=0.5 (val R@20 0.12979); 최종 로그 dial과 일치 여부는 table9/rank per-seed 로그명 참조
- Amazon-Movies/LightGCN: n_grid=12, 선택 f=0.3 α=0.5 (val R@20 0.12952); 최종 로그 dial과 일치 여부는 table9/rank per-seed 로그명 참조
- Amazon-Movies/SimGCL: n_grid=12, 선택 f=0.5 α=0.5 (val R@20 0.11869); 최종 로그 dial과 일치 여부는 table9/rank per-seed 로그명 참조
- Douban-movie/MF: n_grid=12, 선택 f=0.9 α=0.5 (val R@20 0.16027); 최종 로그 dial과 일치 여부는 table9/rank per-seed 로그명 참조
- Douban-movie/LightGCN: n_grid=12, 선택 f=0.3 α=0.75 (full-val 재평가 R@20 0.17835; 스윕 4000명 부분집합 최고는 f0.5a0.5 .17882이나 선택은 full-val 재평가 기준 — 최종 로그 f0.3a0.75 와 일치)

## 표 9 (VG 절제) — 시드별 값과 3시드 평균 (R@20 / N@20 / nALRP@20)

| 백본 | 변형 | n | s20 | s21 | s22 | 평균 R@20 | 평균 nALRP@20 | rev102 표 |
|---|---|---|---|---|---|---|---|---|
| MF | base | 3 | 0.04886/0.02475/0.699 | 0.05589/0.02607/0.732 | 0.05435/0.02601/0.671 | 0.05303 | 0.7008 | .0530 / .701 |
| MF | SSNS only | 3 | 0.05563/0.02628/0.725 | 0.05518/0.02630/0.673 | 0.05504/0.02619/0.655 | 0.05528 | 0.6844 | .0553 / .684 |
| MF | Fresh channel only | 3 | 0.10531/0.05086/0.700 | 0.10582/0.04990/0.703 | 0.10505/0.05130/0.700 | 0.10539 | 0.7010 | .1054 / .701 |
| MF | STFR (shared gain) | 3 | 0.10779/0.05288/0.661 | 0.10733/0.05261/0.663 | 0.11075/0.05312/0.654 | 0.10862 | 0.6595 | .1086 / .659 |
| MF | STFR | 3 | 0.11053/0.05513/0.602 | 0.10684/0.05070/0.610 | 0.11334/0.05547/0.573 | 0.11024 | 0.5949 | .1102 / .595 |
| LightGCN | base | 3 | 0.06852/0.03108/0.772 | 0.06776/0.03110/0.773 | 0.06818/0.03098/0.772 | 0.06815 | 0.7722 | .0682 / .772 |
| LightGCN | SSNS only | 3 | 0.07267/0.03536/0.690 | 0.07871/0.03693/0.689 | 0.07564/0.03691/0.690 | 0.07567 | 0.6898 | .0757 / .690 |
| LightGCN | Fresh channel only | 3 | 0.10273/0.04826/0.754 | 0.10334/0.04857/0.752 | 0.10177/0.04696/0.753 | 0.10261 | 0.7530 | .1026 / .753 |
| LightGCN | STFR (shared gain) | 3 | 0.11185/0.05757/0.704 | 0.11295/0.05700/0.706 | 0.11266/0.05613/0.707 | 0.11249 | 0.7056 | .1125 / .706 |
| LightGCN | STFR | 3 | 0.12682/0.06198/0.607 | 0.11807/0.06231/0.607 | 0.12003/0.06171/0.643 | 0.12164 | 0.6191 | .1216 / .619 |
| SimGCL | base | 3 | 0.07281/0.03513/0.722 | 0.07623/0.03735/0.722 | 0.07538/0.03585/0.719 | 0.07481 | 0.7211 | .0748 / .721 |
| SimGCL | SSNS only | 3 | 0.07439/0.03607/0.712 | 0.07984/0.03844/0.715 | 0.07559/0.03609/0.714 | 0.07661 | 0.7136 | .0758 / .715 **불일치** |
| SimGCL | Fresh channel only | 3 | 0.10199/0.04891/0.735 | 0.11451/0.05364/0.732 | 0.10467/0.05097/0.721 | 0.10706 | 0.7293 | .1071 / .729 |
| SimGCL | STFR (shared gain) | 3 | 0.10508/0.04971/0.731 | 0.10730/0.05259/0.733 | 0.10558/0.05089/0.733 | 0.10599 | 0.7322 | .1060 / .732 |
| SimGCL | STFR | 3 | 0.11249/0.05259/0.692 | 0.11237/0.05381/0.692 | 0.11526/0.05519/0.690 | 0.11337 | 0.6911 | .1134 / .691 |

설정 확인: SSNS only·Fresh only·shared gain·STFR 모두 minmax/NO_STALE/200ep/pat15/eval3 (scripts/run_sfdmm_nob_stageB.sh, scripts/run_global_wi_nob_vg.sh). SSNS only 와 shared gain 은 챔피언 (f, α) 를 상속(별도 탐색 없음). base 는 v3 finals.

## 표 10 (최근 N블록) — 후보 N 전체의 검증·테스트 (3시드 평균, K=20)

| 백본 | arm | N | n | val R@20 | test R@20 | test N@20 | nALRP@20 | 선택 |
|---|---|---|---|---|---|---|---|---|
| LightGCN | recent | 1 | 3 | 0.07102 | 0.04914 | 0.02452 | 0.540 |  |
| LightGCN | recent | 2 | 3 | 0.08622 | 0.06603 | 0.03340 | 0.594 |  |
| LightGCN | recent | 3 | 3 | 0.08912 | 0.07094 | 0.03536 | 0.624 |  |
| LightGCN | recent | 6 | 3 | 0.08983 | 0.08001 | 0.03753 | 0.683 | ◀ |
| LightGCN | recent | 12 | 3 | 0.08672 | 0.07115 | 0.03312 | 0.733 |  |
| LightGCN | recent | 24 | 3 | 0.08110 | 0.07166 | 0.03264 | 0.751 |  |
| LightGCN | recent+F | 1 | 3 | 0.10501 | 0.08496 | 0.04039 | 0.676 |  |
| LightGCN | recent+F | 2 | 3 | 0.11666 | 0.10052 | 0.04784 | 0.638 |  |
| LightGCN | recent+F | 3 | 3 | 0.11412 | 0.10188 | 0.04839 | 0.682 |  |
| LightGCN | recent+F | 6 | 3 | 0.13349 | 0.11678 | 0.05869 | 0.674 |  |
| LightGCN | recent+F | 12 | 3 | 0.13790 | 0.11449 | 0.05644 | 0.711 | ◀ |
| LightGCN | recent+F | 24 | 3 | 0.13348 | 0.11485 | 0.05499 | 0.730 |  |
| MF | recent | 1 | 3 | 0.03127 | 0.02070 | 0.01001 | 0.544 |  |
| MF | recent | 2 | 3 | 0.04490 | 0.03311 | 0.01606 | 0.594 |  |
| MF | recent | 3 | 3 | 0.05139 | 0.03780 | 0.01802 | 0.620 |  |
| MF | recent | 6 | 3 | 0.06901 | 0.04843 | 0.02231 | 0.641 |  |
| MF | recent | 12 | 3 | 0.07334 | 0.05452 | 0.02625 | 0.692 | ◀ |
| MF | recent | 24 | 3 | 0.07049 | 0.05586 | 0.02602 | 0.684 |  |
| MF | recent+F | 1 | 3 | 0.09024 | 0.07998 | 0.03709 | 0.697 |  |
| MF | recent+F | 2 | 3 | 0.09390 | 0.08141 | 0.03768 | 0.705 |  |
| MF | recent+F | 3 | 3 | 0.09687 | 0.08321 | 0.03940 | 0.706 |  |
| MF | recent+F | 6 | 3 | 0.11131 | 0.09178 | 0.04381 | 0.714 |  |
| MF | recent+F | 12 | 3 | 0.12230 | 0.10223 | 0.04934 | 0.703 |  |
| MF | recent+F | 24 | 3 | 0.12859 | 0.10807 | 0.05250 | 0.700 | ◀ |
| SimGCL | recent | 1 | 3 | 0.04487 | 0.02427 | 0.01306 | 0.414 |  |
| SimGCL | recent | 2 | 3 | 0.06545 | 0.04696 | 0.02444 | 0.472 |  |
| SimGCL | recent | 3 | 3 | 0.07338 | 0.05566 | 0.02846 | 0.509 |  |
| SimGCL | recent | 6 | 3 | 0.08376 | 0.06241 | 0.03032 | 0.583 |  |
| SimGCL | recent | 12 | 3 | 0.09397 | 0.07284 | 0.03517 | 0.653 | ◀ |
| SimGCL | recent | 24 | 3 | 0.09272 | 0.07562 | 0.03597 | 0.687 |  |
| SimGCL | recent+F | 1 | 3 | 0.09746 | 0.07822 | 0.03783 | 0.672 |  |
| SimGCL | recent+F | 2 | 3 | 0.10552 | 0.08976 | 0.04479 | 0.676 |  |
| SimGCL | recent+F | 3 | 3 | 0.10684 | 0.09302 | 0.04624 | 0.680 |  |
| SimGCL | recent+F | 6 | 3 | 0.11700 | 0.10328 | 0.04716 | 0.687 |  |
| SimGCL | recent+F | 12 | 3 | 0.12356 | 0.10898 | 0.05127 | 0.706 |  |
| SimGCL | recent+F | 24 | 3 | 0.12406 | 0.10632 | 0.05116 | 0.717 | ◀ |

선택 규칙: arm별로 val R@20 3시드 평균 최대 N (collect_recentN_nob.py). 선택 N: recent 12/6/12, recent+F 24/12/24 = rev102 표 10과 일치. 설정: recent = base + TRAIN_RECENT_BLOCKS=N; recent+F = minmax/NO_STALE/NEG_POP_FRAC=0 + TRAIN_RECENT_BLOCKS=N; 200ep/pat15/eval3 (stageB). 세 시드 모두 존재(각 N×arm×백본 n=3).

## 표 11 (샘플러 교체) — 스윕 선택과 finals 시드별

스윕(seed 20, 30ep, pat15, eval3; val R@20):
- MF swap DNS M=10: val 0.12153 (best ep 27)
- MF swap DNS M=2: val 0.12266 (best ep 29)
- MF swap DNS M=5: val 0.12760 (best ep 29) ◀
- MF swap AUC-NS gamma=0.002: val 0.11493 (best ep 24)
- MF swap AUC-NS gamma=0.006: val 0.11535 (best ep 24) ◀
- MF swap AUC-NS gamma=0.018: val 0.11348 (best ep 24)
- MF swap FairNeg -=default: val 0.11687 (best ep 27) ◀
- LightGCN swap DNS M=10: val 0.14106 (best ep 18) ◀
- LightGCN swap DNS M=2: val 0.13307 (best ep 29)
- LightGCN swap DNS M=5: val 0.14104 (best ep 15)
- LightGCN swap AUC-NS gamma=0.002: val 0.14483 (best ep 29) ◀
- LightGCN swap AUC-NS gamma=0.006: val 0.14210 (best ep 24)
- LightGCN swap AUC-NS gamma=0.018: val 0.14159 (best ep 24)
- LightGCN swap FairNeg -=default: val 0.12947 (best ep 27) ◀
- SimGCL swap DNS M=10: val 0.12385 (best ep 6)
- SimGCL swap DNS M=2: val 0.12497 (best ep 6)
- SimGCL swap DNS M=5: val 0.12830 (best ep 6) ◀
- SimGCL swap AUC-NS gamma=0.002: val 0.12710 (best ep 3) ◀
- SimGCL swap AUC-NS gamma=0.006: val 0.12634 (best ep 3)
- SimGCL swap AUC-NS gamma=0.018: val 0.12685 (best ep 9)
- SimGCL swap FairNeg -=default: val 0.12379 (best ep 6) ◀

| 백본 | arm | 설정 | n | s20 | s21 | s22 | 평균 R@20 | 평균 nALRP@20 |
|---|---|---|---|---|---|---|---|---|
| MF | base (참조) | - | 3 | 0.04886/0.699 | 0.05589/0.732 | 0.05435/0.671 | 0.05303 | 0.7008 |
| MF | swap DNS | M=5 | 3 | 0.10915/0.679 | 0.11395/0.685 | 0.11538/0.680 | 0.11283 | 0.6812 |
| MF | swap AUC-NS | gamma=0.006 | 3 | 0.10439/0.676 | 0.10954/0.658 | 0.09895/0.663 | 0.10429 | 0.6655 |
| MF | swap FairNeg | default | 3 | 0.10447/0.713 | 0.10594/0.723 | 0.10300/0.720 | 0.10447 | 0.7188 |
| MF | SFD-noB champion | f0.7a0.75 | 3 | 0.11053/0.602 | 0.10684/0.610 | 0.11334/0.573 | 0.11024 | 0.5949 |
| LightGCN | base (참조) | - | 3 | 0.06852/0.772 | 0.06776/0.773 | 0.06818/0.772 | 0.06815 | 0.7722 |
| LightGCN | swap DNS | M=10 | 3 | 0.12121/0.689 | 0.12011/0.702 | 0.11378/0.705 | 0.11837 | 0.6986 |
| LightGCN | swap AUC-NS | gamma=0.002 | 3 | 0.12010/0.659 | 0.11872/0.670 | 0.11908/0.679 | 0.11930 | 0.6690 |
| LightGCN | swap FairNeg | default | 3 | 0.11232/0.738 | 0.11171/0.729 | 0.11223/0.742 | 0.11209 | 0.7366 |
| LightGCN | SFD-noB champion | f0.7a1.0 | 3 | 0.12682/0.607 | 0.11807/0.607 | 0.12003/0.643 | 0.12164 | 0.6191 |
| SimGCL | base (참조) | - | 3 | 0.07281/0.722 | 0.07623/0.722 | 0.07538/0.719 | 0.07481 | 0.7211 |
| SimGCL | swap DNS | M=5 | 3 | 0.10618/0.723 | 0.10683/0.740 | 0.10643/0.719 | 0.10648 | 0.7277 |
| SimGCL | swap AUC-NS | gamma=0.002 | 3 | 0.10172/0.745 | 0.10991/0.722 | 0.10595/0.718 | 0.10586 | 0.7282 |
| SimGCL | swap FairNeg | default | 3 | 0.10273/0.737 | 0.10878/0.734 | 0.10304/0.734 | 0.10485 | 0.7349 |
| SimGCL | SFD-noB champion | f0.7a0.75 | 3 | 0.11249/0.692 | 0.11237/0.692 | 0.11526/0.690 | 0.11337 | 0.6911 |

설정 확인: swap arm = minmax/NO_STALE + NEG_SAMPLER_FRAC=챔피언 f, finals 200ep/pat15/eval3 (chain_axis2_v3_nob_vg.sh). 세 시드 모두 존재. rev102 표 11 값과 일치.