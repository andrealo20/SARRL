# Diagnosi residuale: tabelle per caso e modello

Analisi descrittiva di 60 episodi L e sei controlli Z condivisi. Nessuna stima di popolazione.

| Scenario | Seed | Z: esito, distanza finale (m) | P0: esiti, distanza min/max (m) | P1: esiti, distanza min/max (m) |
| --- | --- | --- | --- | --- |
| id_reference | 9801800 | timeout, 0.593683 | {'timeout': 5}; 0.060470/0.353271 | {'timeout': 5}; 0.079191/0.206467 |
| id_reference | 9801801 | abort, 0.987824 | {'timeout': 4, 'abort': 1}; 0.333850/1.101501 | {'timeout': 4, 'abort': 1}; 0.120603/1.215861 |
| motor_fault | 9802000 | timeout, 2.236194 | {'timeout': 5}; 2.008847/2.488908 | {'timeout': 5}; 2.057791/2.408975 |
| motor_fault | 9802001 | timeout, 0.435989 | {'success': 2, 'timeout': 3}; 0.014738/0.496636 | {'timeout': 3, 'success': 2}; 0.047786/0.124094 |
| ood_compound | 9801900 | timeout, 0.755921 | {'timeout': 5}; 0.311126/1.912238 | {'timeout': 5}; 0.380895/0.790085 |
| ood_compound | 9801901 | timeout, 0.630735 | {'timeout': 5}; 0.418522/2.313598 | {'timeout': 5}; 0.421592/2.128209 |

## Tutti gli episodi

Le distanze finali possono riferirsi a tempi di terminazione diversi.

| Controller | Scenario | Seed | Esito | Passi | Distanza finale (m) | Arresto fuori bersaglio | Unsafe | Interventi HOCBF |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: |
| Z | id_reference | 9801800 | timeout | 250 | 0.593683 | True | False | 0 |
| P0_inloop_reference_train_seed_30 | id_reference | 9801800 | timeout | 250 | 0.353271 | True | False | 30 |
| P0_inloop_reference_train_seed_31 | id_reference | 9801800 | timeout | 250 | 0.193257 | True | False | 29 |
| P0_inloop_reference_train_seed_32 | id_reference | 9801800 | timeout | 250 | 0.310747 | True | False | 28 |
| P0_inloop_reference_train_seed_33 | id_reference | 9801800 | timeout | 250 | 0.135762 | True | False | 26 |
| P0_inloop_reference_train_seed_34 | id_reference | 9801800 | timeout | 250 | 0.060470 | True | False | 24 |
| P1_inloop_half_penalty_train_seed_30 | id_reference | 9801800 | timeout | 250 | 0.176475 | False | True | 28 |
| P1_inloop_half_penalty_train_seed_31 | id_reference | 9801800 | timeout | 250 | 0.135918 | True | True | 27 |
| P1_inloop_half_penalty_train_seed_32 | id_reference | 9801800 | timeout | 250 | 0.079191 | True | True | 29 |
| P1_inloop_half_penalty_train_seed_33 | id_reference | 9801800 | timeout | 250 | 0.206467 | True | False | 31 |
| P1_inloop_half_penalty_train_seed_34 | id_reference | 9801800 | timeout | 250 | 0.169280 | True | False | 29 |
| Z | id_reference | 9801801 | abort | 167 | 0.987824 | False | False | 155 |
| P0_inloop_reference_train_seed_30 | id_reference | 9801801 | timeout | 250 | 0.721209 | False | True | 250 |
| P0_inloop_reference_train_seed_31 | id_reference | 9801801 | timeout | 250 | 0.333850 | False | True | 250 |
| P0_inloop_reference_train_seed_32 | id_reference | 9801801 | abort | 172 | 0.866337 | False | False | 172 |
| P0_inloop_reference_train_seed_33 | id_reference | 9801801 | timeout | 250 | 0.960504 | False | True | 184 |
| P0_inloop_reference_train_seed_34 | id_reference | 9801801 | timeout | 250 | 1.101501 | False | True | 207 |
| P1_inloop_half_penalty_train_seed_30 | id_reference | 9801801 | timeout | 250 | 0.662273 | False | True | 240 |
| P1_inloop_half_penalty_train_seed_31 | id_reference | 9801801 | timeout | 250 | 0.120603 | False | True | 235 |
| P1_inloop_half_penalty_train_seed_32 | id_reference | 9801801 | timeout | 250 | 0.708024 | False | True | 250 |
| P1_inloop_half_penalty_train_seed_33 | id_reference | 9801801 | abort | 169 | 1.215861 | False | False | 153 |
| P1_inloop_half_penalty_train_seed_34 | id_reference | 9801801 | timeout | 250 | 0.146408 | False | True | 198 |
| Z | ood_compound | 9801900 | timeout | 250 | 0.755921 | False | False | 50 |
| P0_inloop_reference_train_seed_30 | ood_compound | 9801900 | timeout | 250 | 1.912238 | False | True | 236 |
| P0_inloop_reference_train_seed_31 | ood_compound | 9801900 | timeout | 250 | 0.311126 | False | False | 94 |
| P0_inloop_reference_train_seed_32 | ood_compound | 9801900 | timeout | 250 | 0.381219 | False | False | 104 |
| P0_inloop_reference_train_seed_33 | ood_compound | 9801900 | timeout | 250 | 0.486499 | False | False | 100 |
| P0_inloop_reference_train_seed_34 | ood_compound | 9801900 | timeout | 250 | 0.540476 | False | False | 108 |
| P1_inloop_half_penalty_train_seed_30 | ood_compound | 9801900 | timeout | 250 | 0.490301 | False | False | 87 |
| P1_inloop_half_penalty_train_seed_31 | ood_compound | 9801900 | timeout | 250 | 0.380895 | False | False | 87 |
| P1_inloop_half_penalty_train_seed_32 | ood_compound | 9801900 | timeout | 250 | 0.394355 | False | False | 84 |
| P1_inloop_half_penalty_train_seed_33 | ood_compound | 9801900 | timeout | 250 | 0.403386 | False | False | 81 |
| P1_inloop_half_penalty_train_seed_34 | ood_compound | 9801900 | timeout | 250 | 0.790085 | False | False | 79 |
| Z | ood_compound | 9801901 | timeout | 250 | 0.630735 | False | False | 212 |
| P0_inloop_reference_train_seed_30 | ood_compound | 9801901 | timeout | 250 | 2.313598 | False | False | 232 |
| P0_inloop_reference_train_seed_31 | ood_compound | 9801901 | timeout | 250 | 0.670014 | False | False | 231 |
| P0_inloop_reference_train_seed_32 | ood_compound | 9801901 | timeout | 250 | 0.418522 | False | False | 213 |
| P0_inloop_reference_train_seed_33 | ood_compound | 9801901 | timeout | 250 | 1.276085 | False | False | 215 |
| P0_inloop_reference_train_seed_34 | ood_compound | 9801901 | timeout | 250 | 0.748851 | False | False | 216 |
| P1_inloop_half_penalty_train_seed_30 | ood_compound | 9801901 | timeout | 250 | 1.243679 | False | False | 212 |
| P1_inloop_half_penalty_train_seed_31 | ood_compound | 9801901 | timeout | 250 | 2.128209 | False | False | 241 |
| P1_inloop_half_penalty_train_seed_32 | ood_compound | 9801901 | timeout | 250 | 0.421592 | False | False | 234 |
| P1_inloop_half_penalty_train_seed_33 | ood_compound | 9801901 | timeout | 250 | 0.931168 | False | False | 221 |
| P1_inloop_half_penalty_train_seed_34 | ood_compound | 9801901 | timeout | 250 | 0.607901 | False | False | 222 |
| Z | motor_fault | 9802000 | timeout | 250 | 2.236194 | False | True | 250 |
| P0_inloop_reference_train_seed_30 | motor_fault | 9802000 | timeout | 250 | 2.092493 | False | True | 247 |
| P0_inloop_reference_train_seed_31 | motor_fault | 9802000 | timeout | 250 | 2.488908 | False | True | 244 |
| P0_inloop_reference_train_seed_32 | motor_fault | 9802000 | timeout | 250 | 2.427696 | False | True | 250 |
| P0_inloop_reference_train_seed_33 | motor_fault | 9802000 | timeout | 250 | 2.008847 | False | True | 244 |
| P0_inloop_reference_train_seed_34 | motor_fault | 9802000 | timeout | 250 | 2.376876 | False | True | 242 |
| P1_inloop_half_penalty_train_seed_30 | motor_fault | 9802000 | timeout | 250 | 2.083090 | False | True | 239 |
| P1_inloop_half_penalty_train_seed_31 | motor_fault | 9802000 | timeout | 250 | 2.359347 | False | True | 215 |
| P1_inloop_half_penalty_train_seed_32 | motor_fault | 9802000 | timeout | 250 | 2.057791 | False | True | 250 |
| P1_inloop_half_penalty_train_seed_33 | motor_fault | 9802000 | timeout | 250 | 2.368698 | False | True | 249 |
| P1_inloop_half_penalty_train_seed_34 | motor_fault | 9802000 | timeout | 250 | 2.408975 | False | True | 243 |
| Z | motor_fault | 9802001 | timeout | 250 | 0.435989 | True | False | 15 |
| P0_inloop_reference_train_seed_30 | motor_fault | 9802001 | success | 66 | 0.014738 | False | False | 17 |
| P0_inloop_reference_train_seed_31 | motor_fault | 9802001 | timeout | 250 | 0.116187 | True | False | 15 |
| P0_inloop_reference_train_seed_32 | motor_fault | 9802001 | timeout | 250 | 0.217860 | True | False | 18 |
| P0_inloop_reference_train_seed_33 | motor_fault | 9802001 | timeout | 250 | 0.496636 | True | False | 15 |
| P0_inloop_reference_train_seed_34 | motor_fault | 9802001 | success | 78 | 0.047264 | False | False | 17 |
| P1_inloop_half_penalty_train_seed_30 | motor_fault | 9802001 | timeout | 250 | 0.120358 | True | False | 14 |
| P1_inloop_half_penalty_train_seed_31 | motor_fault | 9802001 | success | 67 | 0.047786 | False | False | 18 |
| P1_inloop_half_penalty_train_seed_32 | motor_fault | 9802001 | timeout | 250 | 0.124094 | True | False | 18 |
| P1_inloop_half_penalty_train_seed_33 | motor_fault | 9802001 | success | 113 | 0.049555 | False | False | 14 |
| P1_inloop_half_penalty_train_seed_34 | motor_fault | 9802001 | timeout | 250 | 0.090154 | True | False | 14 |

## Arresti fuori bersaglio: ultime 50 transizioni

Norme RMS dei vettori. Torque e carichi in N m; accelerazione in rad/s². Gli addendi sono un bilancio contabile, senza attribuzione causale separata.

| Controller | Caso | Distanza media (m) | Velocità max (rad/s) | qdd RMS | Net RMS | Residuo RMS | Proiezione RMS | Saturazione nominale RMS | Mismatch carico RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Z | id_reference/9801800 | 0.593579 | 0.000718064 | 0.00213097 | 0.00139155 | 0 | 0 | 0 | 8.54616 |
| P0_inloop_reference_train_seed_30 | id_reference/9801800 | 0.352379 | 0.0129124 | 0.0105227 | 0.0115693 | 7.09495 | 0 | 0 | 9.05841 |
| P0_inloop_reference_train_seed_31 | id_reference/9801800 | 0.192464 | 0.00687915 | 0.0103933 | 0.0108687 | 7.61925 | 0 | 0 | 7.74027 |
| P0_inloop_reference_train_seed_32 | id_reference/9801800 | 0.310393 | 0.00654195 | 0.0220116 | 0.0220305 | 5.11138 | 0 | 0 | 8.17131 |
| P0_inloop_reference_train_seed_33 | id_reference/9801800 | 0.13576 | 0.000153031 | 0.000346396 | 0.000600109 | 5.79612 | 0 | 0 | 8.51788 |
| P0_inloop_reference_train_seed_34 | id_reference/9801800 | 0.0604869 | 0.0127634 | 0.0449366 | 0.0402546 | 7.48425 | 0 | 0 | 8.43673 |
| P1_inloop_half_penalty_train_seed_31 | id_reference/9801800 | 0.13593 | 0.000179327 | 0.000388722 | 0.000555399 | 10.9288 | 0 | 0 | 8.2313 |
| P1_inloop_half_penalty_train_seed_32 | id_reference/9801800 | 0.079191 | 4.36807e-05 | 0.000236256 | 0.000184168 | 8.77095 | 0 | 0 | 8.70548 |
| P1_inloop_half_penalty_train_seed_33 | id_reference/9801800 | 0.206546 | 0.00201457 | 0.00857914 | 0.00996893 | 7.58337 | 0 | 0 | 7.70808 |
| P1_inloop_half_penalty_train_seed_34 | id_reference/9801800 | 0.1716 | 0.0159338 | 0.0485813 | 0.115144 | 5.77571 | 0 | 0 | 8.0332 |
| Z | motor_fault/9802001 | 0.43598 | 0.000124234 | 0.000317958 | 0.000164309 | 0 | 0 | 0 | 1.48243 |
| P0_inloop_reference_train_seed_31 | motor_fault/9802001 | 0.116187 | 3.70121e-06 | 5.82117e-06 | 2.9808e-06 | 5.9263 | 0 | 0 | 1.58179 |
| P0_inloop_reference_train_seed_32 | motor_fault/9802001 | 0.217858 | 0.000143205 | 0.000220576 | 0.000101162 | 7.75376 | 0 | 0 | 1.56986 |
| P0_inloop_reference_train_seed_33 | motor_fault/9802001 | 0.489511 | 0.0330465 | 0.0288585 | 0.0146501 | 1.5627 | 0 | 0 | 1.44806 |
| P1_inloop_half_penalty_train_seed_30 | motor_fault/9802001 | 0.120354 | 6.40878e-05 | 7.01577e-05 | 5.50403e-05 | 8.29554 | 0 | 0 | 1.5532 |
| P1_inloop_half_penalty_train_seed_32 | motor_fault/9802001 | 0.124094 | 5.06329e-08 | 6.83978e-07 | 4.57241e-07 | 6.98552 | 0 | 0 | 1.57113 |
| P1_inloop_half_penalty_train_seed_34 | motor_fault/9802001 | 0.0901538 | 3.48418e-06 | 5.52461e-06 | 2.81564e-06 | 5.01466 | 0 | 0 | 1.5948 |
