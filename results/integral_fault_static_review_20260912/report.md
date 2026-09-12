# Fault 9802000: ricostruzione statica

12 settembre 2026. Solo dati conservati; decisione v1.9-A invariata: no_go_safety.

k è il tentativo zero-based e coincide col numero di passi già eseguiti. L'osservazione k+1 segue quel passo.
Accelerazioni, margini e contributi sono istantanei allo stato pre-passo; non sono un replay RK4.

## Cronologia I1

| k | q2 rad | v2 rad/s | z2 | Nominale qdd2 | Reale qdd2 | Margine nominale limite alto | Margine reale | Proiezione Nm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 19 | 0.247567 | 1.682793 | 9.113616 | 53.232893 | 6.738586 | 0.000000 | 46.494308 | 1.128524 |
| 20 | 0.282545 | 1.813676 | 9.507575 | 51.049614 | -27.027866 | 0.000000 | 78.077480 | 1.136935 |
| 63 | -0.529040 | 6.254000 | 16.761591 | 26.935992 | 81.612293 | 0.000000 | -54.676301 | 31.399601 |
| 64 | -0.386284 | 8.085689 | 17.150849 | -54.284429 | -6.298399 | 59.334648 | 11.348618 | 38.980016 |
| 67 | 0.005624 | 3.400017 | 12.020423 | 42.109239 | -118.417142 | -0.000000 | 160.526381 | 25.246367 |
| 68 | 0.049670 | 0.996438 | 12.417883 | 65.043872 | -95.219063 | -0.000000 | 160.262935 | 23.401445 |
| 150 | -0.534586 | -3.621647 | 12.761753 | 125.831120 | 58.759686 | 0.000000 | 67.071434 | 3.706624 |
| 180 | 1.229444 | 3.352393 | 17.295294 | 11.989974 | -0.197190 | 0.000000 | 12.187164 | 1.799130 |
| 195 | 2.220911 | 3.200831 | 19.482656 | -11.281081 | -1.017855 | 0.000000 | -10.263226 | 0.165418 |
| 196 | 2.284769 | 3.187247 | 19.605714 | -12.796121 | -0.891252 | 0.054427 | -11.850442 | 0.000000 |
| 198 | 2.411815 | 3.169294 | 19.793717 | -16.966359 | -0.539380 | 1.228030 | -15.198950 | 0.000000 |
| 199 | 2.475145 | 3.166240 | 19.819146 | -19.184140 | -0.428512 | 1.893116 | -16.862513 | 0.000000 |
| 200 | 2.538438 | 3.165770 | 19.798978 | -21.477212 | -0.294917 | 2.608565 | -18.573729 | 0.000000 |
| 207 | 2.985311 | 3.237272 | 18.377932 | -39.843715 | 0.319423 | 9.088220 | -31.074917 | 0.000000 |
| 208 | 3.050158 | 3.249311 | 17.990444 | -42.710182 | 0.166623 | 10.213109 | -32.663696 | 0.000000 |
| 220 | 3.796615 | 2.712980 | 9.735295 | -71.401793 | -5.972835 | 25.606617 | -39.822341 | 0.000000 |
| 229 | 4.170987 | 1.363578 | -0.218384 | -78.640044 | -8.546092 | 36.979587 | -33.114365 | 0.000000 |
| 236 | 4.276994 | 0.144834 | -9.205852 | -76.818844 | -8.865097 | 44.695646 | -23.258101 | 0.000000 |
| 237 | 4.278126 | -0.031083 | -10.523352 | -76.066066 | -8.799208 | 45.673756 | -21.593102 | 0.000000 |
| 249 | 4.029222 | -1.945707 | -25.689986 | -59.296687 | -6.119216 | 54.273203 | 1.095732 | 0.000000 |

## Differenza di accelerazione, I1

a_reale - a_nominale(h) = clipping invio + ritardo + guadagno + massa + carico.
I contributi usano M_reale^-1 per differenze di torque. Massa confronta le due inverse sullo stesso h-l_nominale.
Carico è M_reale^-1(l_nominale-l_reale). Somma verificata su entrambe le componenti e tutte le 500 transizioni.

| Tentativi | qdd2 nominale media | qdd2 reale media | Invio | Ritardo | Guadagno | Massa | Carico |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [0, 19] | 68.713280 | 4.614624 | 0.000000 | -1.476181 | -25.554598 | -29.881409 | -7.186469 |
| [20, 62] | 57.988151 | 4.444086 | -0.000000 | 0.900035 | -33.811664 | -16.629808 | -4.002627 |
| [63, 69] | 21.583931 | -57.315831 | 0.000000 | -26.880589 | -35.228841 | -20.718233 | 3.927901 |
| [170, 195] | 8.135317 | -0.663868 | 0.000000 | 1.628691 | -4.399614 | -1.455828 | -4.572435 |
| [196, 207] | -25.573086 | -0.127425 | 0.000000 | 2.749719 | 6.269314 | 13.979992 | 2.446637 |
| [208, 237] | -68.965110 | -5.790728 | 0.000000 | 1.426481 | 19.520789 | 37.151480 | 5.075631 |
| [238, 249] | -67.952571 | -7.830986 | 0.000000 | -0.836360 | 21.110859 | 34.228021 | 5.619066 |

## Limiti della ricostruzione

Le finestre sono descrittive e scelte dopo l'esito. I contributi sono un'identità contabile lungo le traiettorie osservate,
non effetti causali di rimozioni isolate: massa, carico, ritardo, guadagno e controllore influenzano lo stato successivo.
Il margine reale sostituisce l'accelerazione osservata nella stessa espressione del filtro. Non risolve un nuovo QP
e non certifica fattibilità di un comando alternativo. Non dimostra che eliminare il solo integrale renderebbe sicura la stessa traiettoria.
Timeline e finestre complete di R0 e I1 sono conservate nei JSON; nessun nuovo controllo o episodio eseguito.
