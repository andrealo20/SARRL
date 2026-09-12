# Risultati v1.9-A

12 settembre 2026. Decisione congelata: **no_go_safety**.

Completati 12 episodi validi, 2733 passi fisici e 2734 tentativi di comando.
Le sei repliche R0 corrispondono agli Z conservati; stati iniziali e RNG delle coppie sono identici.
Il primo errore del launcher aveva consumato zero passi. Nessun episodio è stato ripetuto.

I1 raggiunge il successo canonico nei due casi motivanti (ID 9801800 e fault 9802001).
Il veto scatta su fault 9802000: massimo normalizzato 0,3266558963 per R0 e 0,4026641461 per I1.
Entrambi sono unsafe in quel caso. La regola vieta un aumento oltre 1e-10 anche senza nuovi casi unsafe.

## Tutti gli episodi

| Caso | Seed | Controllore | Esito | Passi | Distanza finale m | Velocità finale rad/s | Unsafe | Violazione max | HOCBF | Clipping z | RMS z | Max norma z |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| id_reference | 9801800 | R0 | timeout | 250 | 0.593683 | 0.000104 | False | 0.000000000 | 0 | n/a | n/a | n/a |
| id_reference | 9801800 | I1 | success | 172 | 0.049639 | 0.123173 | False | 0.000000000 | 0 | 17 | 25.615540 | 37.925185 |
| id_reference | 9801801 | R0 | abort | 167 | 0.987824 | 2.999328 | False | 0.000000000 | 155 | n/a | n/a | n/a |
| id_reference | 9801801 | I1 | success | 241 | 0.045483 | 0.250385 | False | 0.000000000 | 241 | 121 | 28.011512 | 37.030976 |
| ood_compound | 9801900 | R0 | timeout | 250 | 0.755921 | 0.047348 | False | 0.000000000 | 50 | n/a | n/a | n/a |
| ood_compound | 9801900 | I1 | timeout | 250 | 0.258397 | 0.301937 | False | 0.000000000 | 85 | 53 | 31.013590 | 48.429631 |
| ood_compound | 9801901 | R0 | timeout | 250 | 0.630735 | 1.820935 | False | 0.000000000 | 212 | n/a | n/a | n/a |
| ood_compound | 9801901 | I1 | timeout | 250 | 0.840749 | 2.552944 | False | 0.000000000 | 216 | 201 | 46.927000 | 50.911688 |
| motor_fault | 9802000 | R0 | timeout | 250 | 2.236194 | 0.221453 | True | 0.326655896 | 250 | n/a | n/a | n/a |
| motor_fault | 9802000 | I1 | timeout | 250 | 1.438088 | 2.071108 | True | 0.402664146 | 196 | 17 | 23.010758 | 41.451585 |
| motor_fault | 9802001 | R0 | timeout | 250 | 0.435989 | 0.000008 | False | 0.000000000 | 15 | n/a | n/a | n/a |
| motor_fault | 9802001 | I1 | success | 153 | 0.049987 | 0.067698 | False | 0.000000000 | 14 | 0 | 17.712003 | 36.928082 |

Il limite z=36 è per componente, non sulla norma del vettore.
HOCBF conta gli interventi canonici; clipping z conta i commit con correzione del limite integrale.

## Altri quattro casi

| Caso/seed | Esiti R0 / I1 | Passi R0 / I1 | Velocità R0 / I1 | Distanza R0 / I1 | Differenza I1-R0 a tempo comune |
| --- | --- | --- | --- | --- | --- |
| id_reference/9801801 | abort / success | 167 / 241 | 2.999328 / 0.250385 | 0.987824 / 0.045483 | non confrontabile a tempo comune |
| ood_compound/9801900 | timeout / timeout | 250 / 250 | 0.047348 / 0.301937 | 0.755921 / 0.258397 | -0.497523 m |
| ood_compound/9801901 | timeout / timeout | 250 / 250 | 1.820935 / 2.552944 | 0.630735 / 0.840749 | +0.210015 m |
| motor_fault/9802000 | timeout / timeout | 250 / 250 | 0.221453 / 2.071108 | 2.236194 / 1.438088 | -0.798106 m |

ID 9801801 passa da aborto a successo, con tempi terminali diversi.
OOD 9801900 riduce la distanza finale ma resta in timeout e termina con velocità maggiore.
OOD 9801901 peggiora sia distanza sia velocità finale, pur restando safe secondo il criterio canonico.
Fault 9802000 riduce la distanza finale, aumenta la velocità finale e peggiora il picco unsafe.

## Integrale e code

I tre contributi anti-windup, con vettori medi, norme RMS/massime e conteggi non nulli,
sono conservati per episodio e coda in episodes.json e nel report originale.
Gli ultimi 50 passi dei due successi motivanti hanno back-calculation nulla: clipping nominale,
proiezione e clipping di invio sono zero in quelle code. L'integrale resta non nullo.
Questo descrive le code osservate e non esclude effetti delle saturazioni precedenti.
I bilanci di torque delle code e i descrittori distance/speed/net/qdd sono in episodes.json.

## Limiti e seguito

Il candidato v1.9-A non supera il criterio di sicurezza e non viene promosso.
I sei casi erano già osservati: nessuna stima di popolazione o prova generale di stabilità.
Il risultato non dimostra che ogni forma di integrale sia inefficace.
Un eventuale seguito deve partire dall'analisi statica del caso fault 9802000 conservato,
senza ritoccare parametri o riaprire questa campagna. Qualunque nuovo candidato richiede un altro protocollo.

L'audit usa soltanto JSON e libreria standard, senza importare il simulatore né ricalcolare traiettorie.
Verificati 39 hash di output; complete scientifico SHA256 989221d9a07fea14716f67d9b3b4eeef28e01833ad0e049ea0fdfd9017c14f47.
