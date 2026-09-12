# Audit diagnostico locale, 2026-09-05

Analisi esplorativa successiva al risultato ufficiale, non nuovo test di efficacia.
Nessun training, nessun cambiamento al verdetto o ai file ufficiali.

Prima dell'esecuzione si fissano 60 episodi: i dieci best congelati v1.8,
due seed per scenario (9801800..9801801 ID, 9801900..9801901 OOD,
9802000..9802001 fault). Non coincidono con i range ufficiali o smoke v1.8.
Le stesse condizioni iniziali sono condivise fra tutti i modelli.
Il piccolo campione serve a osservare traiettorie, non a stimare efficacia.

Si riutilizzano ambiente, filtro ed evaluator ufficiali senza modificarli.
Si registrano distanza, norma della velocita articolare, azione residuale,
correzione del filtro e fattibilita a ogni tentativo. Le soglie di successo
restano quelle dell'ambiente (0,05 m e 0,35 rad/s). Si descrivono distanza
minima/finale, velocita finale e medie negli ultimi 50 stati fisici.
Un timeout viene distinto da un aborto usando la diagnostica canonica.
I confronti fra fallimenti e successi sono descrittivi e post-selezione.

In parallelo si classificano tutti i 7000 episodi ufficiali e si verificano
gli hash degli artefatti elencati negli inventari locali v1.7/v1.8.
L'integrita sul disco non equivale a un backup su un dispositivo separato.
