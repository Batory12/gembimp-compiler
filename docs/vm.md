Maszyna wirtualna składa się z 8 rejestrów (ra, rb, rc, rd, re, rf , rg, rh), licznika rozkazów k
oraz ciągu komórek pamięci pi, dla i = 0, 1, 2, ... (z przyczyn technicznych i ⩽ 262). Maszyna
pracuje na liczbach naturalnych.
Program maszyny składa się z ciągu rozkazów, który niejawnie numerujemy od zera.
W kolejnych krokach wykonujemy zawsze rozkaz o numerze k aż napotkamy instrukcję
HALT. Początkowa zawartość rejestrów i komórek pamięci jest nieokreślona, a licznik rozka-
zów k ma wartość 0. W Tabeli 2 jest podana lista rozkazów wraz z ich interpretacją i kosztem
wykonania. W programie można zamieszczać komentarze w postaci: # komentarz, które się-
gają do końca linii. Białe znaki w kodzie są pomijane. Przejście do nieistniejącego rozkazu
lub wywołanie nieistniejącego rejestru jest traktowane jako błąd.
Rozkaz Interpretacja Czas
READ pobraną liczbę zapisuje w rejestrze ra oraz k ← k + 1 100
WRITE wyświetla zawartość rejestru ra oraz k ← k + 1 100
LOAD j ra ← pj oraz k ← k + 1 50
STORE j pj ← ra oraz k ← k + 1 50
RLOAD x ra ← prx oraz k ← k + 1 50
RSTORE x prx ← ra oraz k ← k + 1 50
ADD x ra ← ra + rx oraz k ← k + 1 5
SUB x ra ← max{ra − rx, 0} oraz k ← k + 1 5
SWP x ra ↔ rx oraz k ← k + 1 5
RST x rx ← 0 oraz k ← k + 1 1
INC x rx ← rx + 1 oraz k ← k + 1 1
DEC x rx ← max{rx − 1, 0} oraz k ← k + 1 1
SHL x rx ← 2 ∗ rx oraz k ← k + 1 1
SHR x rx ← ⌊rx/2⌋ oraz k ← k + 1 1
JUMP j k ← j 1
JPOS j jeśli ra > 0 to k ← j, w p.p. k ← k + 1 1
JZERO j jeśli ra = 0 to k ← j, w p.p. k ← k + 1 1
CALL j ra ← k + 1 oraz k ← j 1
RTRN k ← ra 1
HALT zatrzymaj program 0