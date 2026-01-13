1. Działania arytmetyczne są wykonywane na liczbach naturalnych, wynikiem odejmowa-
nia liczby większej od mniejszej jest 0, ponadto dzielenie przez zero powinno dać wynik
0 i resztę także 0.
2. Deklaracja tab[10:30] oznacza zadeklarowanie tablicy o 21 elementach indeksowanych
od 10 do 30; identyfikator tab[i] oznacza odwołanie do i-tego elementu tablicy tab;
deklaracja zawierająca pierwszą liczbę większą od drugiej powinna być zgłaszana jako
błąd.
3. Procedury nie mogą zawierać wywołań rekurencyjnych, parametry formalne przeka-
zywane są przez referencje (parametry IN-OUT, tryb domyślny), zmienne używane w
procedurze muszą być jej parametrami formalnymi lub być zadeklarowane wewnątrz
procedury; nazwa tablicy w parametrach formalnych powinna być poprzedzona literą
T.
4. W procedurze można wywołać tylko procedury zdefiniowane wcześniej w kodzie pro-
gramu, a jako ich parametry formalne można podać zarówno parametry formalne
procedury wywołującej, jak i jej zmienne lokalne.
5. Poprzedzenie w parametrach formalnych zmiennej literą I oznacza, że ma być trak-
towana w procedurze jak stała (nie wolno jej modyfikować, może być przekazana do
podprocedury tylko w miejscu także oznaczonym przez I). Poprzedzenie przez O ozna-
cza, że zmienna ma nieokreśloną wartość (nie wolno jej czytać przed podstawieniem
wartości, nie może być przekazana do podprocedury w miejscu oznaczonym przez I).
6. Pętla FOR ma iterator lokalny, przyjmujący wartości od wartości stojącej po FROM do
wartości stojącej po TO/DOWNTO kolejno w odstępie +1 lub odpowiednio w odstępie -1;
liczba iteracji pętli FOR jest ustalana na początku i nie podlega zmianie w trakcie wyko-
nywania pętli (nawet jeśli zmieniają się wartości zmiennych wyznaczających początek
i koniec pętli); iterator pętli FOR nie może być modyfikowany wewnątrz pętli (kompilator
w takim przypadku powinien zgłaszać błąd).
7. Pętla REPEAT-UNTIL kończy pracę kiedy warunek napisany za UNTIL jest spełniony (pętla
wykona się przynajmniej raz).
8. Instrukcja READ czyta wartość z zewnątrz i podstawia pod zmienną, a WRITE wypisuje
wartość zmiennej/liczby na zewnątrz.
9. Pozostałe instrukcje są zgodne z ich znaczeniem w większości języków programowania.
10. pidentifier jest opisany wyrażeniem regularnym [_a-z]+.
11. num jest liczbą naturalną w zapisie dziesiętnym (w kodzie wejściowym liczby podawane
jako stałe są ograniczone do typu 64 bitowego, na maszynie wirtualnej nie ma ograni-
czeń na wielkość liczb, obliczenia mogą generować dowolną liczbę naturalną ).
12. Małe i duże litery są rozróżniane.
13. W programie można użyć komentarzy zaczynających się od # i obowiązujących do
końca linii.