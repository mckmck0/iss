# Modelowanie tempomatu pojazdu – opis zastosowanych modeli i założeń (ISS)

## 1. Cel ćwiczenia

Celem ćwiczenia było zaprojektowanie i porównanie działania dwóch algorytmów sterowania tempomatem:

* klasycznego regulatora PID,
* regulatora rozmytego typu Mamdaniego,

dla różnych typów pojazdów oraz warunków ruchu (zadana prędkość, nachylenie drogi).
Porównanie wykonano na podstawie przebiegów prędkości oraz sił działających na pojazd.

---

## 2. Model fizyczny pojazdu

### 2.1. Równanie ruchu

Pojazd opisano jednowymiarowym modelem ruchu wzdłużnego opartym na II zasadzie dynamiki Newtona:

m · dv/dt = F_trac − F_aero − F_slope

gdzie:

* m – masa pojazdu [kg],
* v – prędkość pojazdu [m/s],
* F_trac – siła napędowa (lub hamująca) [N],
* F_aero – opór aerodynamiczny [N],
* F_slope – składowa siły grawitacji wynikająca z nachylenia drogi [N].

Równanie zdyskretyzowano metodą Eulera jawnego:

v(k+1) = v(k) + (Tp / m) · (F_trac − F_aero − F_slope)

gdzie Tp jest okresem próbkowania regulatora.

---

### 2.2. Siła napędowa

Sterowanie regulatorów realizowane jest poprzez uogólnioną siłę napędową:

F_pedal = (u / 100) · F_max

gdzie:

* u – sygnał sterujący regulatora [%],
* F_max – maksymalna siła napędowa pojazdu.

Ujemne wartości sygnału u odpowiadają sile hamującej (np. hamowanie silnikiem).

---

### 2.3. Ograniczenie mocy

Aby zachować realizm fizyczny przy dużych prędkościach, wprowadzono ograniczenie wynikające z maksymalnej mocy silnika:

F_limit = P_max / max(v, v_eps)

Rzeczywista siła napędowa pojazdu jest ograniczona zależnością:

F_trac = min(F_pedal, F_limit) dla u > 0
F_trac = max(F_pedal, −F_limit) dla u < 0

Takie uproszczenie pozwala odwzorować fakt, że przy dużych prędkościach pojazd jest ograniczony mocą, a nie maksymalną siłą.

---

### 2.4. Opór aerodynamiczny

Opór aerodynamiczny opisano klasycznym wzorem:

F_aero = 0.5 · ρ · Cd · A · v²

gdzie:

* ρ – gęstość powietrza (przyjęto 1.2 kg/m³),
* Cd – współczynnik oporu aerodynamicznego,
* A – powierzchnia czołowa pojazdu.

---

### 2.5. Nachylenie drogi

Wpływ nachylenia drogi uwzględniono poprzez składową siły grawitacji:

F_slope = m · g · sin(α)

gdzie:

* g – przyspieszenie ziemskie (9.81 m/s²),
* α – kąt nachylenia drogi.

Dodatnie wartości α odpowiadają jeździe pod górę, ujemne – zjazdowi.

---

## 3. Regulator PID

Zastosowano dyskretny regulator PID z:

* filtrowanym członem różniczkującym,
* zabezpieczeniem typu anti-windup.

Postać sterowania:

u = Kp · (β · e + I + D)

gdzie:

* e = v_zad − v – błąd regulacji,
* I – człon całkujący,
* D – człon różniczkujący.

Człon różniczkujący poddano filtracji dolnoprzepustowej:

D(k) = α · D_raw(k) + (1 − α) · D(k−1)

co ogranicza wpływ szumów i gwałtownych zmian sygnału.

Dodatkowo zastosowano człon feedforward kompensujący nachylenie drogi.

---

## 4. Regulator rozmyty Mamdaniego

Regulator rozmyty wykorzystuje:

* dwa wejścia: błąd prędkości e oraz zmianę błędu ce,
* trójkątne funkcje przynależności,
* wnioskowanie typu Mamdaniego (reguły IF–THEN),
* wyostrzanie metodą środka ciężkości.

Dodatkowo zastosowano pseudo-całkowanie błędu (leaky integrator), pełniące rolę odpowiednika członu I.

Regulator rozmyty generuje sygnał sterujący u w tym samym zakresie co PID, co umożliwia bezpośrednie porównanie obu metod.

---

## 5. Parametry pojazdów

### Pojazd sportowy

* masa: 1500 kg
* Cd: 0.30
* A: 2.2 m²
* F_max: 16000 N
* P_max: 350 kW
* v_max: 90 m/s

### Pojazd osobowy

* masa: 1600 kg
* Cd: 0.32
* A: 2.4 m²
* F_max: 7000 N
* P_max: 130 kW
* v_max: 60 m/s

### Pojazd ciężarowy

* masa: 40000 kg
* Cd: 0.70
* A: 10 m²
* F_max: 27000 N
* P_max: 400 kW
* v_max: 30 m/s

---

## 6. Przyjęte uproszczenia

W modelu przyjęto następujące uproszczenia:

* brak oporu toczenia,
* brak strat w układzie napędowym,
* jednowymiarowy model ruchu (brak dynamiki poprzecznej),
* brak jawnego modelu skrzyni biegów.

Uproszczenia te pozwalają skupić się na analizie działania algorytmów sterowania, a nie na pełnym modelowaniu pojazdu.

---

## 7. Wstępne wnioski

1. Zarówno regulator PID, jak i regulator rozmyty są w stanie stabilizować prędkość pojazdu w zadanych warunkach.
2. Wizualizacja siły napędowej F_trac pozwala lepiej interpretować działanie regulatorów niż prezentacja sygnału sterującego w procentach.
3. Ograniczenie mocy ma istotny wpływ na zachowanie pojazdu przy dużych prędkościach i zapobiega niefizycznym wartościom siły.
4. Regulator rozmyty wykazuje większą gładkość sterowania kosztem wolniejszej reakcji, natomiast PID reaguje szybciej, ale bardziej agresywnie.

Szczegółowe wnioski porównawcze zostaną przedstawione na podstawie wyników symulacji dla różnych scenariuszy jazdy.
