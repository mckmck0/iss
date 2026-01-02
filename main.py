
import math
from functools import lru_cache

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simpful import FuzzySet, FuzzySystem, LinguisticVariable, Triangular_MF

# Optional MF shapes (availability depends on Simpful version)
try:
    from simpful import Trapezoidal_MF
except Exception:
    Trapezoidal_MF = None

try:
    from simpful import Gaussian_MF
except Exception:
    Gaussian_MF = None

try:
    from simpful import Sigmoid_MF
except Exception:
    Sigmoid_MF = None

try:
    from simpful import Bell_MF
except Exception:
    Bell_MF = None


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Cruise Control"

g = 9.81  # [m/s^2]


# ----------------- Model pojazdu (szybka symulacja) -----------------
def vehicle_acceleration(v, u_percent, load, air_density, slope_deg, friction):
    """
    Model pojazdu: zwraca przyspieszenie a [m/s^2].
    Szybka integracja: Euler (zamiast odeint).
    """
    m_vehicle = 1400.0
    m = m_vehicle + load

    Cd = 0.24
    rho = air_density
    A = 5.0
    F_c_max = 10000.0

    theta = math.radians(slope_deg)

    F_c = (u_percent / 100.0) * F_c_max
    F_op = 0.5 * rho * Cd * A * (v ** 2)
    F_roll = friction * m * g * math.cos(theta)
    F_grade = m * g * math.sin(theta)

    F_w = F_c - F_op - (F_roll + F_grade)
    return F_w / m


def step_velocity(v, u, dt, load, air_density, slope, friction):
    a = vehicle_acceleration(v, u, load, air_density, slope, friction)
    v_next = v + a * dt
    return max(0.0, float(v_next))


def _clamp_throttle(value: float) -> float:
    return float(max(-50.0, min(100.0, value)))


# ----------------- Linguistic terms -----------------
def _terms_input(n: int):
    if n == 3:
        return ["U", "Z", "D"]
    if n == 5:
        return ["DU", "MU", "Z", "MD", "DD"]
    if n == 7:
        return ["DU", "ŚU", "MU", "Z", "MD", "ŚD", "DD"]
    raise ValueError("Dozwolone wartości: 3, 5, 7")


# ----------------- Membership functions -----------------
def _mf_available(mf_type: str) -> bool:
    mf_type = (mf_type or "").lower()
    if mf_type == "triangular":
        return True
    if mf_type == "trapezoid":
        return Trapezoidal_MF is not None
    if mf_type == "gaussian":
        return Gaussian_MF is not None
    if mf_type == "sigmoid":
        return Sigmoid_MF is not None
    if mf_type == "bell":
        return Bell_MF is not None
    return False


def _build_mf_sets(span: float, terms, mf_type: str):
    """
    Buduje zbiory rozmyte równomiernie na [-span, span].
    Jeśli MF nie jest dostępna w Twojej wersji simpful -> fallback do trójkątnej.
    """
    mf_req = (mf_type or "triangular").lower()
    mf_used = mf_req if _mf_available(mf_req) else "triangular"

    n = len(terms)
    centers = np.linspace(-span, span, n).tolist()
    step = centers[1] - centers[0] if n > 1 else span

    sets = []
    for i, term in enumerate(terms):
        c = centers[i]

        if mf_used == "triangular":
            if i == 0:
                a, b, d = -span, -span, centers[i + 1]
            elif i == n - 1:
                a, b, d = centers[i - 1], span, span
            else:
                a, b, d = centers[i - 1], c, centers[i + 1]
            mf = Triangular_MF(a, b, d)

        elif mf_used == "trapezoid":
            # a----b====c----d
            w = abs(step)
            a = c - 1.0 * w
            b = c - 0.4 * w
            cc = c + 0.4 * w
            d = c + 1.0 * w
            a, b, cc, d = max(-span, a), max(-span, b), min(span, cc), min(span, d)
            mf = Trapezoidal_MF(a, b, cc, d)

        elif mf_used == "gaussian":
            sigma = max(1e-6, abs(step) / 2.0)
            mf = Gaussian_MF(c, sigma)

        elif mf_used == "sigmoid":
            a = 5.0 / max(1e-6, abs(step))
            if i <= (n // 2):
                mf = Sigmoid_MF(-a, c)
            else:
                mf = Sigmoid_MF(a, c)

        elif mf_used == "bell":
            a = max(1e-6, abs(step))
            b = 2.0
            mf = Bell_MF(c, a, b)

        sets.append(FuzzySet(function=mf, term=term))

    return sets, mf_used


# ----------------- AND operator (T-norma) -----------------
def _safe_make_fuzzysystem(and_tnorm: str):
    if (and_tnorm or "min").lower() != "product":
        return FuzzySystem()
    try:
        return FuzzySystem(operators=["AND_PRODUCT"])
    except Exception:
        return FuzzySystem()


# ----------------- Rule base -----------------
def _build_rules_pid_3d(terms):
    """
    Prosty fuzzy-PID:
      IF e is ... AND ce is ... AND se is ... THEN u is ...
    Mapowanie poziomów: u_level ≈ e_level + ce_level + se_level (symetrycznie).
    """
    n = len(terms)
    mid = n // 2
    rules = []
    for i_e, te in enumerate(terms):
        e_level = i_e - mid
        for i_ce, tce in enumerate(terms):
            ce_level = i_ce - mid
            for i_se, tse in enumerate(terms):
                se_level = i_se - mid
                out_level = e_level + ce_level + se_level
                out_idx = max(0, min(n - 1, out_level + mid))
                rules.append(
                    f"IF (e IS {te}) AND (ce IS {tce}) AND (se IS {tse}) THEN (u IS {terms[out_idx]})"
                )
    return rules


# ----------------- Cached controllers -----------------
@lru_cache(maxsize=96)
def _cached_mamdani_controller(buckets: int, mf_type: str, and_tnorm: str, wyostrzanie: str, setpoint_int: int):
    """
    Mamdani + WYOSTRZANIE (defuzyfikacja).
    Mapa metod (nazwy jak w simpful):
      - centroid  -> środek ciężkości
      - mom       -> środek maksimum
      - som       -> pierwsze maksimum
      - lom       -> ostatnie maksimum
    """
    terms = _terms_input(buckets)

    sp = float(max(0.0, setpoint_int))
    e_span = max(20.0, 2.0 * sp)
    u_span = 100.0

    fs = _safe_make_fuzzysystem(and_tnorm)

    e_sets, mf_used = _build_mf_sets(e_span, terms, mf_type)
    ce_sets, _ = _build_mf_sets(e_span, terms, mf_type)
    se_sets, _ = _build_mf_sets(e_span, terms, mf_type)
    u_sets, _ = _build_mf_sets(u_span, terms, mf_type)

    fs.add_linguistic_variable("e", LinguisticVariable(e_sets, universe_of_discourse=[-e_span, e_span]))
    fs.add_linguistic_variable("ce", LinguisticVariable(ce_sets, universe_of_discourse=[-e_span, e_span]))
    fs.add_linguistic_variable("se", LinguisticVariable(se_sets, universe_of_discourse=[-e_span, e_span]))
    fs.add_linguistic_variable("u", LinguisticVariable(u_sets, universe_of_discourse=[-u_span, u_span]))

    fs.add_rules(_build_rules_pid_3d(terms))

    method = (wyostrzanie or "centroid").lower().strip()
    if method not in {"centroid", "mom", "som", "lom"}:
        method = "centroid"
    try:
        fs.set_defuzzification(method)
    except Exception:
        # jeśli dana wersja nie wspiera -> centroid
        try:
            fs.set_defuzzification("centroid")
        except Exception:
            pass

    def compute_u(e, ce, se):
        fs.set_variable("e", max(-e_span, min(e_span, float(e))))
        fs.set_variable("ce", max(-e_span, min(e_span, float(ce))))
        fs.set_variable("se", max(-e_span, min(e_span, float(se))))
        out = fs.inference()["u"]
        return _clamp_throttle(out)

    return compute_u, mf_used


@lru_cache(maxsize=96)
def _cached_tsk_controller(
    buckets: int,
    mf_type: str,
    and_tnorm: str,
    setpoint_int: int,
    order: str,
    a: float,
    b: float,
    c: float,
    d: float,
):
    terms = _terms_input(buckets)

    sp = float(max(0.0, setpoint_int))
    e_span = max(20.0, 2.0 * sp)

    fs = _safe_make_fuzzysystem(and_tnorm)

    e_sets, mf_used = _build_mf_sets(e_span, terms, mf_type)
    ce_sets, _ = _build_mf_sets(e_span, terms, mf_type)
    se_sets, _ = _build_mf_sets(e_span, terms, mf_type)

    fs.add_linguistic_variable("e", LinguisticVariable(e_sets, universe_of_discourse=[-e_span, e_span]))
    fs.add_linguistic_variable("ce", LinguisticVariable(ce_sets, universe_of_discourse=[-e_span, e_span]))
    fs.add_linguistic_variable("se", LinguisticVariable(se_sets, universe_of_discourse=[-e_span, e_span]))

    n = len(terms)
    centers = np.linspace(-100.0, 100.0, n).tolist()
    order = (order or "liniowy").lower()

    out_fn_for_term = {}
    for term, k in zip(terms, centers):
        fn = f"OUT_{term}"
        out_fn_for_term[term] = fn
        if order.startswith("sta"):
            fs.set_output_function(fn, f"{float(k)}")
        else:
            expr = f"{float(a)}*e + {float(b)}*ce + {float(c)}*se + {float(d)} + {float(k)}"
            fs.set_output_function(fn, expr)

    mid = n // 2
    rules = []
    for i_e, te in enumerate(terms):
        e_level = i_e - mid
        for i_ce, tce in enumerate(terms):
            ce_level = i_ce - mid
            for i_se, tse in enumerate(terms):
                se_level = i_se - mid
                out_level = e_level + ce_level + se_level
                out_idx = max(0, min(n - 1, out_level + mid))
                tout = terms[out_idx]
                rules.append(
                    f"IF (e IS {te}) AND (ce IS {tce}) AND (se IS {tse}) THEN (u IS {out_fn_for_term[tout]})"
                )
    fs.add_rules(rules)

    def compute_u(e, ce, se):
        fs.set_variable("e", max(-e_span, min(e_span, float(e))))
        fs.set_variable("ce", max(-e_span, min(e_span, float(ce))))
        fs.set_variable("se", max(-e_span, min(e_span, float(se))))
        out = fs.Sugeno_inference(["u"])["u"]
        return _clamp_throttle(out)

    return compute_u, mf_used


def _make_error_figure(message: str):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ----------------- UI -----------------
app.layout = html.Div(
    style={
        "backgroundColor": "#F3F4F6",
        "height": "100vh",
        "overflow": "hidden",
        "display": "flex",
        "flexDirection": "column",
    },
    children=[
        html.Div(
            html.H1("Cruise Control", style={"textAlign": "center", "color": "white", "margin": 0, "fontWeight": "bold"}),
            style={"backgroundColor": "#1D4ED8", "padding": "14px 0", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "flexShrink": 0},
        ),
        dbc.Container(
            fluid=True,
            style={"flex": 1, "overflow": "hidden", "paddingTop": "10px", "paddingBottom": "10px"},
            children=[
                dbc.Row(
                    style={"height": "100%"},
                    children=[
                        dbc.Col(
                            width=3,
                            style={"height": "100%"},
                            children=[
                                dbc.Card(
                                    style={
                                        "backgroundColor": "white",
                                        "border": "1px solid #BFDBFE",
                                        "boxShadow": "0 2px 6px rgba(148,163,184,0.35)",
                                        "borderRadius": "14px",
                                        "padding": "14px",
                                        "height": "100%",
                                        "overflowY": "auto",
                                    },
                                    children=[
                                        html.H5("Parametry symulacji", style={"color": "#1D4ED8", "marginBottom": "10px", "fontWeight": "bold"}),
                                        html.Label("Prędkość zadana [m/s]", style={"fontWeight": "500"}),
                                        dcc.Slider(id="setPoint", min=0, max=100, step=1, value=50, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Czas symulacji [s]", style={"fontWeight": "500"}),
                                        dcc.Slider(id="t", min=10, max=600, step=10, value=450, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Krok symulacji dt [s]", style={"fontWeight": "500"}),
                                        dcc.Slider(id="dt", min=0.2, max=2.0, step=0.1, value=1.0, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Obciążenie [kg]", style={"fontWeight": "500"}),
                                        dcc.Slider(id="load", min=100, max=2000, step=50, value=200, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Gęstość powietrza [kg/m³]", style={"fontWeight": "500"}),
                                        dcc.Slider(id="airDensity", min=1.0, max=2.0, step=0.01, value=1.2, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Nachylenie drogi [°]", style={"fontWeight": "500"}),
                                        dcc.Slider(id="slope", min=0, max=30, step=1, value=0, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Współczynnik tarcia [-]", style={"fontWeight": "500"}),
                                        dcc.Slider(id="friction", min=0.0, max=0.5, step=0.01, value=0.1, marks=None, tooltip={"always_visible": True, "placement": "top"}),

                                        html.Hr(),
                                        html.H5("Regulator PID", style={"color": "#1D4ED8", "marginBottom": "8px", "fontWeight": "bold"}),
                                        html.Label("Kp", style={"fontWeight": "500"}),
                                        dcc.Slider(id="kp", min=0, max=1, step=0.001, value=0.03, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Ki", style={"fontWeight": "500"}),
                                        dcc.Slider(id="ki", min=0, max=2, step=0.001, value=0.10, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                        html.Br(),
                                        html.Label("Kd", style={"fontWeight": "500"}),
                                        dcc.Slider(id="kd", min=0, max=2, step=0.001, value=0.40, marks=None, tooltip={"always_visible": True, "placement": "top"}),

                                        html.Hr(),
                                        html.H5("Regulator rozmyty", style={"color": "#1D4ED8", "marginBottom": "8px", "fontWeight": "bold"}),

                                        dbc.Label("Model"),
                                        dcc.Dropdown(
                                            id="fuzzy_model",
                                            options=[
                                                {"label": "Mamdani (PID rozmyty)", "value": "mamdani"},
                                                {"label": "TSK (Takagi–Sugeno–Kang)", "value": "tsk"},
                                            ],
                                            value="mamdani",
                                            clearable=False,
                                            style={"marginBottom": "10px"},
                                        ),

                                        dbc.Label("Liczba zbiorów lingwistycznych"),
                                        dcc.Dropdown(
                                            id="buckets",
                                            options=[{"label": "3", "value": 3}, {"label": "5", "value": 5}, {"label": "7", "value": 7}],
                                            value=5,
                                            clearable=False,
                                            style={"marginBottom": "10px"},
                                        ),

                                        dbc.Label("Typ funkcji przynależności"),
                                        dcc.Dropdown(
                                            id="mf_type",
                                            options=[
                                                {"label": "Trójkątna", "value": "triangular"},
                                                {"label": "Trapezowa", "value": "trapezoid"},
                                                {"label": "Gaussowska", "value": "gaussian"},
                                                {"label": "Sigmoidalna", "value": "sigmoid"},
                                                {"label": "Dzwonowa", "value": "bell"},
                                            ],
                                            value="triangular",
                                            clearable=False,
                                            style={"marginBottom": "10px"},
                                        ),

                                        dbc.Label("T-norma (AND)"),
                                        dcc.Dropdown(
                                            id="and_tnorm",
                                            options=[{"label": "MIN", "value": "min"}, {"label": "PROD", "value": "product"}],
                                            value="min",
                                            clearable=False,
                                            style={"marginBottom": "10px"},
                                        ),

                                        # Mamdani-only: WYOSTRZANIE
                                        dbc.Collapse(
                                            id="panel_mamdani",
                                            is_open=True,
                                            children=[
                                                dbc.Label("Wyostrzanie (metoda wyznaczenia wartości liczbowej)"),
                                                dcc.Dropdown(
                                                    id="wyostrzanie",
                                                    options=[
                                                        {"label": "Środek ciężkości", "value": "centroid"},
                                                        {"label": "Środek maksimum", "value": "mom"},
                                                        {"label": "Pierwsze maksimum", "value": "som"},
                                                        {"label": "Ostatnie maksimum", "value": "lom"},
                                                    ],
                                                    value="centroid",
                                                    clearable=False,
                                                    style={"marginBottom": "10px"},
                                                ),
                                            ],
                                        ),

                                        # TSK-only
                                        dbc.Collapse(
                                            id="panel_tsk",
                                            is_open=False,
                                            children=[
                                                dbc.Label("Postać konkluzji y = f(x)"),
                                                dcc.Dropdown(
                                                    id="tsk_order",
                                                    options=[{"label": "Stała (0 rząd)", "value": "staly"}, {"label": "Liniowa (1 rząd)", "value": "liniowy"}],
                                                    value="liniowy",
                                                    clearable=False,
                                                    style={"marginBottom": "10px"},
                                                ),
                                                dbc.Label("a (przy e)"),
                                                dcc.Slider(id="tsk_a", min=-2.0, max=2.0, step=0.05, value=0.20, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                                html.Br(),
                                                dbc.Label("b (przy ce)"),
                                                dcc.Slider(id="tsk_b", min=-2.0, max=2.0, step=0.05, value=0.10, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                                html.Br(),
                                                dbc.Label("c (przy se)"),
                                                dcc.Slider(id="tsk_c", min=-2.0, max=2.0, step=0.05, value=0.02, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                                html.Br(),
                                                dbc.Label("d (wyraz wolny)"),
                                                dcc.Slider(id="tsk_d", min=-20.0, max=20.0, step=0.5, value=0.0, marks=None, tooltip={"always_visible": True, "placement": "top"}),
                                                html.Br(),
                                            ],
                                        ),

                                        dbc.Alert(id="warning", color="warning", is_open=False, style={"marginTop": "10px"}),

                                        dbc.Button("Uruchom", id="run", n_clicks=0, color="primary", style={"width": "100%", "borderRadius": "10px", "marginTop": "6px"}),
                                    ],
                                )
                            ],
                        ),
                        dbc.Col(
                            width=9,
                            style={"height": "100%"},
                            children=[
                                dbc.Card(
                                    style={
                                        "backgroundColor": "white",
                                        "border": "1px solid #BFDBFE",
                                        "boxShadow": "0 2px 6px rgba(148,163,184,0.35)",
                                        "borderRadius": "14px",
                                        "padding": "8px 10px",
                                        "height": "100%",
                                    },
                                    children=[dcc.Loading(type="circle", children=dcc.Graph(id="graph", style={"height": "100%", "width": "100%"}))],
                                )
                            ],
                        ),
                    ],
                )
            ],
        ),
    ],
)


@app.callback(
    Output("panel_mamdani", "is_open"),
    Output("panel_tsk", "is_open"),
    Input("fuzzy_model", "value"),
)
def toggle_panels(model):
    model = (model or "mamdani").lower()
    if model == "tsk":
        return False, True
    return True, False


@app.callback(
    Output("graph", "figure"),
    Output("warning", "is_open"),
    Output("warning", "children"),
    Input("run", "n_clicks"),
    State("kp", "value"),
    State("ki", "value"),
    State("kd", "value"),
    State("t", "value"),
    State("dt", "value"),
    State("load", "value"),
    State("setPoint", "value"),
    State("airDensity", "value"),
    State("slope", "value"),
    State("friction", "value"),
    State("fuzzy_model", "value"),
    State("buckets", "value"),
    State("mf_type", "value"),
    State("and_tnorm", "value"),
    State("wyostrzanie", "value"),
    State("tsk_order", "value"),
    State("tsk_a", "value"),
    State("tsk_b", "value"),
    State("tsk_c", "value"),
    State("tsk_d", "value"),
)
def run_sim(
    n_clicks,
    kp, ki, kd,
    simulation_time,
    dt,
    load_,
    setPoint,
    airDensity,
    slope,
    friction,
    fuzzy_model,
    buckets,
    mf_type,
    and_tnorm,
    wyostrzanie,
    tsk_order,
    tsk_a,
    tsk_b,
    tsk_c,
    tsk_d,
):
    try:
        tf = float(simulation_time or 450)
        dt = float(dt or 1.0)
        dt = max(0.05, min(5.0, dt))
        nsteps = int(tf / dt) + 1
        ts = np.linspace(0, tf, nsteps)

        load = float(load_ or 200)
        sp = float(setPoint or 50)
        airDens = float(airDensity or 1.2)
        slope = float(slope or 0)
        friction = float(friction or 0.1)

        # ---------- PID ----------
        vs = np.zeros(nsteps)
        gas_pid = np.zeros(nsteps)
        v = 0.0
        integral = 0.0
        prev_error = sp - v

        kp = float(kp or 0.03)
        ki = float(ki or 0.10)
        kd = float(kd or 0.40)

        for i in range(1, nsteps):
            error = sp - v
            integral += error * dt
            derivative = (error - prev_error) / dt

            u = kp * error + ki * integral + kd * derivative

            # clamp + anti-windup
            if u > 100:
                u = 100
                integral -= error * dt
            if u < -50:
                u = -50
                integral -= error * dt

            gas_pid[i] = u
            prev_error = error
            v = step_velocity(v, u, dt, load, airDens, slope, friction)
            vs[i] = v

        # ---------- FUZZY ----------
        warn_open = False
        warn_msg = ""

        mf_req = (mf_type or "triangular").lower()
        if not _mf_available(mf_req):
            warn_open = True
            warn_msg = f"Wybrany typ MF („{mf_req}”) nie jest dostępny w tej wersji simpful. Używam trójkątnej."

        buckets = int(buckets or 5)
        fuzzy_model = (fuzzy_model or "mamdani").lower()
        sp_int = int(round(sp))

        if fuzzy_model == "tsk":
            compute_u, mf_used = _cached_tsk_controller(
                buckets=buckets,
                mf_type=mf_req,
                and_tnorm=and_tnorm,
                setpoint_int=sp_int,
                order=tsk_order or "liniowy",
                a=float(tsk_a or 0.20),
                b=float(tsk_b or 0.10),
                c=float(tsk_c or 0.02),
                d=float(tsk_d or 0.0),
            )
        else:
            compute_u, mf_used = _cached_mamdani_controller(
                buckets=buckets,
                mf_type=mf_req,
                and_tnorm=and_tnorm,
                wyostrzanie=(wyostrzanie or "centroid"),
                setpoint_int=sp_int,
            )

        if mf_used != mf_req and not warn_open:
            warn_open = True
            warn_msg = "Używam funkcji trójkątnych (fallback)."

        f_vs = np.zeros(nsteps)
        gas_fuzzy = np.zeros(nsteps)
        fv = 0.0
        prev_f_error = sp - fv
        sum_error = 0.0

        for i in range(1, nsteps):
            e = sp - fv
            sum_error += e * dt
            ce = (e - prev_f_error) / dt

            u = compute_u(e, ce, sum_error)
            gas_fuzzy[i] = u
            prev_f_error = e
            fv = step_velocity(fv, u, dt, load, airDens, slope, friction)
            f_vs[i] = fv

        # ---------- PLOT ----------
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("PID controller", "Fuzzy controller"))

        fig.add_trace(go.Scatter(x=ts, y=np.round(vs, 2), name="Prędkość [m/s]", line=dict(color="#1f77b4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=np.round(gas_pid, 2), name="Gaz [%]", line=dict(color="#d62728")), row=1, col=1)

        fig.add_trace(go.Scatter(x=ts, y=np.round(f_vs, 2), name="Prędkość [m/s]", line=dict(color="#1f77b4", dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts, y=np.round(gas_fuzzy, 2), name="Gaz [%]", line=dict(color="#d62728", dash="dot")), row=2, col=1)

        ymax = max(110.0, sp * 1.4, float(np.max(vs)) * 1.2, float(np.max(f_vs)) * 1.2)
        ymin = -55.0
        fig.update_yaxes(range=[ymin, ymax], title_text="v [m/s] oraz gaz [%]", row=1, col=1)
        fig.update_yaxes(range=[ymin, ymax], title_text="v [m/s] oraz gaz [%]", row=2, col=1)
        fig.update_xaxes(title_text="t [s]", row=2, col=1)

        fig.update_layout(
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(x=0.72, y=1.15),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#1F2937"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)

        return fig, warn_open, warn_msg

    except Exception as e:
        return _make_error_figure(f"Błąd: {e}"), True, f"Błąd: {e}"


if __name__ == "__main__":
    app.run(debug=True)
