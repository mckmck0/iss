import numpy as np

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simpful import FuzzySet, FuzzySystem, LinguisticVariable, Triangular_MF


# ======================================================
# GLOBAL CONFIG
# ======================================================

APP_TITLE = "ISS – Tempomat"
G = 9.81
U_MIN, U_MAX = -100.0, 100.0

# Cache for fuzzy simulation results
fuzzy_cache = {
    "key": None,  # (vehicle, buckets, sp, T, Tp)
    "t": None,
    "vf": None,
    "uf": None,
}


VEHICLES = {
    "sportowe": {
        "label": "Sportowe",
        "m": 1200.0,
        "Cd": 0.30,
        "A": 2.2,
        "Fmax": 5000.0,
        "v_max": 100.0,
        "pid": dict(Kp=0.32, Ti=7.0, Td=0.9, Tp=0.1),
        "fuzzy_span": 60.0,
    },
    "osobowe": {
        "label": "Osobowe",
        "m": 1500.0,
        "Cd": 0.32,
        "A": 2.4,
        "Fmax": 4000.0,
        "v_max": 70.0,
        "pid": dict(Kp=0.24, Ti=10.0, Td=1.0, Tp=0.2),
        "fuzzy_span": 45.0,
    },
    "ciezarowe": {
        "label": "Ciężarowe",
        "m": 20000.0,
        "Cd": 0.7,
        "A": 10.0,
        "Fmax": 15000.0,
        "v_max": 30.0,
        "pid": dict(Kp=0.55, Ti=18.0, Td=0.6, Tp=0.5),
        "fuzzy_span": 25.0,
    },
}


# ======================================================
# MODEL POJAZDU
# ======================================================

def clamp(u):
    return max(U_MIN, min(U_MAX, u))


def step_velocity(v, u, dt, p):
    rho = 1.2
    F_trac = (u / 100.0) * p["Fmax"]
    F_aero = 0.5 * rho * p["Cd"] * p["A"] * v**2
    a = (F_trac - F_aero) / p["m"]
    return max(0.0, min(p["v_max"], v + a * dt))


# ======================================================
# PID
# ======================================================

def pid_step(e, e_prev, I, cfg):
    Kp, Ti, Td, Tp = cfg["Kp"], cfg["Ti"], cfg["Td"], cfg["Tp"]

    I_new = I + (Tp / Ti) * e if Ti > 0 else I
    D = (Td / Tp) * (e - e_prev)
    u = Kp * (e + I_new + D)
    u_sat = clamp(u)

    if u != u_sat:
        I_new = I

    return u_sat, I_new


# ======================================================
# FUZZY MAMDANI (SIMPFUL - 2 INPUTS)
# ======================================================

def fuzzy_controller(buckets, span):

    TERMS = {
        3: ["N", "Z", "P"],
        5: ["NB", "NS", "Z", "PS", "PB"],
        7: ["NB", "NM", "NS", "Z", "PS", "PM", "PB"],
    }

    terms = TERMS[buckets]
    n = len(terms)
    mid = n // 2

    # ============================
    # Agresywność zależna od liczby zbiorów
    # ============================
    SPAN_SCALE = {3: 0.45, 5: 0.75, 7: 1.00}[buckets]
    OUT_GAIN   = {3: 2.30, 5: 1.40, 7: 1.00}[buckets]
    CE_DAMP    = {3: 0.20, 5: 0.50, 7: 0.75}[buckets]

    span_e = span * SPAN_SCALE
    ce_span = span_e * 0.8

    fs = FuzzySystem()

    # ============================
    # Zbiory trójkątne
    # ============================
    def tri_sets(s):
        centers = np.linspace(-s, s, n)
        out = []
        for i, term in enumerate(terms):
            if i == 0:
                a, b, c = -s, -s, centers[i + 1]
            elif i == n - 1:
                a, b, c = centers[i - 1], s, s
            else:
                a, b, c = centers[i - 1], centers[i], centers[i + 1]
            out.append(FuzzySet(function=Triangular_MF(a, b, c), term=term)
)
        return out

    # ============================
    # Wejścia
    # ============================
    fs.add_linguistic_variable(
        "e",
        LinguisticVariable(tri_sets(span_e), universe_of_discourse=[-span_e, span_e]),
    )

    fs.add_linguistic_variable(
        "ce",
        LinguisticVariable(tri_sets(ce_span), universe_of_discourse=[-ce_span, ce_span]),
    )

    fs.add_linguistic_variable(
        "u",
        LinguisticVariable(tri_sets(100.0), universe_of_discourse=[-100.0, 100.0]),
    )

    # ============================
    # Reguły
    # ============================
    rules = []
    for i, te in enumerate(terms):
        for j, tce in enumerate(terms):

            base_idx = i

            if buckets == 3 and j == mid:
                adjustment = 1 if i >= mid else 0
            elif j == mid:
                adjustment = 1 if i > mid else 0
            else:
                adjustment = (j - mid) // 2

            idx = max(0, min(n - 1, base_idx + adjustment))

            rules.append(
                f"IF (e IS {te}) AND (ce IS {tce}) THEN (u IS {terms[idx]})"
            )

    fs.add_rules(rules)

    # ============================
    # Pseudo-integrator (fuzzy-PI)
    # ============================
    ie = 0.0
    IE_MAX = span * 8.0
    KI = 0.35
    LEAK = 0.002

    # ============================
    # Funkcja sterująca
    # ============================
    def compute(e, ce, _se):
        nonlocal ie

        e = max(-span_e, min(span_e, e))
        ce = max(-ce_span, min(ce_span, ce))

        fs.set_variable("e", e)
        fs.set_variable("ce", ce)

        u0 = fs.inference()["u"]

        # całkowanie błędu (leaky integrator)
        ie = (1.0 - LEAK) * ie + e
        ie = max(-IE_MAX, min(IE_MAX, ie))

        u = (
            OUT_GAIN * u0
            + KI * (ie / IE_MAX) * 100.0
            - CE_DAMP * (ce / ce_span) * 40.0
        )

        return clamp(u)

    return compute



# ======================================================
# DASH APP
# ======================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = APP_TITLE

app.layout = dbc.Container(fluid=True, children=[

    dcc.Store(id="vehicle", data="osobowe"),
    dcc.Store(id="fuzzy-buckets", data=5),

    dbc.Row([
        dbc.Col(
            html.H3(APP_TITLE, className="text-white text-center"),
            style={"background": "#2563eb", "padding": "10px"}
        )
    ]),

    dbc.Row([

        dbc.Col(width=3, children=[
            dbc.Card(style={"padding": "12px", "marginTop": "20px"}, children=[

                html.B("Wybór pojazdu"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Sportowe", id="veh-sportowe", outline=True, color="primary"),
                        dbc.Button("Osobowe", id="veh-osobowe", outline=True, color="primary"),
                        dbc.Button("Ciężarowe", id="veh-ciezarowe", outline=True, color="primary"),
                    ],
                    className="mb-2 w-100"
                ),

                html.Label("Prędkość docelowa [m/s]"),
                dcc.Slider(id="sp", min=5, max=100, step=1, value=30,
                           marks=None,
                           tooltip={"always_visible": True, "placement": "top"}),

                html.Label("Czas symulacji [s]"),
                dcc.Slider(id="T", min=60, max=600, step=30, value=300,
                           marks=None,
                           tooltip={"always_visible": True, "placement": "top"}),

                html.Hr(),

                html.B("Regulator PID"),

                html.Label("Kp – wzmocnienie regulatora [-]"),
                dcc.Slider(id="kp", min=0.01, max=0.5, step=0.01,
                           marks=None,
                           tooltip={"always_visible": True, "placement": "top"}),

                html.Label("Ti – czas zdwojenia [s]"),
                dcc.Slider(id="ti", min=1, max=60, step=1,
                           marks=None,
                           tooltip={"always_visible": True, "placement": "top"}),

                html.Label("Td – czas wyprzedzenia [s]"),
                dcc.Slider(id="td", min=0.0, max=10.0, step=0.1,
                           marks=None,
                           tooltip={"always_visible": True, "placement": "top"}),

                html.Label("Tp – okres próbkowania [s]"),
                dcc.Slider(id="tp", min=0.1, max=1.0, step=0.1,
                           marks=None,
                           tooltip={"always_visible": True, "placement": "top"}),

                html.Hr(),

                html.B("Liczba zbiorów rozmytych"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("3", id="fz-3", outline=True, color="primary"),
                        dbc.Button("5", id="fz-5", outline=True, color="primary"),
                        dbc.Button("7", id="fz-7", outline=True, color="primary"),
                    ],
                    className="w-100"
                ),

            ])
        ]),

        dbc.Col(width=9, children=[
            dbc.Button("Uruchom symulację", id="run",
                       color="primary", className="w-100 mb-2", style={"marginTop": "20px"}),
            dcc.Graph(id="graph", style={"height": "600px"})
        ])
    ])
])


# ======================================================
# CALLBACKS
# ======================================================

@app.callback(
    Output("vehicle", "data"),
    Output("veh-sportowe", "active"),
    Output("veh-osobowe", "active"),
    Output("veh-ciezarowe", "active"),
    Input("veh-sportowe", "n_clicks"),
    Input("veh-osobowe", "n_clicks"),
    Input("veh-ciezarowe", "n_clicks"),
    prevent_initial_call=True,
)
def select_vehicle(a, b, c):
    ctx = dash.callback_context
    selected = ctx.triggered[0]["prop_id"].split("-")[1].split(".")[0]
    return selected, selected == "sportowe", selected == "osobowe", selected == "ciezarowe"


@app.callback(
    Output("fuzzy-buckets", "data"),
    Output("fz-3", "active"),
    Output("fz-5", "active"),
    Output("fz-7", "active"),
    Input("fz-3", "n_clicks"),
    Input("fz-5", "n_clicks"),
    Input("fz-7", "n_clicks"),
    prevent_initial_call=True,
)
def select_fuzzy(a, b, c):
    ctx = dash.callback_context
    selected = int(ctx.triggered[0]["prop_id"].split("-")[1].split(".")[0])
    return selected, selected == 3, selected == 5, selected == 7


@app.callback(
    Output("kp", "value"),
    Output("ti", "value"),
    Output("td", "value"),
    Output("tp", "value"),
    Output("sp", "max"),
    Input("vehicle", "data"),
)
def update_pid(v):
    p = VEHICLES[v]
    pid = p["pid"]
    return pid["Kp"], pid["Ti"], pid["Td"], pid["Tp"], p["v_max"]


@app.callback(
    Output("graph", "figure"),
    Input("run", "n_clicks"),
    State("vehicle", "data"),
    State("fuzzy-buckets", "data"),
    State("sp", "value"),
    State("T", "value"),
    State("kp", "value"),
    State("ti", "value"),
    State("td", "value"),
    State("tp", "value"),
)
def simulate(_, vehicle, buckets, sp, T, kp, ti, td, tp):
    global fuzzy_cache

    p = VEHICLES[vehicle]
    pid_defaults = p["pid"]
    cfg = dict(
        Kp=kp if kp is not None else pid_defaults["Kp"],
        Ti=ti if ti is not None else pid_defaults["Ti"],
        Td=td if td is not None else pid_defaults["Td"],
        Tp=tp if tp is not None else pid_defaults["Tp"]
    )

    t = np.arange(0, T, cfg["Tp"])

    # Check if we need to recalculate fuzzy
    fuzzy_key = (vehicle, buckets, sp, T, cfg["Tp"])

    if fuzzy_cache["key"] == fuzzy_key:
        # Reuse cached fuzzy results
        vf = fuzzy_cache["vf"]
        uf = fuzzy_cache["uf"]
    else:
        # Calculate fuzzy simulation
        fz = fuzzy_controller(buckets, p["fuzzy_span"])
        v_fz = 0.0
        ef_prev = 0.0
        vf, uf = [], []

        for _ in t:
            ef = sp - v_fz
            ufz = fz(ef, (ef - ef_prev) / cfg["Tp"], 0)
            v_fz = step_velocity(v_fz, ufz, cfg["Tp"], p)
            vf.append(v_fz)
            uf.append(ufz)
            ef_prev = ef

        # Cache the results
        fuzzy_cache["key"] = fuzzy_key
        fuzzy_cache["t"] = t
        fuzzy_cache["vf"] = vf
        fuzzy_cache["uf"] = uf

    # Always calculate PID simulation (fast)
    v_pid = 0.0
    I = 0.0
    e_prev = 0.0
    vp, up = [], []

    for _ in t:
        e = sp - v_pid
        u, I = pid_step(e, e_prev, I, cfg)
        v_pid = step_velocity(v_pid, u, cfg["Tp"], p)
        vp.append(v_pid)
        up.append(u)
        e_prev = e

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=t, y=vp, name="PID"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=vf, name="Fuzzy"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=[sp]*len(t), name="Zadana", line=dict(dash="dash")), 1, 1)

    fig.add_trace(go.Scatter(x=t, y=up, name="Gaz PID"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=uf, name="Gaz Fuzzy"), 2, 1)

    fig.update_yaxes(title="Prędkość [m/s]", row=1, col=1, range=[0, p["v_max"]+30])
    fig.update_yaxes(title="Gaz [%]", row=2, col=1, range=[U_MIN, U_MAX])
    fig.update_xaxes(title="Czas [s]", row=2, col=1)

    fig.update_layout(height=600, autosize=False)

    return fig


if __name__ == "__main__":
    app.run(debug=True)
