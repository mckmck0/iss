import math
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.dash import no_update
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Cruise Control"

g = 9.81  # [m/s^2]


def vehicle_model(v, t, u_percent, load, air_density, slope_deg, friction):
    m_vehicle = 1400.0
    m = m_vehicle + load
    Cd = 0.24
    rho = air_density
    A = 5.0
    F_c_max = 10000.0

    theta = math.radians(slope_deg)

    F_c = (u_percent / 100.0) * F_c_max
    F_op = 0.5 * rho * Cd * A * v**2
    F_roll = friction * m * g * math.cos(theta)
    F_grade = m * g * math.sin(theta)
    F_t = F_roll + F_grade

    F_w = F_c - F_op - F_t
    a = F_w / m
    return a


app.layout = html.Div(
    # CAŁA STRONA: pełna wysokość okna, bez scrolli
    style={
        "backgroundColor": "#F3F4F6",
        "height": "100vh",
        "overflow": "hidden",
        "display": "flex",
        "flexDirection": "column",
    },
    children=[
        # HEADER – stała wysokość
        html.Div(
            html.H1(
                "Cruise Control",
                style={
                    "textAlign": "center",
                    "color": "white",
                    "margin": 0,
                    "fontWeight": "bold",
                },
            ),
            style={
                "backgroundColor": "#1D4ED8",
                "padding": "16px 0",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                "flexShrink": 0,
            },
        ),

        # GŁÓWNY OBSZAR – wypełnia resztę wysokości
        dbc.Container(
            fluid=True,
            style={
                "flex": 1,
                "overflow": "hidden",
                "paddingTop": "10px",
                "paddingBottom": "10px",
            },
            children=[
                dbc.Row(
                    style={"height": "100%"},
                    children=[
                        # LEWY PANEL
                        dbc.Col(
                            width=3,
                            style={"height": "100%"},
                            children=[
                                dbc.Card(
                                    style={
                                        "backgroundColor": "white",
                                        "border": "1px solid #BFDBFE",
                                        "boxShadow": "0 2px 6px rgba(148,163,184,0.4)",
                                        "borderRadius": "12px",
                                        "padding": "15px",
                                        "height": "100%",
                                        "overflowY": "auto",  # jak coś się nie zmieści, scroll tylko w panelu
                                    },
                                    children=[
                                        html.H5(
                                            "Parameters",
                                            style={
                                                "color": "#1D4ED8",
                                                "marginBottom": "10px",
                                                "fontWeight": "bold",
                                            },
                                        ),

                                        html.Label(
                                            "Desired velocity [0 - 100 m/s]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="setPoint",
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=30,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Simulation Time [10 - 600 s]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="t",
                                            min=10,
                                            max=600,
                                            step=10,
                                            value=500,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Proportional Kp [0 - 1]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="kp",
                                            min=0,
                                            max=1,
                                            step=0.001,
                                            value=0.02,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Integral Ki [0 - 2]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="Ti",
                                            min=0,
                                            max=2,
                                            step=0.001,
                                            value=0.1,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Derivative Kd [0 - 2]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="Td",
                                            min=0,
                                            max=2,
                                            step=0.001,
                                            value=0.5,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Load [100 - 2000] kg:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="load",
                                            min=100,
                                            max=2000,
                                            step=50,
                                            value=200,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Air density [1 - 2 kg/m³]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="airDensity",
                                            min=1.0,
                                            max=2.0,
                                            step=0.01,
                                            value=1.2,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Slope degree [0 - 30]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="slope",
                                            min=0,
                                            max=30,
                                            step=1,
                                            value=0,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        html.Label(
                                            "Friction [0 - 0.5]:",
                                            style={"fontWeight": "500"},
                                        ),
                                        dcc.Slider(
                                            id="friction",
                                            min=0.0,
                                            max=0.5,
                                            step=0.01,
                                            value=0.1,
                                            marks=None,
                                            tooltip={
                                                "always_visible": True,
                                                "placement": "top",
                                            },
                                        ),
                                        html.Br(),

                                        dbc.Button(
                                            "Submit",
                                            id="submit-button-state",
                                            n_clicks=0,
                                            color="primary",
                                            size="sm",
                                            style={"width": "100%"},
                                        ),
                                        html.Br(),
                                        dbc.Alert(
                                            "WRONG NUMBERS, YOU IDIOT!",
                                            color="danger",
                                            id="alert-auto",
                                            is_open=False,
                                            duration=5000,
                                            style={"marginTop": "8px"},
                                        ),
                                    ],
                                )
                            ],
                        ),

                        # PRAWY PANEL – WYKRESY
                        dbc.Col(
                            width=9,
                            style={"height": "100%"},
                            children=[
                                dbc.Card(
                                    style={
                                        "backgroundColor": "white",
                                        "border": "1px solid #BFDBFE",
                                        "boxShadow": "0 2px 6px rgba(148,163,184,0.4)",
                                        "borderRadius": "12px",
                                        "padding": "8px 12px",
                                        "height": "100%",
                                    },
                                    children=[
                                        dcc.Loading(
                                            id="loading_icon",
                                            type="circle",
                                            children=html.Div(
                                                style={"height": "100%"},
                                                children=dcc.Graph(
                                                    id="the_graph",
                                                    style={
                                                        "height": "100%",
                                                        "width": "100%",
                                                    },
                                                    config={"displayModeBar": True},
                                                ),
                                            ),
                                        )
                                    ],
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
    Output("the_graph", "figure"),
    Output("alert-auto", "is_open"),
    Input("submit-button-state", "n_clicks"),
    State("alert-auto", "is_open"),
    State("kp", "value"),
    State("Ti", "value"),
    State("Td", "value"),
    State("t", "value"),
    State("load", "value"),
    State("setPoint", "value"),
    State("airDensity", "value"),
    State("slope", "value"),
    State("friction", "value"),
)
def update_output(
    clicks,
    is_open,
    kp,
    Ki,
    Kd,
    simulation_time,
    load_,
    setPoint,
    airDensity,
    slope,
    friction,
):
    if clicks == 0:
        return no_update, is_open

    if (
        load_ is None
        or setPoint is None
        or airDensity is None
        or kp is None
        or Ki is None
        or Kd is None
        or simulation_time is None
        or slope is None
        or friction is None
    ):
        return no_update, True

    tf = float(simulation_time)
    dt = 1.0
    nsteps = int(tf / dt) + 1
    ts = np.linspace(0, tf, nsteps)

    load = float(load_)
    sp = float(setPoint)
    airDens = float(airDensity)

    vs = np.zeros(nsteps)
    gas_store = np.zeros(nsteps)

    v = 0.0
    vs[0] = v
    gas_store[0] = 0.0

    integral = 0.0
    prev_error = sp - v

    for i in range(1, nsteps):
        error = sp - v
        integral += error * dt
        derivative = (error - prev_error) / dt

        u = kp * error + Ki * integral + Kd * derivative  # [%]

        if u > 100:
            u = 100
            integral -= error * dt
        if u < -50:
            u = -50
            integral -= error * dt

        gas_store[i] = u
        prev_error = error

        v_next = odeint(
            vehicle_model,
            v,
            [0, dt],
            args=(u, load, airDens, slope, friction),
        )[-1, 0]

        v = max(0.0, float(v_next))
        vs[i] = v

    vs = np.round(vs, 2)
    gas_store = np.round(gas_store, 2)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Model 1: v(t) & gas pedal(t)", "Model 2: placeholder"),
    )

    fig.add_trace(
        go.Scatter(x=ts, y=vs, name="Velocity [m/s]"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=gas_store, name="Usage of gas pedal [%]"),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="v [m/s] / gas [%]", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="(second model)", row=2, col=1, range=[0, 1])

    fig.update_xaxes(title_text="t [s]", row=1, col=1)
    fig.update_xaxes(title_text="t [s]", row=2, col=1)

    fig.update_layout(
        # bez sztywnej wysokości – Graph wypełni kartę, karta wypełni kolumnę
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.75, y=1.15),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1F2937"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)

    return fig, False


if __name__ == "__main__":
    app.run(debug=True)
