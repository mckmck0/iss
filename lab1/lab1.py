import math
import os
import sys
import subprocess
import importlib

# Ensure plotly (and kaleido for static image export) are available
try:
    import plotly.graph_objects as go
    import plotly.io as pio  # noqa: F401
except ImportError:
    print("plotly not found. Installing plotly and kaleido...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "kaleido"])
    importlib.invalidate_caches()
    import plotly.graph_objects as go
    import plotly.io as pio  # noqa: F401

print("plotly is available.")


def update_h(prev_h, A=1.5, beta=0.035, Tp=0.1, Qd=0.05):
    # Stable update: clamp to avoid sqrt of negative due to numeric drift
    h_safe = max(prev_h, 0.0)
    dh = (Qd - beta * math.sqrt(h_safe)) * Tp / A
    return max(prev_h + dh, 0.0)


if __name__ == "__main__":
    # Parameters
    A = 1.5  # m^2/s
    beta = 0.035  # m^(2/5)/s
    Tp = 0.1  # s (time step)
    T_total = 1800  # s (total simulation time)
    Qd = 0.05  # m^3/s

    # Simulation
    steps = int(T_total / Tp)
    times = [i * Tp for i in range(steps + 1)]
    hs = [0.0]  # initial h

    for _ in range(steps):
        hs.append(update_h(hs[-1], A, beta, Tp, Qd))

    hs_rounded = [round(i, 2) for i in hs]
    os.makedirs("results", exist_ok=True)  # Make dir if not exists
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=hs_rounded,
            mode="lines",
            name="h(t)",
            line=dict(width=2, color="#1f77b4"),
        )
    )
    fig.update_layout(
        title="Water level h over time",
        xaxis_title="Time [s]",
        yaxis_title="h [m]",
        template="plotly_white",
    )

    html_path = "results/h_vs_t.html"

    # save an interactive HTML
    fig.write_html(html_path, include_plotlyjs="cdn")

    print(f"Saved interactive plot to {html_path}")
