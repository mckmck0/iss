import math

def update_h(prev_h, A=1.5, beta=0.035, Tp=0.1, Qd=0.05):
    # Update equation for stable h
    dh = (Qd - beta * math.sqrt(prev_h)) * Tp / A
    return prev_h + dh

if __name__ == '__main__':
    A = 1.5 # m^2/s
    beta = 0.035 # m^(2/5)/s
    Tp = 0.1 # s
    t = 1800 # s
    Qd = 0.05 # m^3/s

    h_all = []
    h_current = 0.0  # Initial h value

    # Create time series
    for _ in range(int(t//Tp) + 1):
        h_next = update_h(h_current, A, beta, Tp, Qd)
        print(h_next)
        h_all.append(h_next)
        h_current = h_next  # Update for next iteration

    print(h_all)