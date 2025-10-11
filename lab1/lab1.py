import math

def h(n,
      A=1.5,
      beta=0.035,
      Tp=0.1,
      Qd=0.05
):
    return (1/A * (Qd * n - beta * math.sqrt(h(n))) * Tp) + h(n)



if __name__ == '__main__':
    A = 1.5 # m^2/s
    beta = 0.035 # m^(2/5)/s
    Tp = 0.1 # s
    t = 1800 # s
    Qd = 0.05 # m^3/s

    h_all = []
    # Create time series
    for i in range(t):
        h_calculated = h(n = i*Tp)
        print(h_calculated)
        h_all.append(h_calculated)

    print(h_all)

