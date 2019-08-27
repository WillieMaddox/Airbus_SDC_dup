
import numpy as np


def holt_winters_second_order_ewma(x, span, beta):
    # Ref http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
    N = x.size
    alpha = 2.0 / (1 + span)
    s = np.zeros((N,))
    b = np.zeros((N,))
    s[0] = x[0]
    for i in range(1, N):
        s[i] = alpha * x[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    return s


def reversed_recombined_holt_winters(x, span=15, beta=0.3):
    # take EWMA in both directions with a smaller span term
    fwd = holt_winters_second_order_ewma(x, span, beta)
    bwd = holt_winters_second_order_ewma(x[::-1], span, beta)
    c = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
    c = np.mean(c, axis=0)  # average
    return c