import random as r
import numpy as np


# modelo de pontos
a, b = -3.11, 13.338
n = 100
xs = [(-1.992), (-1.984), (-1.912), (-1.712), (-1.576), (-1.496), (-1.488), (-1.416), (-1.376), (-1.36), (-1.344), (-1.328), (-1.232), (-1.192), (-1.184), (-1.152), (-1.032), (-0.936), (-0.8), (-0.592), (-0.552), (-0.464), (-0.152), (0.04), (0.064), (0.088), (0.104), (0.168), (0.216), (0.248), (0.288), (0.464), (0.568), (0.576), (0.64), (0.664), (0.712), (0.96), (0.992), (1.0), (1.024), (1.192), (1.232), (1.248), (1.256), (1.296), (1.432), (1.68), (1.712), (1.944)]

xs.sort()
num = 10
data_coefs = [r.random() for _ in range(num)]
def data(x):
    # a + b * x + erro
    erro = r.random() / 10
    val = sum(c * x ** i for i, c in enumerate(data_coefs)) + erro
    return val
ys = [( 5.672), ( 5.834), ( 6.97), ( 8.829), ( 8.704), ( 8.572), ( 8.325), ( 7.947), ( 7.745), ( 7.818), ( 7.854), ( 7.465), ( 7.067), ( 6.311), ( 6.298), ( 6.483), ( 5.286), ( 4.854), ( 4.099), ( 3.597), ( 3.834), ( 3.904), ( 4.446), ( 5.572), ( 5.34), ( 5.515), ( 5.66), ( 5.728), ( 6.108), ( 6.241), ( 6.346), ( 6.772), ( 6.753), ( 6.511), ( 6.383), ( 6.638), ( 6.591), ( 5.669), ( 5.294), ( 5.112), ( 4.994), ( 3.762), ( 3.848), ( 3.559), ( 3.67), ( 3.203), ( 2.156), ( 1.456), ( 1.913), ( 4.197)]


#método, dos mínimos quadrados para retas
def min_q(pontos):
    n = len(pontos)
    sumxk = sum(x for x, _ in pontos)
    sumxk2 = sum(x ** 2 for x, _ in pontos)
    sumyk = sum(y for _, y in pontos)
    sumykxk = sum(x * y for x, y in pontos)
    A = [[n, sumxk], [sumxk,sumxk2]]
    B = [sumyk, sumykxk]
    coefs = np.linalg.solve(A, B)
    return coefs # a0 e a1

# método dos minimos quadrados geral
def least_squares(pontos, k): # se k == 1, então o resultado é o mesmo de min_q(pontos)
    # vamos obter um sistema (k+1)x(k+1)
    n = len(pontos)
    A = {}
    B = []
    for i in range(k + 1):
        A[i] = {}
        for j in range(k + 1):
            if j >= i:
                A[i][j] = sum(x ** (i + j) for x, _ in pontos)
            else:
                A[i][j] = A[j][i]
    A = [[A[i][j] for j in range(k + 1)] for i in range(k + 1)]
    for i in range(k + 1):
        B.append(sum(y * x ** i for x, y in pontos))
    coefs = np.linalg.solve(A, B)
    return coefs


pontos = list(zip(xs, ys))
c = least_squares(pontos, k=5)  
print(c)

def fit_poly(x):
    return sum(c * x ** i for i, c in enumerate(c)) # sum c[j] * x ** j, j=0,1,2,...,k

erro = sum((y - fit_poly(x)) ** 2 for x, y in pontos)
t = np.arange(a, b, 0.01)
