from sympy import *

init_printing(use_unicode=True)

A = Matrix([['a1', 'a2', 'a3'],
            ['a4', 'a5', 'a6'],
            ['a7', 'a8', 'a8']])

B = Matrix([['b1', 'b2'],
            ['b3', 'b4'],
            ['b5', 'b6']])

c = Matrix([['c1'],
            ['c2'],
            ['c3']])

s = Matrix([['s1'],
            ['s2'],
            ['s3']])

K = Matrix([['K1', 'K2', 'K3'],
            ['K4', 'K5', 'K6']])

k = Matrix([['k1'],
            ['k2']])

V_0 = Matrix([['v0']])

V_x = Matrix([['v1'],
              ['v2'],
              ['v3']])

V_xx = Matrix([['V0', 'V1', 'V2'],
               ['V1', 'V4', 'V5'],
               ['V2', 'V5', 'V8']])

u = K * s + k
print("U is {}".format(u))

next_state = A * s + B * u + c
print("Next state is {}".format(next_state))

analytic_k_grad = B.T * V_x + 2 * B.T * V_xx * next_state
analytic_K_grad = B.T * V_x * s.T + 2 * B.T * V_xx * next_state * s.T

value_est = V_0 + V_x.T * next_state + next_state.T * V_xx * next_state
print("Value est is {}".format(value_est))

print("Gradient with respect to K")
print(diff(next_state, K))

print("Gradient with respect to k")
print(diff(next_state, k))
