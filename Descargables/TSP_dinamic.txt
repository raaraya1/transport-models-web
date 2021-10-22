from gurobipy import Model, quicksum, GRB
import numpy as np

class dinamic_TSP():
  def __init__(self, matrix_cost, initial_cost, final_cost):
    self.C_ij = matrix_cost
    self.CN_i = initial_cost
    self.CN_f = final_cost

  def solve(self):
    n = int(np.sqrt(len(self.C_ij)))
    I = [i+1 for i in range(n)]
    J = [i+1 for i in range(n)]
    T = [i+1 for i in range(n-1)]
    T_1 = T[:-1]

    m = Model()

    X = {}
    for i in I:
      for j in J:
        for t in T:
          X[i, j, t] = m.addVar(vtype=GRB.BINARY, name=f'x[{i}, {j}, {t}]')

    W_start_i = {}
    for i in I:
      W_start_i[i] = m.addVar(vtype=GRB.BINARY, name=f'w_start[{i}]')

    W_it = {}
    for i in I:
      for t in T:
        W_it[i, t] = m.addVar(vtype=GRB.BINARY, name=f'w[{i}, {t}]')

    pos_it = {}
    for i in I:
      for t in T:
        pos_it[i, t] = m.addVar(vtype=GRB.BINARY, name=f'pos[{i}, {t}]')

    W_end_i = {}
    for i in I:
      W_end_i[i] = m.addVar(vtype=GRB.BINARY, name=f'w_end_i[{i}, {t}]')


    obj = m.addVar(vtype='C', name='obj')

    m.addConstr(obj == quicksum(self.C_ij[i, j]*X[i, j, t] for i in I for j in J for t in T) +
                      quicksum(self.CN_i[i]*W_start_i[i] for i in I) +
                      quicksum(self.CN_f[i]*W_end_i[i] for i in I))

    m.addConstr(quicksum(W_start_i[i] for i in I) == 1)

    for i in I:
      m.addConstr(quicksum(X[i, j, 1] for j in J) == W_start_i[i])


    for j in J:
      m.addConstr(quicksum(X[i, j, t] for i in I for t in T) <= 1)

    for i in I:
      m.addConstr(quicksum(X[i, j, t] for j in J for t in T) <= 1)

    for t in T:
      for i in I:
        m.addConstr(X[i, i, t] == 0)

    for i in I:
      for j in J:
        for t in T:
          m.addConstr(X[i, j, t] <= (1 - W_it[j, t]))

    for i in I:
      m.addConstr(W_it[i, 1] == W_start_i[i])

    for j in J:
      for t in T_1:
        m.addConstr(W_it[j, t+1] == W_it[j, t] + quicksum(X[i, j, t] for i in I))

    m.addConstr(quicksum(W_it[i, T[-1]] for i in I) == len(T))

    m.addConstr(quicksum(X[i, j, t] for i in I for j in J for t in T) == len(T))

    for t in T:
      for j in J:
        m.addConstr(quicksum(X[i, j, t] for i in I) <= 1)

    for t in T:
      for i in I:
        m.addConstr(quicksum(X[i, j, t] for j in J) <= 1)

    for t in T:
      m.addConstr(quicksum(X[i, j, t] for i in I for j in J) >= 1)

    for t in T:
      for i in I:
        m.addConstr(pos_it[i, t] == quicksum(X[i, j, t] for j in J))

    for t in T_1:
      for j in J:
        m.addConstr(quicksum(X[i, j, t] for i in I) == pos_it[j, t+1])

    for j in J:
      m.addConstr(W_end_i[j] == quicksum(X[i, j, T[-1]] for i in I))

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    X_solve = {f'x[{i}, {j}, {t}]':X[i, j, t].x for i in I for j in J for t in T}
    W_start_solve = {f'w_start[{i}]':W_start_i[i].x for i in I}
    W = {f'w[{i}, {t}]':W_it[i, t].x for i in I for t in T}
    pos = {f'pos[{i}, {t}]':pos_it[i, t].x for i in I for t in T}
    W_end = {f'w_end[{i}]':W_end_i[i].x for i in I}
    obj_solve = obj.x

    return X_solve, W_start_solve, W, pos, W_end, obj_solve
