import numpy as np
from gurobipy import Model, quicksum, GRB

class Colaboratory_Transport():
  def __init__(self, matrix_cost, N_i, cap):
    self.C_ij = matrix_cost
    self.N_i = N_i
    self.cap = cap
    self.I = [i+1 for i in range(len(N_i))]
    self.dem = np.sum([N_i[i] for i in N_i])

  def solve(self, T=[]):
    I = self.I
    I[-1] = 'c'
    I_c = I[:-1]

    m = Model()

    if T == []:

      X = {}
      for i in I:
        for j in I:
          X[i, j] = m.addVar(vtype=GRB.BINARY, name='x[{}, {}]'.format(i, j))

      obj = m.addVar(vtype='C', name='obj')

      m.addConstr(obj == quicksum(self.C_ij[i, j]*X[i, j] for i in I for j in I))

      for i in I_c:
        m.addConstr(quicksum(X[i, j] for j in I_c) <= 1, name='Rest1')

      for j in I_c:
        m.addConstr(quicksum(X[i, j] for i in I_c) <= 1, name='Rest2')

      for i in I_c:
        for j in I_c:
          m.addConstr(X[i, 'c'] <= (1 - X[i, j]), name='Rest3')

      for i in I:
        m.addConstr(X[i, i] == 0, name='Rest4')

      for i in I:
        for j in I:
          m.addConstr(X[j, i] <= (1 - X[i, j]), name='Rest5')

      for i in I:
        for j in I_c:
          m.addConstr(X[j, 'c'] >= X[i, j], name='Rest6')

      for i in I:
        for j in I:
          m.addConstr((self.N_i[i]+self.N_i[j])*X[i, j] <= self.cap, name='Rest7')

      m.addConstr(
          quicksum(self.N_i[i]*X[i, 'c'] for i in I_c) +
          quicksum(self.N_i[i]*X[i, j] for i in I_c for j in I_c) == self.dem, name='Rest8')

      m.setObjective(obj, GRB.MINIMIZE)
      m.optimize()

      X_solved = {(i, j):X[i, j].x for i in I for j in I}
      obj_solved = obj.x

      return X_solved, obj_solved

    else:
      T = T
      X = {}
      for i in I:
        for j in I:
          for t in T:
            X[i, j, t] = m.addVar(vtype=GRB.BINARY, name='x[{}, {}, {}]'.format(i, j, t))

      obj = m.addVar(vtype='C', name='obj')

      m.addConstr(obj == quicksum(self.C_ij[i, j]*X[i, j, t] for i in I for j in I for t in T))

      for t in T:
        for i in I_c:
          m.addConstr(quicksum(X[i, j, t] for j in I_c) <= 1, name='Rest1')

      for t in T:
        for j in I_c:
          m.addConstr(quicksum(X[i, j, t] for i in I_c) <= 1, name='Rest2')

      for t in T:
        for i in I_c:
          for j in I_c:
            m.addConstr(X[i, 'c', t] <= (1 - X[i, j, t]), name='Rest3')

      for t in T:
        for i in I:
          m.addConstr(X[i, i, t] == 0, name='Rest4')

      for t in T:
        for i in I:
          for j in I:
            m.addConstr(X[j, i, t] <= (1 - X[i, j, t]), name='Rest5')

      for t in T:
        for i in I:
          for j in I_c:
            m.addConstr(X[j, 'c', t] >= X[i, j, t], name='Rest6')

      for t in T:
        for i in I:
          for j in I:
            m.addConstr((self.N_i[i]+self.N_i[j])*X[i, j, t] <= self.cap, name='Rest7')

      for t in T:
        m.addConstr(
            quicksum(self.N_i[i]*X[i, 'c', t] for i in I_c) +
            quicksum(self.N_i[i]*X[i, j, t] for i in I_c for j in I_c) == self.dem, name='Rest8')

      for i in I_c:
        m.addConstr(quicksum(X[i, j, t] for j in I_c for t in T) == 1, name='Rest9')

      m.setObjective(obj, GRB.MINIMIZE)
      m.optimize()

      try:
          X_solved = {(i, j, t):X[i, j, t].x for i in I for j in I for t in T}
          obj_solved = obj.x
          return X_solved, obj_solved
      except:
          sol = 'Infactible'
          return sol
