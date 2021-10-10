from gurobipy import Model, quicksum, GRB
import numpy as np

class TSP_MTZ():
  def __init__(self, Matrix_cost):
    self.C_ij = Matrix_cost
    self.N = int(np.sqrt(len(self.C_ij)))

  def solve(self):
    I = [i+1 for i in range(self.N)]
    J = I
    I_1 = I[1:]
    J_1 = I_1

    m = Model()

    X = {}
    for i in I:
      for j in J:
        X[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x[{i}, {j}]')

    U = {}
    for i in I:
      U[i] = m.addVar(vtype='C', lb=0, name=f'u[{i}]')

    obj = m.addVar(vtype='C', name='obj')

    m.addConstr(obj == quicksum(self.C_ij[i, j]*X[i, j] for i in I for j in J))

    for i in I:
      m.addConstr(quicksum(X[i, j] for j in J) == 1)

    for j in J:
      m.addConstr(quicksum(X[i, j] for i in I) == 1)

    m.addConstr(U[1] == 1)

    for i in I_1:
      m.addConstr(2 <= U[i])

    for i in I:
      for j in J_1:
        m.addConstr(U[i] - U[j] + 1 <= (self.N - 1)*(1 - X[i, j]))

    for i in I:
      m.addConstr(X[i, i] == 0)

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    X_solve = {f'x[{i}, {j}]':X[i, j].x for i in I for j in J}
    obj_solve = obj.x

    return X_solve, obj_solve
