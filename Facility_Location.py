from gurobipy import Model, quicksum, GRB

class Facility_Location():
  def __init__(self, matrix_cost, dem, initial_route_cost, last_route_cost, cap, Q=False):
    self.D_ij = matrix_cost
    self.N_i = dem
    self.Dc_i = initial_route_cost
    self.CAPb = cap
    if Q == False:
      self.Nbus = int(len(self.D_ij)/len(self.N_i))
    else:
      self.Nbus = Q
    self.Dj_j = last_route_cost


  def solve(self):
    I = [i+1 for i in range(len(self.N_i))]
    I[-1] = 'c'
    I_c = I[:-1]
    J = [i+1 for i in range(self.Nbus)]

    m = Model()

    X = {}
    for j in J:
      X[j] = m.addVar(vtype=GRB.BINARY, name=f'x[{j}]')

    Y = {}
    for i in I:
      for j in J:
        Y[i, j] = m.addVar(vtype=GRB.BINARY, name=f'y[{i}, {j}]')

    Z = {}
    for i in I:
      Z[i] = m.addVar(vtype=GRB.BINARY, name=f'z[{i}]')

    W = {}
    for j in J:
      W[j] = m.addVar(vtype='C', name=f'w[{j}]')

    obj = m.addVar(vtype='C', name='obj')

    m.addConstr(obj == quicksum(self.D_ij[i, j]*Y[i, j] for i in I_c for j in J ) +
                      quicksum(self.Dj_j[j]*X[j] for j in J) +
                      quicksum(self.Dc_i[i]*Z[i] for i in I_c))

    for i in I_c:
      for j in J:
        m.addConstr(Y[i, j] <= X[j], name='Rest1')

    for j in J:
      m.addConstr(W[j] == quicksum(self.N_i[i]*Y[i, j] for i in I_c), name='Rest2')

    for i in I_c:
      for j in J:
        m.addConstr(Y[i, j] <= 2-((W[j] + self.N_i[i])/self.CAPb), name='Rest3')

    for i in I_c:
      for j in J:
        m.addConstr(Y[i, j] <= (1-Z[i]), name='Rest4')

    for i in I_c:
      m.addConstr(quicksum(Y[i, j] for j in J) <= 1, name='Rest5')

    for j in J:
      m.addConstr(quicksum(self.N_i[i]*Y[i, j] for i in I_c) <= self.CAPb, name='Rest6')

    m.addConstr(quicksum(W[j] for j in J) + quicksum(self.N_i[i]*Z[i] for i in I_c) == quicksum(self.N_i[i] for i in I_c), name='Rest7')

    for j in J:
      m.addConstr(X[j] <= self.Nbus, name='Rest8')

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    X_solve = {f'X_{j}':X[j].x for j in J}
    Y_solve = {f'Y_{(i, j)}':Y[i, j].x for i in I for j in J}
    Z_solve = {f'Z_{i}':Z[i].x for i in I}
    W_solve = {f'W_{j}':W[j].x for j in J}
    obj_solve = obj.x

    return X_solve, Y_solve, Z_solve, W_solve, obj_solve
