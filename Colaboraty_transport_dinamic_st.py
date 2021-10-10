import streamlit as st
import pandas as pd
import numpy as np
import requests
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

class Colaboraty_transport_dinamic_st():
    def __init__(self):
        st.write(r'''
        ## **Modelo con sistemas de apoyo entre vecinos en el tiempo**
        **Supuestos:**

        1. Todos los niños de las familias deben ser transportados al colegio.
        2. Todas las familias cuentan con un mismo vehiculo cuya capacidad no supera las 4 personas (sin incluir al conductor)
        3. El costo de la bencina es igual para todas las familias
        4. Solo puedo ayudar a 1 vecino a transportar a sus hijos.
        5. Si ayudo a un vecino, transportaré a todos sus hijos.


        **Conjuntos:**

         - $i \in I$: Locacion de las familias + colegio, donde $i=c$ es el colegio.
         - $t \in T$: Periodo (5 en total)

        **Parametros:**

         - $N_{i}$: Numero de niños, perteneciente a la familia i.
         - $Dc_{i}$: Distancia entre la locacion de la familia i al colegio.
         - $D_{ij}$: Distancia entre la locacion de la familia i y la locacion de la familia j.
         - $Cb$: Costo de la bencina por km.

        **Variables:**

        $X_{ijt} \in (0, 1)$: Si se realiza un viaje desde la locacion i hasta la locacion j en el periodo t.

        **Funcion objetivo:**

        $$
        Min \sum_{t} \sum_{i}\sum_{j} CbD_{ij}X_{ijt}
        $$

        **Restricciones:**

        1. Solo puedo ayudar a un vecino por periodo.

        $$
        \sum_{j \neq c} X_{ijt} \leq 1 \quad \forall i \in I-(c), t \in T
        $$

        2. Solo puedo recibir ayuda de 1 vecino por periodo.

        $$
        \sum_{i \neq c} X_{ijt} \leq 1 \quad \forall j \in I-(c), t \in T
        $$

        3. Si ayudo, no me voy directo al colegio.

        $$
        X_{ict} \leq (1 - X_{ijt}) \quad \forall i, j \in I, t \in T
        $$

        4. Ayudarse a uno mismo no cuenta.

        $$
        X_{iit} = 0 \quad \forall i \in I, t \in T
        $$

        5. No devolverse en los recorridos.

        $$
        X_{jit} \leq (1 - X_{ijt}) \quad \forall i, j \in I, t \in T
        $$

        6. Asegurarse de enviar a los niños luego de recogerlos.

        $$
        X_{jct} \geq X_{ijt} \quad \forall i, j \in I-(c), t \in T
        $$

        7. No puedo transportar mas niños que la capacidad del auto.

        $$
        (N_{i} + N_{j})X_{ijt} \leq 4 \quad  \forall i, j \in I, t \in T
        $$

        8. Debo transportar a todos los niños. (Niños que van directo + niños que son recogidos = total de niños).

        $$
        \sum_{i \neq c} (N_{i}X_{ict} + \sum_{j \neq c}N_{i}X_{ijt}) = \sum_{i} N_{i} \quad \forall t \in T
        $$

        9. Cada familia debe prestar ayuda una vez por periodo.

        $$
        \sum_{t} \sum_{j \neq c} X_{ijt} \geq 1 \quad \forall i \in I-(c)
        $$
        ''')

    def interactive_model(self):
        st.write('''
                ### Cargar Datos
                #### Matriz de costos ($C_{ij} = Cb*D_{ij}$)
                ''')

        st.sidebar.write('**Archivos descargables**')

        cost_matrix_csv = pd.read_csv('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/cost_matrix.csv')
        st.sidebar.download_button(label='matrix_cost.csv',
                                   data=cost_matrix_csv.to_csv(index=False),
                                   file_name='matrix_cost.csv',
                                   mime='text/csv')

        model_file = requests.get('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Colaboratory_Transport.py')
        #st.write(model_file.content)
        st.sidebar.download_button(label='model_file.txt',
                                   data=model_file.content,
                                   file_name='model_file.txt')

        st.sidebar.write('**Datos**')

        cost_matrix = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos (csv - xlsx)')
        if cost_matrix is not None:
            df = pd.read_csv(cost_matrix, sep=';', header=None)
            n_locations = len(df)
            columns = [i+1 for i in range(n_locations)]
            rows = [i+1 for i in range(n_locations)]
            df.columns = columns
            df.index = rows
            st.table(df)
        else:
            st.write('Subir archivo con matriz de costos')

        st.write('''
                #### Cantidad de niños en lugar i ($N_{i}$)
                ''')

        if cost_matrix is not None:
            cap = st.sidebar.text_input('Capacidad de los autos')
            t = st.sidebar.text_input('Cantidad de periodos')
            try:
                cap = int(cap)
            except:
                st.write('Ajustar capacidad de los autos')
            try:
                t = int(t)
            except:
                st.write('Ajustar la cantidad de periodos')


        if cost_matrix is not None and type(cap) != np.str and type(t) != np.str:
            N_i = {}
            n_i = []
            for i in range(n_locations-1):
                valor = st.sidebar.slider(f'Cantidad de niños del lugar {i+1}', 1, cap)
                N_i[i+1] = valor
                n_i.append([valor])

            df_ni = pd.DataFrame(n_i, columns=['N_i'], index=[i+1 for i in range(len(n_i))])
            st.dataframe(df_ni)


        if cost_matrix is not None and type(cap) != np.str and type(t) != np.str:

            # Modelos
            ## Parametros
            filas = [i+1 for i in range(len(df))]
            columnas = [i+1 for i in range(len(df))]
            last_term = columnas[-1]

            #filas[-1] = 'c'
            #columnas[-1] = 'c'

            C_ij = {}
            for i in filas:
                for j in columnas:
                    if i == last_term and j == last_term:
                        C_ij[('c', 'c')] = df[i][j]
                    elif i == last_term:
                        C_ij[('c', j)] = df[i][j]
                    elif j == last_term:
                        C_ij[(i, 'c')] = df[i][j]
                    else:
                        C_ij[(i, j)] = df[i][j]

            #st.write(str(C_ij))

            N_i['c'] = 0
            #st.write(str(N_i))

            st.write('''
                    ### **Resultados**
                    ''')

            T = [i+1 for i in range(t)]

            modelo = Colaboratory_Transport(matrix_cost=C_ij, N_i=N_i, cap=cap)

            try:
                X, obj = modelo.solve(T=T)
                st.write('''
                        ##### Configuración optima de transporte
                ''')

                for t in T:
                    st.write(f'**Periodo {t}**')
                    for i in X:
                        if X[i] > 0 and i[-1] == t:
                            st.write(f'Punto {i[0]} -> Punto {i[1]}')
                st.write(f'**Costo Total = ${obj}**')
            except:
                sol = modelo.solve(T=T)
                st.write(str(sol))