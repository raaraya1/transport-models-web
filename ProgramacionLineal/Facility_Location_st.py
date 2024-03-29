import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import requests
from gurobipy import Model, quicksum, GRB
from ProgramacionLineal.mapas import *
import folium

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

class Facility_Location_st():
    def __init__(self):
        st.write(r'''
        ## **Modelo con buses de acercamiento en puntos específicos**

        **Supuestos:**

        1. Todos los niños de las familias deben ser transportados al colegio.
        2. Todas las familias cuentan con un mismo vehículo cuya capacidad no supera las 4 personas (sin incluir al conductor)
        3. El costo de la bencina es el mismo entre auto y bus.
        4. Del paradero solo partirá un bus (puede no estar lleno)
        5. No existe un costo asociado a la instalación del paradero.
        6. Si decido instalar un paradero, si o si saldrá un bus desde esa locación.
        7. Todas las familias cuentan con un vehículo con la capacidad de almacenamiento suficiente para enviar a sus hijos.

        **Conjuntos**
        - $i \in I$: Locación de las familias + colegio, donde $i=c$ es el colegio.
        - $j \in J$: Posibles puntos de paradero.

        **Parámetros**

         - $N_{i}$: Número de niños, perteneciente a la familia i.
         - $Dc_{i}$: Distancia entre la locación de la familia i al colegio.
         - $D_{ij}$: Distancia de la familia i al paradero j.
         - $Dj_{j}$: Distancia entre el paradero j al colegio.
         - $CAPb$: Capacidad del bus.
         - $Cb$: Costo de la bencina por km.
         - $N_{bus}$: Cantidad de buses disponibles

        **Variables**

        $X_{j} \in (0, 1)$: Si escojo el lugar j para instalar un paradero.

        $Y_{ij} \in (0, 1)$: Si escojo enviar los niños de la familia i al paradero j.

        $Z_{i} \in (0, 1)$: Si escojo enviar directamente los niños de la familia i al colegio.

        $W_{j}$: Variable de nivel que marca la cantidad de niños en el bus del paradero j.

        **Función Objetivo**

        $$
        min \sum_{i}\sum_{j} CbD_{ij}Y_{ij} + \sum_{j} CbDj_{j}X_{j} + \sum_{i} CbDc_{i}Z_{i}
        $$

        **Restricciones**

        1) No puedo enviar niños al paradero j si este no se encuentra designado antes.

        $$
        Y_{ij} \leq X_{j} \quad \forall i \in I, j \in J
        $$

        2) La cantidad de niños que van dentro del bus es igual a la cantidad de niños que nos enviados al paradero.

        $$
        W_{j} = \sum_{i}N_{i}Y_{ij} \quad \forall j \in J
        $$

        3) No puedo enviar niños al paradero j si, sumando esta cantidad de niños que trae, como resultado se llegase a sobrepasar la capacidad del bus.

        $$
        Y_{ij} \leq (2 - \frac{W_{j} + N_{i}}{CAPb}) \quad \forall i \in I, j \in J
        $$


        4) No puedo enviar los niños directamente al colegio y al paradero al mismo tiempo.

        $$
        Y_{ij} \leq (1 - Z_{i}) \quad \forall i \in I, j \in J
        $$

        5) No puedo enviar a los niños a dos o más paraderos diferentes.

        $$
        \sum_{j} Y_{ij} \leq 1 \quad \forall i \in I
        $$

        6) La suma de niños que llegan al paradero no puede superar la capacidad de almacenamiento del bus.

        $$
        \sum_{i}N_{i}Y_{ij} \leq CAPb \quad \forall j \in J
        $$

        7) La cantidad de niños transportados en bus + la cantidad de niños transportados directamente debe ser = a la cantidad de niños del problema.

        $$
        \sum_{j} W_{j} + \sum_{i} N_{i}Z_{i} = \sum_{i}N_{i}
        $$


        8) (Opcional) Numero de paraderos máximo (Para este caso, también puede ser visto como el numero máximo de buses que dispongo)

        $$
        X_{j} \leq N_{bus} \quad \forall j \in J
        $$
        ''')

    def interactive_model(self):
        st.write('''
                ## Cargar Datos
                ### Matriz de costos (hogares - paraderos)
                ''')

        st.sidebar.write('**Archivos descargables**')

        coord_file = requests.get('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/coordenadas%20(hogares%20-%20paraderos%20-%20colegio).csv')
        st.sidebar.download_button(label='Locaciones (hogares - paraderos - colegio).csv',
                                   data=coord_file.content,
                                   file_name='coordenadas (H-P-C).csv',
                                   mime='text/csv')

        cost_matrix_hogares_colegio_csv = pd.read_csv('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/cost_matrix%20(hogares%20-%20colegio).csv')
        st.sidebar.download_button(label='matrix_cost (hogares-colegio).csv',
                                   data=cost_matrix_hogares_colegio_csv.to_csv(index=False),
                                   file_name='matrix_cost(hogares-colegio).csv',
                                   mime='text/csv')

        cost_matrix_hogares_paraderos_csv = pd.read_csv('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/cost_matrix%20(hogares%20-%20paraderos).csv')
        st.sidebar.download_button(label='matrix_cost (hogares-paraderos).csv',
                                   data=cost_matrix_hogares_paraderos_csv.to_csv(index=False),
                                   file_name='matrix_cost (hogares-paraderos).csv',
                                   mime='text/csv')

        cost_matrix_paraderos_colegio_csv = pd.read_csv('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/cost_matrix%20(paraderos%20-%20colegio).csv')
        st.sidebar.download_button(label='matrix_cost (paraderos-colegio).csv',
                                   data=cost_matrix_paraderos_colegio_csv.to_csv(index=False),
                                   file_name='matrix_cost(paraderos-colegio).csv',
                                   mime='text/csv')

        model_file = requests.get('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/Facility_Location.txt')
        #st.write(model_file.content)
        st.sidebar.download_button(label='model_file.txt',
                                   data=model_file.content,
                                   file_name='model_file.txt')


        st.sidebar.write('**Datos**')

        cost_matrix = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos entre los hogares y los paraderos (csv - xlsx)')
        initial_route_cost = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos entre los hogares y el colegio (csv - xlsx)')
        last_route_cost = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos entre los paraderos y el colegio (csv - xlsx)')

        if cost_matrix is not None:
            df = pd.read_csv(cost_matrix, sep=';', header=None)
            n_locations = len(df)
            n_paraderos = len(df.to_numpy()[0])
            columns = [i+1 for i in range(n_paraderos)]
            rows = [i+1 for i in range(n_locations)]
            df.columns = columns
            df.index = rows
            st.dataframe(df)
        else:
            st.write('Subir archivo con matriz de costos (hogares - paraderos)')

        st.write('''
                ### Matriz de costos (hogares - colegio)
        ''')

        if initial_route_cost is not None:
            df1 = pd.read_csv(initial_route_cost, sep=';',header=None)
            st.dataframe(df1)
        else:
            st.write('Subir archivo con matriz de costos (hogares - colegio)')

        st.write('''
                ### Matriz de costos (paraderos - colegio)
        ''')

        if last_route_cost is not None:
            df2 = pd.read_csv(last_route_cost, sep=';',header=None)
            st.dataframe(df2)
        else:
            st.write('Subir archivo con matriz de costos (paraderos - colegio)')


        st.write('''
                #### Cantidad de niños en lugar i ($N_{i}$)
                ''')

        if cost_matrix is not None:
            cap = st.sidebar.text_input('Capacidad de los buses')
            try:
                cap = int(cap)
            except:
                st.write('Ajustar capacidad de los buses')


        if cost_matrix is not None and type(cap) != np.str:
            N_i = {}
            n_i = []
            for i in range(n_locations):
                valor = st.sidebar.slider(f'Cantidad de niños del lugar {i+1}', 1, cap)
                N_i[i+1] = valor
                n_i.append([valor])

            df_ni = pd.DataFrame(n_i, columns=['N_i'], index=[i+1 for i in range(len(n_i))])
            st.dataframe(df_ni)

        if cost_matrix is not None and type(cap) != np.str:

            # Modelos
            ## Parametros
            hogares = [i+1 for i in range(n_locations)]
            paraderos = [i+1 for i in range(n_paraderos)]

            # costo entre hogares y paraderos
            C_ij = {}
            for i in hogares:
                for j in paraderos:
                    C_ij[(i, j)] = df[j][i]

            # costos entre hogares y el colegio
            Cc_i = {}
            for i in hogares:
                Cc_i[i] = df1[0][i-1]

            # costo entre paraderos y el colegio
            Cc_j = {}
            for j in paraderos:
                Cc_j[j] = df2[0][j-1]


            N_i['c'] = 0
            #st.write(str(N_i))

            st.write('''
                    ### **Resultados**
                    ''')

            modelo = Facility_Location(matrix_cost=C_ij,
                               dem=N_i,
                               initial_route_cost=Cc_i,
                               last_route_cost=Cc_j,
                               cap=cap)
            try:
                X_solve, Y_solve, Z_solve, W_solve, obj_solve = modelo.solve()
                st.write('''
                        ##### Paraderos escogidos
                ''')

                for i in X_solve:
                  if X_solve[i] > 0:
                    st.write(f'Paradero {i}')

                st.write('''
                        ##### Transportes en buses (hogar -> paradero -> colegio)
                ''')

                for i in Y_solve:
                  if Y_solve[i] > 0:
                    st.write(f'{i} ')

                st.write('''
                        ##### Viajes directos (hogar -> colegio)
                ''')

                for i in Z_solve:
                  if Z_solve[i] > 0:
                    st.write(f'{i}')

                st.write('''
                        ##### Cantidad de niños en el paradero
                ''')

                for i in W_solve:
                  if W_solve[i] > 0:
                    st.write(f'{i} = {W_solve[i]}')

                st.write('''
                        ##### Costo total
                ''')

                st.write(f'${obj_solve}')
            except:
                sol = modelo.solve()
                st.write(str(sol))

            st.sidebar.write('**Visualizacion de resultados**')

            st.write(r'''
            ### Visualización de Resultados
            Para la visualización de los resultados es
            necesario crearnos una cuenta de usuario en
            https://openrouteservice.org y luego utilizar la clave
            generada (esta luego la debemos introducir en el panel a la izquierda)

            - **Puntos Verdes**: Hogares
            - **Puntos Azules**: Paraderos
            - **Punto Rojo**: Colegio

            **Nota:** Es necesario establecer, tanto en la matriz de costo como
            en las coordenadas de los lugares, la última locación como el colegio.

            ''')

            clave = st.sidebar.text_input('Clave Token')
            coordenadas = st.sidebar.file_uploader('Archivo con las locaciones (.csv)')

            if coordenadas is not None:

                df =  pd.read_csv(coordenadas, delimiter=';', encoding= 'unicode_escape')
                st.dataframe(df)

                Hogares = [i.split(',') for i in df['Hogares'][:len(hogares)]]
                Hogares = [[float(i[0]), float(i[1])] for i in Hogares]

                Paraderos = [i.split(',') for i in df['Paraderos'][:len(paraderos)]]
                Paraderos = [[float(i[0]), float(i[1])] for i in Paraderos]

                colegio = [float(i) for i in df['Colegio'][0].split(',')]

                def lat_lon(lista):
                    lat = [i[1] for i in lista]
                    lon = [i[0] for i in lista]
                    lugares = [[lat[i], lon[i]] for i in range(len(lat))]
                    return lugares

                h_folium = lat_lon(Hogares)
                p_folium = lat_lon(Paraderos)
                c_folium = [colegio[1], colegio[0]]

                h_posiciones = {i+1:Hogares[i] for i in range(len(Hogares))}
                p_posiciones = {i+1:Paraderos[i] for i in range(len(Paraderos))}
                rutas_h_p = {(i, j): [h_posiciones[i], p_posiciones[j]] for i in h_posiciones for j in p_posiciones}
                rutas_h_c = {i: [h_posiciones[i], colegio] for i in h_posiciones}
                rutas_p_c = {j: [p_posiciones[j], colegio] for j in p_posiciones}

                rutas_selec_h_p = []
                for i in Y_solve:
                  if Y_solve[i] > 0:
                    valor = i[3:-1].split(',')
                    valor = (int(valor[0]), int(valor[1]))
                    rutas_selec_h_p.append(valor)

                rutas_selec_h_c = []
                for i in Z_solve:
                  if Z_solve[i] > 0:
                      valor = int(i[2:])
                      rutas_selec_h_c.append(valor)

                rutas_selec_p_c = []
                for i in X_solve:
                  if X_solve[i] > 0:
                      valor = int(i[2:])
                      rutas_selec_p_c.append(valor)

                rutas_selec_h_p_pos = [rutas_h_p[i] for i in rutas_selec_h_p]
                rutas_selec_h_c_pos = [rutas_h_c[i] for i in rutas_selec_h_c]
                rutas_selec_p_c_pos = [rutas_p_c[i] for i in rutas_selec_p_c]

                rutas_especificas_h_p = {}
                rutas_especificas_h_c = {}
                rutas_especificas_p_c = {}

                def gen_rutas(rutas_selec_pos, origen, destino):
                    rutas_especificas = {}
                    for ind, i in enumerate(rutas_selec_pos):
                        origen_lat = i[0][0]
                        origen_lon =  i[0][1]
                        destino_lat = i[1][0]
                        destino_lon = i[1][1]
                        rutas_especificas[f'ruta{ind+1} ({origen} -> {destino})'] = [[float(origen_lat), float(origen_lon)], [float(destino_lat), float(destino_lon)]]
                    return rutas_especificas

                rutas_especificas_h_p = gen_rutas(rutas_selec_h_p_pos, 'H', 'P')
                rutas_especificas_h_c = gen_rutas(rutas_selec_h_c_pos, 'H', 'C')
                rutas_especificas_p_c = gen_rutas(rutas_selec_p_c_pos, 'P', 'C')


                lugares = []
                for i in Hogares: lugares.append(i)
                for i in Paraderos: lugares.append(i)
                lugares.append(colegio)

                rutas_especificas = {}
                for i in rutas_especificas_h_p: rutas_especificas[i] = rutas_especificas_h_p[i]
                for i in rutas_especificas_h_c: rutas_especificas[i] = rutas_especificas_h_c[i]
                for i in rutas_especificas_p_c: rutas_especificas[i] = rutas_especificas_p_c[i]

                mapa = Mapas(clave, lugares)
                map = mapa.Mapa_con_rutas(rutas_especificas=rutas_especificas)

                #colorear los puntos
                rojo = [colegio[1], colegio[0]]
                azul = lat_lon(Paraderos)
                verde = lat_lon(Hogares)

                folium.Marker(rojo, popup='C', icon=folium.Icon(color="red")).add_to(map)
                for idx, i in enumerate(azul): folium.Marker(i, popup=f'P{idx+1}', icon=folium.Icon(color="blue")).add_to(map)
                for idx, i in enumerate(verde): folium.Marker(i, popup=f'H{idx+1}', icon=folium.Icon(color="green")).add_to(map)

                folium_static(map)
