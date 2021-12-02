import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import requests
from gurobipy import Model, quicksum, GRB
from ProgramacionLineal.mapas import *
import folium

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

class TSP_MTZ_st():
    def __init__(self):
        st.write(r'''
        ### Modelo con bus de acercamiento a domicilio
        **Supuestos**

        - El bus cuenta con la suficiente capacidad para transportar a todos los niños.

        **Conjuntos**

        - $i \in I$: Conjunto de localidades

        - $j \in J$: Conjunto de localidades (el mismo que $I$)

        **Parámetros**

        - $C_{ij}$: Costo de ir desde el punto i al punto j.

        - $N$: Total de localidades

        **Variables**

        - $X_{ij} \in (0, 1)$: Si recorro desde el punto i al punto j.

        - $U_{i}$: Variable auxiliar en punto i

        **Función Objetivo**

        $$
        min \sum_{i} \sum_{j} C_{ij}X_{ij}
        $$

        **Restricciones**

        1) Desde un nodo solo sale 1 único arco.

        $$
        \sum_{j} X_{ij} = 1 \quad \forall i \in I
        $$


        2) Hacia un nodo solo entra 1 único arco.

        $$
        \sum_{i} X_{ij} = 1 \quad \forall j \in I
        $$

        3) Eliminación de subciclos (Restricciones MTZ)

        $$
        U_{1} = 1
        $$

        $$
        2 \leq U_{i} \quad \forall i \neq 1 \in I
        $$

        $$
        U_{i} \leq N \quad \forall i \neq 1 \in I
        $$

        $$
        U_{i} - U_{j} + 1 \leq (N - 1)(1 - X_{ij}) \quad \forall i \in I, j \neq 1 \in J
        $$
        ''')

    def interactive_model(self):
        st.write('''
                ### Cargar Datos
                #### Matriz de costos ($C_{ij} = Cb*D_{ij}$)
                ''')

        st.sidebar.write('**Archivos descargables**')

        coord_file = pd.read_csv('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/coordenadas.csv')
        st.sidebar.download_button(label='Locaciones.csv',
                                   data=coord_file.to_csv(index=False),
                                   file_name='coordenadas.csv',
                                   mime='text/csv')

        cost_matrix_csv = pd.read_csv('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/cost_matrix.csv')
        st.sidebar.download_button(label='matrix_cost.csv',
                                   data=cost_matrix_csv.to_csv(index=False),
                                   file_name='matrix_cost.csv',
                                   mime='text/csv')

        model_file = requests.get('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/Descargables/TSP_MTZ.txt')
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

        if cost_matrix is not None:

            # Modelo
            ## Parametros
            filas = [i+1 for i in range(len(df))]
            columnas = [i+1 for i in range(len(df))]

            #filas[-1] = 'c'
            #columnas[-1] = 'c'

            C_ij = {}
            for i in filas:
                for j in columnas:
                    C_ij[(i, j)] = df[i][j]

            st.write('''
                    ### **Resultados**
                    ''')

            modelo = TSP_MTZ(Matrix_cost=C_ij)

            try:
                X, obj = modelo.solve()
                st.write('''
                        ##### Configuración optima de transporte
                ''')

                for i in X:
                    if X[i] > 0:
                        st.write(f'{i}')
                st.write(f'**Costo Total = ${obj}**')
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

            - **Puntos Verdes**: Donde se inicia el transporte
            - **Puntos Azules**: Donde se recogen niños
            - **Punto Rojo**: Destino final (para este caso el colegio)

            **Nota:** Es necesario establecer, tanto en la matriz de costo como
            en las coordenadas de los lugares, la última locación como el colegio.
            ''')


            clave = st.sidebar.text_input('Clave Token')
            coordenadas = st.sidebar.file_uploader('Archivo con las locaciones (.csv)')

            if coordenadas is not None:
                df = pd.read_csv(coordenadas, header=None, sep=',')
                df_m = df.copy()
                df_m.columns=['Longitude', 'Latitude']
                df_show = df_m.copy()
                df_show.index=[i+1 for i in range(len(df_m))]
                st.write('''#### Direcciones''')
                st.table(df_show)
                lat = [i for i in df[0]]
                lon = [i for i in df[1]]
                lugares = [[lat[i], lon[i]] for i in range(len(lat))]

                posiciones = {i+1:[df_m['Longitude'][i], df_m['Latitude'][i]] for i in range(len(df_m))}
                rutas = {(i, j): [posiciones[i], posiciones[j]] for i in posiciones for j in posiciones}

                rutas_selec = []
                for i in X:
                  if X[i] > 0:
                      if i[0] == 'c':
                          rutas_selec.append((last_term, i[1]))
                      elif i[1] == 'c':
                          rutas_selec.append((i[0], last_term))
                      else:
                          rutas_selec.append(i)
                # correccion
                rutas_selec = [i[2:-1].split(',') for i in rutas_selec]
                rutas_selec = [(int(i[0]), int(i[1])) for i in rutas_selec]

                st.write('''#### Rutas''')
                for ind, i in enumerate(rutas_selec):
                    st.write(f'**Ruta {ind+1}** = {i}')

                rutas_selec_pos = []
                for i in rutas_selec:
                    rutas_selec_pos.append(rutas[i])

                rutas_especificas = {}

                for ind, i in enumerate(rutas_selec_pos):
                    origen_lat = i[0][0]
                    origen_lon =  i[0][1]
                    destino_lat = i[1][0]
                    destino_lon = i[1][1]
                    rutas_especificas[f'ruta{ind+1}'] = [[float(origen_lat), float(origen_lon)], [float(destino_lat), float(destino_lon)]]

                mapa = Mapas(clave, lugares)
                map = mapa.Mapa_con_rutas(rutas_especificas)

                #colorear los puntos
                rojo = [len(posiciones)]
                amarillo = [i[1] for i in rutas_selec if i[1] != rojo]
                verde = [i[0] for i in rutas_selec if i[0] not in amarillo]


                for i in posiciones:
                    cord_fol = [posiciones[i][1], posiciones[i][0]]# para ordenar las coordenadas
                    if i in rojo:
                        folium.Marker(cord_fol, popup=f'P{i}', icon=folium.Icon(color="red")).add_to(map)
                    elif i in amarillo:
                        folium.Marker(cord_fol, popup=f'P{i}', icon=folium.Icon(color="blue")).add_to(map)
                    elif i in verde:
                        folium.Marker(cord_fol, popup=f'P{i}', icon=folium.Icon(color="green")).add_to(map)



                folium_static(map)
