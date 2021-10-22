import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import requests
from gurobipy import Model, quicksum, GRB
from ProgramacionLineal.mapas import *
import folium

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

class TSP_dinamic_st():
    def __init__(self):
        st.write(r'''
        ## **Variante TSP**
        La motivación de elaborar esta variante es que no me gusta la manera
        tradicional en como se modelaba este problema. Así, espero poder
        desarrollarlo de una manera mas intuitiva y con esto, también espero
        que este modelo sea equiparable en cuanto a tiempos de procesamiento
        con los modelos tradicionales.

        ### **¿En que consiste el TSP?**
        Este problema hace referencia a un vendedor que tiene por objetivo
        viajar por diferentes lugares para ir vendiendo sus mercancías. De esta
        manera, el problema que surge entonces es que a este vendedor le
        gustaría recorrer todas estas ciudades al menor costo posible.
        (En este recorrido se exige que el punto de inicio corresponda con el
        punto de termino del recorrido)

        ### **¿Cómo me imagino el problema?**

        Generalmente cuanto empiezo a modelar un problema, siempre me pongo en
        el lugar de quien tiene que tomar las decisiones o acciones (la mayoría
        de las veces estas pasan a ser variables directas del modelo).

        Para este caso, me tendré que poner en el lugar del vendedor y, en
        consecuencia, las preguntas validas que me podría hacer son las siguientes:

        1. **¿Cuál es mi inicio?**
        2. **¿Dónde estoy parado?**
        3. **¿Cuántos viajes tendré que hacer?**
        4. **¿Por cual empiezo?**
        5. **¿Cuál es el siguiente lugar que quiero visitar?**
        6. **¿Cuáles son los lugares que me faltan por recorrer?**

        Ahora, si nos damos cuenta, algunas de estas preguntas se encuentran en
        función de otras (por ejemplo, la pregunta 5 esta directamente relacionada
        con la pregunta 6) y no solo eso, sino que algunas de estas respuestas
        variaran de acuerdo al momento en que se formulan (por ejemplo, si en
        un inicio me pregunto cuantos lugares me faltan por recorrer, lo mas
        seguro es que si me vuelvo a hacer esta pregunta luego de haber recorrido
        algunas ciudades, la respuesta a esta pregunta cambie)

        Así, lo primero que se me ocurre es que, para ir abordando estas preguntas,
        podemos establecer un ordenamiento secuencial con la siguiente estructura:

        **Ejemplo de secuencia.**

        **inicio**

        - ¿Cuál es mi inicio? -> $P_{0}$
        - ¿Cuántos viajes tendré que hacer? -> T = 5
        - ¿Cuáles son los lugares que me faltan por recorrer? -> $[P_{1}, P_{2}, P_{3}, P_{4}, P_{5}]$
        - ¿Por cual empiezo? -> $P_{1}$

        **Ciclo**

        - ¿Dónde estoy parado? -> $P_{1}$
        - ¿Cuáles son los lugares que me faltan por recorrer? -> $[P_{2}, P_{3}, P_{4}, P_{5}]$
        - ¿Cuál es el siguiente lugar que quiero visitar? -> $P_{2}$
        - ...
        - ¿Dónde estoy parado? -> $P_{4}$
        - ¿Cuáles son los lugares que me faltan por recorrer? -> $[P_{5}]$
        - ¿Cuál es el siguiente lugar que quiero visitar? -> $P_{5}$

        Así, lo que sigue a continuación es redactar esta estructura como un modelo
        de programación dinámica.

        ### **Modelo Matemático**
        **Conjuntos**

        - $i, j \in I$: Locaciones.

        - $t \in T$: Pasos o periodos. (también puede ser entendido como el numero de arcos que debo asignar, así T = N-2)


        **Parámetros**

        - $C_{ij}$: Costo de seleccionar el tramo desde i hasta j en el paso t.

        - $N$: Nodo que excluyo del problema y que luego condiciono para cerrar el ciclo.

        - $CNi_{i}$: Costo para unir el primer nodo con el nodo excluido.

        - $CNf_{j}$: costo para unir el ultimo nodo con el nodo excluido.


        **Variables**

        - $X_{ijt} \in (0, 1)$: Si escojo utilizar el tramo desde i hasta j para el paso t.

        - $Wstart_{i} \in (0, 1)$: variable que marca el inicio del recorrido.

        - $W_{it} \in (0, 1)$: Marca si la ciudad i fue visitada en t (en caso de que si, esta variable pasa a valer 1).

        - $pos_{it} \in (0, 1)$: Marca en que punto me encuentro parado en el periodo t.

        - $Wend_{i} \in (0, 1)$: Marca el ultimo lugar que visite en el recorrido.


        **Función Objetivo**

        $$
        \sum_{t} \sum_{i} \sum_{j} C_{ij}X_{ijt} + \sum_{i} CNi_{i}Wstart_{i} + \sum_{i}CNf_{i}Wend_{i}
        $$


        **Restricciones**

        1) Escoger 1 solo inicio.

        $$
        \sum_{i}Wstart_{i} = 1
        $$

        2) El primer trayecto parte desde el nodo start y solo va hacia 1 destino.

        $$
        \sum_{j}X_{ij1} = Wstart_{i} \quad \forall i
        $$

        3) Solo puede entrar 1 arco al nodo

        $$
        \sum_{t} \sum_{i} X_{ijt} = 1 \quad \forall j
        $$

        4) Solo puede salir 1 arco del nodo

        $$
        \sum_{j} \sum_{j} X_{ijt} = 1 \quad \forall i
        $$

        5) No puedo visitar el mismo nodo

        $$
        X_{iit} = 0 \quad \forall i, t
        $$


        6) No puedo visitar un lugar que ya visite

        $$
        X_{ijt} \leq (1 - W_{jt}) \quad \forall i, j, t
        $$

        7) Si visito un lugar i, este luego debe quedar como no disponible.

        $$
        W_{i1} = Wstart_{i} \quad \forall i
        $$

        $$
        W_{jt+1} = W_{jt} + \sum_{i} X_{ijt} \quad \forall j, t \in (1, .., T-1)
        $$


        8) Debo visitar todos los lugares.

        $$
        \sum_{i} W_{iT} = T
        $$

        $$
        \sum_{t} \sum_{i} \sum_{j} X_{ijt} = T
        $$

        9) No puedo venir de dos o mas lugares hacia un mismo nodo de manera simultanea

        $$
        \sum_{i} X_{ijt} \leq 1 \quad \forall t, j
        $$


        10) No puedo dirigirme hacia dos o mas lugares desde un mismo nodo de manera simultanea

        $$
        \sum_{j} X_{ijt} \leq 1 \quad \forall t, i
        $$

        11) Me aseguro de que en todos los tiempos escoja al menos 1 lugar para visitar

        $$
        \sum_{i} \sum_{j} X_{ijt} \geq 1 \quad \forall t
        $$

        12) Actualizo la variable de posición

        $$
        pos_{it} = \sum_{j} X_{ijt} \quad \forall t, i
        $$

        13) Restricción de flujo, asegura que el destino en t corresponda al comienzo en t+1

        $$
        \sum_{i} X_{ijt} = pos_{jt+1} \quad \forall j, t \in (1, ..., T-1)
        $$

        14) Rescato la ultima posición visitada

        $$
        Wend_{j} = \sum_{i} X_{ijT} \quad \forall j
        $$
                ''')

    def interactive_model(self):
        st.write('''
                ### Cargar Datos
                #### Matriz de costos ($C_{ij} = Cb*D_{ij}$)
                ''')

        st.sidebar.write('**Archivos descargables**')

        coord_file = pd.read_csv('https://raw.githubusercontent.com/raaraya1/Personal-Proyects/main/Proyectos/web_app/coordenadas.csv')
        st.sidebar.download_button(label='Locaciones.csv',
                                   data=coord_file.to_csv(index=False),
                                   file_name='coordenadas.csv',
                                   mime='text/csv')


        cost_matrix_csv = pd.read_csv('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/cost_matrix.csv')
        st.sidebar.download_button(label='matrix_cost.csv',
                                   data=cost_matrix_csv.to_csv(index=False),
                                   file_name='matrix_cost.csv',
                                   mime='text/csv')

        model_file = requests.get('https://raw.githubusercontent.com/raaraya1/transport-models-web/main/TSP_dinamic.py')
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
            filas = [i+1 for i in range(len(df)-1)]
            columnas = [i+1 for i in range(len(df)-1)]

            CN_i = {i:df[i][6] for i in filas}
            #st.write(str(CN_i))

            CN_f = {i:df[6][i] for i in filas}
            #st.write(str(CN_f))

            C_ij = {(i, j):df[j][i] for i in filas for j in filas}
            #st.write(str(C_ij))

            st.write('''
                    ### **Resultados**
                    ''')

            modelo = dinamic_TSP(matrix_cost=C_ij, initial_cost=CN_i, final_cost=CN_f)

            try:
                X_solve, W_start_solve, W, pos, W_end, obj_solve = modelo.solve()
                st.write('''
                        ##### Configuración optima de transporte
                ''')

                for i in X_solve:
                    if X_solve[i] > 0:
                        st.write(f'{i}')

                for i in W_start_solve:
                    if W_start_solve[i] > 0:
                        st.write(f'{i}')

                for i in W:
                    if W[i] > 0:
                        st.write(f'{i}')

                for i in pos:
                    if pos[i] > 0:
                        st.write(f'{i}')

                for i in W_end:
                    if W_end[i] > 0:
                        st.write(f'{i}')

                st.write(f'**Costo Total = ${obj_solve}**')
            except:
                sol = modelo.solve()
                st.write(str(sol))


            st.sidebar.write('**Visualizacion de resultados**')

            st.write(r'''
            ### Visualizacion de Resultados
            Para la visualizacion de los resultados es
            necesario crearnos una cuenta de usuario en
            https://openrouteservice.org y luego utilizar la clave
            generada (esta luego la debemos introducir en el panel a la izquierda)

            - **Puntos Verdes**: Donde se inicia el transporte
            - **Puntos Azules**: Donde se recogen niños
            - **Punto Rojo**: Destino final (para este caso el colegio)

            **Nota:** Es necesario establecer, tanto en la matriz de costo como
            en las coordenadas de los lugares, la ultima locacion como el colegio.
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

                #st.write(str(rutas))

                # Ultima locacion (colegio)
                last_term = len(posiciones)

                # rutas seleccionadas
                rutas_selec = {}

                # primera ruta
                for i in W_start_solve:
                    if W_start_solve[i] > 0:
                        llave = (last_term, int(i[8:-1]))
                        rutas_selec[1] = rutas[llave]

                #st.write(str(rutas_selec))

                # otras rutas
                for i in X_solve:
                    if X_solve[i] > 0:
                        lista = [int(j) for j in i[2:-1].split(',')]
                        llave = (lista[0], lista[1])
                        rutas_selec[lista[2] + 1] = rutas[llave]

                #st.write(str(rutas_selec))

                # ultima ruta
                for i in W_end:
                    if W_end[i] > 0:
                        llave = (int(i[6:-1]), last_term)
                        rutas_selec[last_term] = rutas[llave]


                #st.write(str(rutas_selec))

                rutas_especificas = {}
                for i in rutas_selec:
                    #st.write(str(i))
                    origen_lat = rutas_selec[i][0][0]
                    origen_lon =  rutas_selec[i][0][1]
                    destino_lat = rutas_selec[i][1][0]
                    destino_lon = rutas_selec[i][1][1]
                    rutas_especificas[f'ruta{i}'] = [[float(origen_lat), float(origen_lon)], [float(destino_lat), float(destino_lon)]]

                mapa = Mapas(clave, lugares)
                map = mapa.Mapa_con_rutas(rutas_especificas)
                for i in posiciones:
                    if i == last_term:
                        cord_fol = [posiciones[i][1], posiciones[i][0]]# para ordenar las coordenadas
                        folium.Marker(cord_fol, popup=f'P{i}', icon=folium.Icon(color="red")).add_to(map)
                    else:
                        cord_fol = [posiciones[i][1], posiciones[i][0]]# para ordenar las coordenadas
                        folium.Marker(cord_fol, popup=f'P{i}', icon=folium.Icon(color="blue")).add_to(map)

                folium_static(map)
