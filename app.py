# Importar biblioteca para la aplicacion web
import streamlit as st

# Importar modelos de transporte
from Colaboratory_Transport import Colaboratory_Transport
from Facility_Location import Facility_Location
from TSP_MTZ import TSP_MTZ

# Otras bibliotecas
import pandas as pd
import numpy as np

st.title('Modelos de Transporte')

st.write('''
## **Contexto**
Este problema surgio de la conversacion entre un grupo de amigos, y este consistia principalmente en organizar las idas y llegadas al colegio, debido a que, con el sistema actual, colapsaba el numero de estacionamientos en el recinto.

Asi, durante la conversacion, se originaron distintas estrategias de como abordar el problema, todas ellas planteando distintos enfoques de modelos de optimizacion.

1. **Modelo con sistema de turnos y apoyo entre vecinos**
2. **Modelo con buses de acercamiento en puntos especificos**
3. **Modelo con buses de acercamiento a domicilio**

''')

model_name = st.sidebar.selectbox('Seleccionar Modelo',
                                 ['Colaboratory_Transport', 'Dinamic_Colaboratory_Transport', 'Facility_Location', 'TSP_MTZ'])


if model_name == 'Colaboratory_Transport':
    st.write(r'''
    ## **Modelo con sistemas de apoyo entre vecinos**
    **Supuestos:**

    1. Todos los niños de las familias deben ser transportados al colegio.
    2. Todas las familias cuentan con un mismo vehiculo cuya capacidad no supera las 4 personas (sin incluir al conductor)
    3. El costo de la bencina es igual para todas las familias
    4. Solo puedo ayudar a 1 vecino a transportar a sus hijos.
    5. Si ayudo a un vecino, transportaré a todos sus hijos.
    6. Sin temporalidad


    **Conjuntos:**

     - $i \in I$: Locacion de las familias + colegio, donde $i=c$ es el colegio.

    **Parametros:**

     - $N_{i}$: Numero de niños, perteneciente a la familia i.
     - $Dc_{i}$: Distancia entre la locacion de la familia i al colegio.
     - $D_{ij}$: Distancia entre la locacion de la familia i y la locacion de la familia j.
     - $Cb$: Costo de la bencina por km.

    **Variables:**

    $X_{ij} \in (0, 1)$: Si se realiza un viaje desde la locacion i hasta la locacion j.

    **Funcion objetivo:**

    $$
    min \sum_{i}\sum_{j} CbD_{ij}X_{ij}
    $$

    **Restricciones:**

    1. Solo puedo ayudar a un vecino.

    $$
    \sum_{j \neq c} X_{ij} \leq 1 \quad \forall i \in I-(c)
    $$

    2. Solo puedo recibir ayuda de 1 vecino.

    $$
    \sum_{i \neq c} X_{ij} \leq 1 \quad \forall j \in I-(c)
    $$

    3. Si ayudo, no me voy directo al colegio.

    $$
    X_{ic} \leq (1 - X_{ij}) \quad \forall i, j \in I
    $$

    4. Ayudarse a uno mismo no cuenta.

    $$
    X_{ii} = 0 \quad \forall i \in I
    $$

    5. No devolverse en los recorridos.

    $$
    X_{ji} \leq (1 - X_{ij}) \quad \forall i, j \in I
    $$

    6. Asegurarse de enviar a los niños luego de recogerlos.

    $$
    X_{jc} \geq X_{ij} \quad \forall i, j \in I-(c)
    $$

    7. No puedo transportar mas niños que la capacidad del auto.

    $$
    (N_{i} + N_{j})X_{ij} \leq 4 \quad  \forall i, j \in I
    $$

    8. Debo transportar a todos los niños. (Niños que van directo + niños que son recogidos = total de niños).

    $$
    \sum_{i \neq c} (N_{i}X_{ic} + \sum_{j \neq c}N_{i}X_{ij}) = \sum_{i} N_{i}
    $$
    ''')

    st.write('''
            ### Cargar Datos
            #### Matriz de costos ($C_{ij} = Cb*D_{ij}$)
            ''')

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
        try:
            cap = int(cap)
        except:
            st.write('Ajustar capacidad de los autos')


    if cost_matrix is not None and type(cap) != np.str:
        N_i = {}
        n_i = []
        for i in range(n_locations-1):
            valor = st.sidebar.slider(f'Cantidad de niños del lugar {i+1}', 1, cap)
            N_i[i+1] = valor
            n_i.append([valor])

        df_ni = pd.DataFrame(n_i, columns=['N_i'], index=[i+1 for i in range(len(n_i))])
        st.dataframe(df_ni)

    if cost_matrix is not None and type(cap) != np.str:

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

        modelo = Colaboratory_Transport(matrix_cost=C_ij, N_i=N_i, cap=cap)

        X, obj = modelo.solve()
        st.write('''
                ##### Configuración optima de transporte
        ''')
        for i in X:
          if X[i] > 0:
            st.write(f'X{i}')
        st.write(f'**Costo Total = ${obj}**')



elif model_name == 'Dinamic_Colaboratory_Transport':
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


    st.write('''
            ### Cargar Datos
            #### Matriz de costos ($C_{ij} = Cb*D_{ij}$)
            ''')

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





elif model_name == 'Facility_Location':
    st.write(r'''
    ## **Modelo con buses de acercamiento en puntos especificos**

    **Supuestos:**

    1. Todos los niños de las familias deben ser transportados al colegio.
    2. Todas las familias cuentan con un mismo vehiculo cuya capacidad no supera las 4 personas (sin incluir al conductor)
    3. El costo de la bencina es el mismo entre auto y bus.
    4. Del paradero solo partira un bus (puede no estar lleno)
    5. No existe un costo asociado a la instalacion del paradero.
    6. Si decido instalar un paradero, si o si saldra un bus desde esa locacion.
    7. Todos las familias cuentan con un vehiculo con la capacidad de almacenamiento suficiente para enviar a sus hijos.

    **Conjuntos**
    - $i \in I$: Locacion de las familias + colegio, donde $i=c$ es el colegio.
    - $j \in J$: Posibles puntos de paradero.

    **Parametros**

     - $N_{i}$: Numero de niños, perteneciente a la familia i.
     - $Dc_{i}$: Distancia entre la locacion de la familia i al colegio.
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

    **Funcion Objetivo**

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

    3) No puedo enviar niños al paradero j si, sumando esta cantidad de niños que trae, como resultado se llegasé a sobrepasar la capacidad del bus.

    $$
    Y_{ij} \leq (2 - \frac{W_{j} + N_{i}}{CAPb}) \quad \forall i \in I, j \in J
    $$


    4) No puedo enviar los niños directamente al colegio y al paradero al mismo tiempo.

    $$
    Y_{ij} \leq (1 - Z_{i}) \quad \forall i \in I, j \in J
    $$

    5) No puedo enviar a los niños a dos o mas paraderos diferentes.

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


    8) (Opcional) Numero de paraderos maximo (Para este caso, tambien puede ser visto como el numero maximo de buses que dispongo)

    $$
    X_{j} \leq N_{bus} \quad \forall j \in J
    $$
    ''')

    st.write('''
            ## Cargar Datos
            ### Matriz de costos (hogares - paraderos)
            ''')

    cost_matrix = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos entre los hogares y los paraderos (csv - xlsx)')
    initial_route_cost = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos entre los hogares y el colegio (csv - xlsx)')
    last_route_cost = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos entre los paraderos y el colegio (csv - xlsx)')

    if cost_matrix is not None:
        df = pd.read_csv(cost_matrix, sep=';', header=None)
        n_locations = len(df)
        n_paraderos = len(df.to_numpy()[0])
        columns = [i+1 for i in range(3)]
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




elif model_name == 'TSP_MTZ':
    st.write(r'''
    ### Modelo con bus de acercamiento a domicilio
    **Supuestos**

    - El bus cuenta con la suficiente capacidad para trasnportar a todos los niños.

    **Conjuntos**

    - $i \in I$: Conjunto de localidades

    - $j \in J$: Conjunto de localidades (el mimso que $I$)

    **Parametros**

    - $C_{ij}$: Costo de ir desde el punto i al punto j.

    - $N$: Total de localidades

    **Variables**

    - $X_{ij} \in (0, 1)$: Si recorró desde el punto i al punto j.

    - $U_{i}$: Variable auxiliar en punto i

    **Funcion Objetivo**

    $$
    min \sum_{i} \sum_{j} C_{ij}X_{ij}
    $$

    **Restricciones**

    1) Desde un nodo solo sale 1 unico arco.

    $$
    \sum_{j} X_{ij} = 1 \quad \forall i \in I
    $$


    2) Hacia un nodo solo entra 1 unico arco.

    $$
    \sum_{i} X_{ij} = 1 \quad \forall j \in I
    $$

    3) Eliminacion de subciclos (Restricciones MTZ)

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

    st.write('''
            ### Cargar Datos
            #### Matriz de costos ($C_{ij} = Cb*D_{ij}$)
            ''')

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
