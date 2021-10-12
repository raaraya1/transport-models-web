import openrouteservice
import folium
import numpy as np

class Mapas():
  def __init__(self, key, lugares):
    self.key = key
    self.client = openrouteservice.Client(key=key)
    self.matrix_de_distancias = []
    self.lugares = lugares


  def matriz_de_distancias(self, unidad='km'):
    matrix_de_distancias = self.client.distance_matrix(locations=self.lugares, metrics = ("duration", "distance"), units = unidad)
    return matrix_de_distancias['distances']

  def Mapa_con_lugares(self):
    coordenadas_folium = [[i[1], i[0]] for i in self.lugares]
    punto_central = np.mean(coordenadas_folium, axis=0)
    map = folium.Map(location=punto_central, zoom_start=15)

    for indx, coord in enumerate(coordenadas_folium):
      folium.Marker(coord, popup=f'P{indx+1}', icon=folium.Icon(color="blue")).add_to(map)

    return map

  def Mapa_con_rutas(self, rutas_especificas=0):
    map = self.Mapa_con_lugares()

    if rutas_especificas == 0:
      rutas = []
      for i in self.lugares:
        for j in self.lugares:
          if i != j:
            input = [i, j]
            rutas.append(self.client.directions(coordinates=input, profile='driving-car', format='geojson'))

      dic_routes = {}
      for i in range(len(rutas)):
        dic_routes[f'ruta_{i+1}'] = rutas[i]

      for i in dic_routes:
        folium.GeoJson(dic_routes[i], name=i).add_to(map)

      folium.LayerControl().add_to(map)

    elif rutas_especificas != 0:
      dic_routes = {}
      for i in rutas_especificas:
        origin = rutas_especificas[i][0]
        origin_folium = [origin[1], origin[0]]
        folium.Marker(origin_folium, icon=folium.Icon(color="green")).add_to(map)

        final = rutas_especificas[i][1]
        final_folium = [final[1], final[0]]
        folium.Marker(final_folium, icon=folium.Icon(color="red")).add_to(map)

        input = [origin, final]
        ruta = self.client.directions(coordinates=input, profile='driving-car', format='geojson')
        dic_routes[i] = ruta

      for i in dic_routes:
        folium.GeoJson(dic_routes[i], name=i).add_to(map)

      folium.LayerControl().add_to(map)

    return map
