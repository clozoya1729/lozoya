import folium

BASEMAP = 'C:\\Users\\frano\PycharmProjects\PASS simplified\map overlay'
geo_path = r'data/antarctic_ice_edge.json'
topo_path = r'data/antarctic_ice_shelf_topo.json'
ice_map = folium.Map(location=[-59.1759, -11.6016],
                     tiles='Mapbox Bright', zoom_start=2)
ice_map.geo_json(geo_path=geo_path)
ice_map.geo_json(geo_path=topo_path, topojson='objects.antarctic_ice_shelf')
ice_map.create_map()
ice_map.save_map(BASEMAP)
