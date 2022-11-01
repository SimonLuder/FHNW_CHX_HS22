from function_class import distance, route, parking_route

# dist = distance([7.337910, 47.175010], ['aeschen', 'anfos', 'badbahnhof', 'bahnhofsued', 'centralbahnparking', 'city', 'clarahuus', 'elisabethen', 'europe', 'kunstmuseum', 'messe', 'postbasel', 'rebgasse', 'steinen', 'storchen'])

# print(dist.get_closest_parking())

# rt = route([7.337910, 47.175010])
# rtf = rt.get_route('aeschen')

# print(rtf)

pr = parking_route([7.337910, 47.175010]).get_route_to()
for i in pr:
    if type(i[0]) == float and type(i[1]) == float:
        pass
    else:
        print('Nooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')