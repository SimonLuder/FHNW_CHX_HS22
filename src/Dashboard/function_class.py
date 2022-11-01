from asyncio.windows_events import NULL
import datetime
import requests
import pandas as pd
import openrouteservice as ors
import json

class prediction:

    # TODO
    def __init__(self):
        pass
        # self.parkings = {}
        # self.__timestamp = self.get_timestamp()

    # def get_timestamp(self):
    #     return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # def __call_newest_data(self, url):
    #     '''
    #     Requests the newest data from the data.bs.ch api and returns it as pandas dataframe.
    #     Args:
    #         url (str): url where the api is located
    #     '''
    #     try:
    #         r = requests.get(url)
    #         df = pd.DataFrame(r.json())
    #         df = df.dropna(axis="rows")
    #         return df['Anteil', 'id2']
    #     except requests.exceptions.RequestException as e:  # This is the correct syntax
    #         raise (f"An exception occured: {e}")

    def get_free_parkings(self):
        return ['aeschen', 'anfos', 'badbahnhof', 'bahnhofsued', 'centralbahnparking', 'city', 'clarahuus', 'elisabethen', 'europe', 'kunstmuseum', 'messe', 'postbasel', 'rebgasse', 'steinen', 'storchen']


class distance:
    def __init__(self, location: list, parkings_with_free_spots: list):
        self.all_parkings = {'aeschen': [7.5943046, 47.5504299],
                            'anfos': [7.593512, 47.5515968],
                            'badbahnhof': [7.6089067, 47.5651794],
                            'bahnhofsued': [7.5884556, 47.5458851],
                            'centralbahnparking': [7.5922975, 47.547299],
                            'city': [7.5824076, 47.561101],
                            'clarahuus': [7.5917937, 47.5622725],
                            'elisabethen': [7.5874932, 47.5506254],
                            'europe': [7.5967098, 47.5630411],
                            'kunstmuseum': [7.5927014, 47.5545146],
                            'messe': [7.602175, 47.563241],
                            'postbasel': [7.5929374, 47.5468617],
                            'rebgasse': [7.594263, 47.5607142],
                            'steinen': [7.5858936, 47.5524554],
                            'storchen': [7.58658, 47.5592347]}
        self.parkings = {k: v for k, v in self.all_parkings.items() if k in parkings_with_free_spots}
        self.location = location
        self.distance = {}
        self.__client = '5b3ce3597851110001cf624861b0095d93d34e199ad718c53eb21e01'

    def __call_distance(self, parking: str):
        '''
        Calculates the distance between two points.
        Args:
            parking (str): name of the parking
        '''
        body = {"locations":[self.location, *self.parkings.values()]}
        headers = {'Authorization': self.__client}
        
        # get distance in seconds
        call = requests.post('https://api.openrouteservice.org/v2/matrix/driving-car', json=body, headers=headers)
        output = call.json()
        return {j: i for i, j in zip(output['durations'][0][1:], self.parkings.keys())}

    
    def get_distances(self):
        '''
        Calculates the distance for all parkings.
        '''
        for i in self.parkings.keys():
            self.distance[i] = self.__call_distance(i)
            break

    def get_closest_parking(self):
        '''
        Returns the closest parking.
        '''
        self.get_distances()
        return min(self.distance, key=self.distance.get)


class route:
    def __init__(self, location: list):
        self.__client = ors.Client(key='5b3ce3597851110001cf624861b0095d93d34e199ad718c53eb21e01')
        self.all_parkings = {'aeschen': [7.5943046, 47.5504299],
                            'anfos': [7.593512, 47.5515968],
                            'badbahnhof': [7.6089067, 47.5651794],
                            'bahnhofsued': [7.5884556, 47.5458851],
                            'centralbahnparking': [7.5922975, 47.547299],
                            'city': [7.5824076, 47.561101],
                            'clarahuus': [7.5917937, 47.5622725],
                            'elisabethen': [7.5874932, 47.5506254],
                            'europe': [7.5967098, 47.5630411],
                            'kunstmuseum': [7.5927014, 47.5545146],
                            'messe': [7.602175, 47.563241],
                            'postbasel': [7.5929374, 47.5468617],
                            'rebgasse': [7.594263, 47.5607142],
                            'steinen': [7.5858936, 47.5524554],
                            'storchen': [7.58658, 47.5592347]}
        self.location = location

    def get_route(self, parking: str):
        '''
        Calculates the route between two points.
        Args:
            parking (str): name of the parking
        '''
        route = self.__client.directions(
            coordinates=[self.location, self.all_parkings[parking]],
            profile='driving-car',
            format='geojson')
        return route['features'][0]['geometry']['coordinates']


class parking_route:
    def __init__(self, location_: list):
        self.location_ = location_
    
    def get_free_parkings_prediction(self):
        '''
        Returns all parkings with free spots. (list with names)
        '''
        valid_parkings = prediction().get_free_parkings()
        return valid_parkings

    def get_closest_parking_calculation(self):
        '''
        Returns the closest parking.
        '''
        valid_parkings = self.get_free_parkings_prediction()
        closest_parking = distance(self.location_, valid_parkings).get_closest_parking()
        return closest_parking

    def get_route_to(self):
        '''
        Returns the route to the closest parking.
        '''
        close = self.get_closest_parking_calculation()
        return route(self.location_).get_route(close)