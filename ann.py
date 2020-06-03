# Model wykorzystujący sztuczne sieci neuronowe

from baza_danych import BazaDanych
import tensorflow as tf


class ANN:
    def __init__(self):
        self.baza_trening = {'X': [], 'y': []}  # {'X': [[], ...,[]], 'y':[]}
        self.baza_test = {'X': [], 'y': []}  # {'X': [[], ...,[]], 'y':[]}

    def analiza(self):
        """Przeprowadz analize."""
        self.przygotuj_baze()

    def przygotuj_baze(self):
        """Przygotowanie bazy testowej i treningowej."""
        baza = BazaDanych('pol.xml')
        baza.utworz_baze()

        # <0, 10) -> 0
        # <10, 20) -> 1
        # <20, 30) -> 2
        # <30, 40) -> 3
        # <40, 50) -> 4
        for sciezka in baza.baza_treningowa:
            dane_x = []
            dane_x = sciezka['sciezka'].copy()
            for i in range(len(sciezka['sciezka']), baza.najdluzsza_sciezka()):
                dane_x.append(-1)
            self.baza_trening['X'].append(dane_x.copy())
            self.baza_trening['y'].append(int((sciezka['osnr']-0.001)/10))
