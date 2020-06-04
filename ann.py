# Model wykorzystujący sztuczne sieci neuronowe

from baza_danych import BazaDanych
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os


class ANN:
    def __init__(self):
        # self.baza_trening = {'X': np.array([]), 'y': np.array}  # {'X': [[], ...,[]], 'y':[]}
        # self.baza_test = {'X': np.array([]), 'y': np.array}  # {'X': [[], ...,[]], 'y':[]}
        self.baza_trening = {}
        self.baza_test = {}
        self.baza = {}

    def analiza(self):
        """Przeprowadz analize."""
        self.przygotuj_baze()
        print(self.baza_test)
        # print(self.baza)
        # hello =
        # print(self.baza_trening)

        model = Sequential([
            Dense(128, input_shape=(11,), activation='relu'),
            Dense(64, activation='relu'),
            # Dense(32, activation='relu'),
            Dense(5, activation='softmax')
        ])
        model.summary()
        model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.baza['X'], self.baza['y'], validation_split=0.2,
                  batch_size=20, epochs=100, shuffle=True, verbose=2)

        przewidywanie = model.predict(self.baza_test['X'], batch_size=10, verbose=0)

        zaokraglone_przewidywanie = model.predict_classes(
            self.baza_test['X'], batch_size=10, verbose=0)

        # for i in przewidywanie:
        #     print(i)

        print('Sciezka, prawdziwa kategoria, przewidziana kategoria')
        for p, x, y in zip(zaokraglone_przewidywanie, self.baza_test['X'], self.baza_test['y']):
            print(x, y, p)

    def przygotuj_baze(self):
        """Przygotowanie bazy testowej i treningowej."""
        baza = BazaDanych('pol.xml')
        baza.utworz_baze()
        x = []
        y = []
        # koszt = []

        # <0, 10) -> 0, <10, 20) -> 1, <20, 30) -> 2, <30, 40) -> 3, <40, 50) -> 4
        # for sciezka in baza.baza_treningowa:
        #     dane_x = [int(i) for i in sciezka['sciezka']]
        #     for i in range(len(sciezka['sciezka']), baza.najdluzsza_sciezka()):
        #         dane_x.append(-1)
        #     x.append(dane_x)
        #     y.append(int((sciezka['osnr']-0.001)/10))
        # self.baza_trening['X'] = np.array(x)
        # self.baza_trening['y'] = np.array(y)
        #
        # print(baza.baza_testowa)
        # print(baza.baza_treningowa)
        # for sciezka in baza.baza_testowa:
        #     dane_x = [int(i) for i in sciezka['sciezka']]
        #     for i in range(len(sciezka['sciezka']), baza.najdluzsza_sciezka()):
        #         dane_x.append(-1)
        #     x.append(dane_x.copy())
        #     y.append(int((sciezka['osnr']-0.001)/15))
        # self.baza_test['X'] = np.array(x)
        # self.baza_test['y'] = np.array(y)

        for sciezka in baza.sciezki:
            dane_x = [int(i) for i in sciezka['sciezka']]
            for i in range(len(sciezka['sciezka']), baza.najdluzsza_sciezka()):
                dane_x.append(-1)
            x.append(dane_x.copy())
            y.append(int((sciezka['osnr']-0.001)/10))
        self.baza['X'] = np.array(x)
        self.baza['y'] = np.array(y)

        self.baza_trening['X'], self.baza_test['X'], self.baza_trening['y'], self.baza_test['y'] = train_test_split(
            self.baza['X'], self.baza['y'], test_size=0.2)

        # self.baza_trening['X'] = tf.keras.utils.normalize(self.baza_trening['X'], axis=1)
        # self.baza_test['X'] = tf.keras.utils.normalize(self.baza_test['X'], axis=1)

        print(self.baza_trening['X'])
