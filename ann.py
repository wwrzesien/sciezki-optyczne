# Model wykorzystujący sztuczne sieci neuronowe

from baza_danych import BazaDanych
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import logging

GRUPA_ZAKRES = [17.04, 24.06, 29.55, 35.53]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ANN:
    def __init__(self):
        self.baza_trening = {}
        self.baza_trening_skal = {}
        self.baza_test = {}
        self.baza = {}

    def analiza(self):
        """Przeprowadz analize."""
        self.przygotuj_baze()
        self.preprocessing()
        dane_blad = self.siec()

        # Wykres straty
        self.wykres(dane_blad['loss'], dane_blad['val_loss'], 'Wykres straty', 'Strata')

        # Wykres dokladnosci
        self.wykres(dane_blad['accuracy'], dane_blad['val_accuracy'],
                    'Wykres dokladnosci', 'Dokladnosc')

    def siec(self):
        """Utworz siec."""
        # Utworz siec
        model = Sequential([
            Dense(128, input_shape=(11,), activation='relu'),
            # Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.4),
            # Dense(32, activation='relu'),

            Dense(32, activation='relu'),
            # Dropout(0.1),
            Dense(5, activation='softmax')
        ])

        # es_callback = EarlyStopping(monitor='val_loss', patience=5)

        # Wyswietl podsumowanie sieci
        model.summary()

        # Zdefiniuj optymalizator, funkcje strat itp
        model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Zdefiniuj dane
        model_dane = model.fit(
            self.baza_trening['X'],
            self.baza_trening['y'],
            validation_data=(self.baza_test['X'], self.baza_test['y']),
            batch_size=10,
            epochs=300,
            shuffle=True,
            verbose=2
            # callbacks=[es_callback]
        )

        # Przeprowadz klasyfikacje na nowych danych
        klasyfik = model.predict_classes(self.baza_test['X'], batch_size=10, verbose=0)

        print('Sciezka, prawdziwa kategoria, przewidziana kategoria')
        for p, x, y in zip(klasyfik, self.baza_test['X'], self.baza_test['y']):
            print(x, y, p)

        return model_dane.history

    def przygotuj_baze(self):
        """Przygotowanie bazy testowej i treningowej."""
        baza = BazaDanych('pol.xml')
        baza.utworz_baze()
        x = []
        y = []
        osnr = []

        # Przypisz sciezkom grupe
        for sciezka in baza.sciezki:
            dane_x = [int(i) for i in sciezka['sciezka']]
            for i in range(len(sciezka['sciezka']), baza.najdluzsza_sciezka()):
                dane_x.append(-1)
            x.append(dane_x.copy())
            osnr.append(sciezka['osnr'])
            if sciezka['osnr'] <= GRUPA_ZAKRES[0]:
                y.append(0)
            elif sciezka['osnr'] <= GRUPA_ZAKRES[1]:
                y.append(1)
            elif sciezka['osnr'] <= GRUPA_ZAKRES[2]:
                y.append(2)
            elif sciezka['osnr'] <= GRUPA_ZAKRES[3]:
                y.append(3)
            else:
                y.append(4)

        self.baza['X'] = np.array(x)
        self.baza['y'] = np.array(y)
        self.baza['osnr'] = np.array(osnr)

    def preprocessing(self):
        """Podzielenie bazy na treningowa i testowa, preprocessing"""
        self.baza_trening['X'], self.baza_test['X'], self.baza_trening['y'], self.baza_test['y'] = train_test_split(
            self.baza['X'],
            self.baza['y'],
            test_size=0.1
        )

        skaler = MinMaxScaler(feature_range=(0, 1))
        self.baza_trening_skal['X'] = skaler.fit_transform(self.baza_trening['X'])

        logger.info('Liczebnosc grup zbioru treningowego: {}'
                    .format(self.liczebnosc_grup(self.baza_trening['y'])))
        logger.info('Liczebnosc grup zbioru testowego: {}'
                    .format(self.liczebnosc_grup(self.baza_test['y'])))

        # self.baza_trening['X'] = tf.keras.utils.normalize(self.baza_trening['X'], axis=1)
        # self.baza_test['X'] = tf.keras.utils.normalize(self.baza_test['X'], axis=1)

    def wykres(self, dane_tren, dane_test, tytul, y_etyk):
        """Narysuj wykres."""
        plt.plot(dane_tren)
        plt.plot(dane_test)
        plt.title(tytul)
        plt.ylabel(y_etyk)
        plt.xlabel("Epoch")
        plt.legend(['Trening', 'Test'])
        plt.savefig(y_etyk + '.png')
        plt.show()

    def liczebnosc_grup(self, zbior_y):
        """Policz liczebnosc poszczegolnych grup."""
        liczebnosc_grupy = [0, 0, 0, 0, 0]
        for y in zbior_y:
            liczebnosc_grupy[y] += 1
        return liczebnosc_grupy
