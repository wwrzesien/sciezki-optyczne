# Model wykorzystujący regresję K-najbliższych sąsiadów

from baza_danych import BazaDanych
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
import warnings
import logging
warnings.filterwarnings("ignore", message="Reloaded modules: <module_name>")

GRUPA_ZAKRES = [17.04, 24.06, 29.55, 35.53]

logger = logging.getLogger("KNeighbours")
logging.basicConfig(level=logging.INFO)

class KNeighbours:
    def __init__(self):
        self.baza_trening = {}
        self.baza_test = {}
        self.baza = {}

    def analiza(self):
        """Przeprowadz analize."""
        self.przygotuj_baze()
        self.preprocessing()
        self.sasiedzi()

    def sasiedzi(self):
        """Przeprowadz analize."""
        min_liczba_probek=25

        ########### K-neares Neighbours Regressor
        print("kNearest")
        n_neighbors = 5
        weights ='distance'
        accuracyKNN=[]
        for i in range(min_liczba_probek,len(self.baza_trening['X'])):
            knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            model_dane=knn.fit(self.baza_trening['X'][:i],self.baza_trening['y'][:i] )
            y_ = knn.fit(self.baza_trening['X'][:i],self.baza_trening['y'][ : i ] ).predict(self.baza_test['X'][:i] )
            accuracy=knn.score(self.baza_test['X'][:i],self.baza_test['y'][:i],sample_weight=None)
            accuracyKNN.append(accuracy)
        print("Accuracy KNN: ")
        print(accuracy)

        ########### Extra Trees Regressor
        print("Extra Tress Regresor")
        accuracyETR=[]
        for i in range(min_liczba_probek,len(self.baza_trening['X'])):
            etr = ensemble.ExtraTreesRegressor(n_estimators=10, max_features=10, random_state=0)
            model_dane=etr.fit(self.baza_trening['X'][:i],self.baza_trening['y'][:i] )
            y_ = etr.fit(self.baza_trening['X'][:i],self.baza_trening['y'][:i] ).predict(self.baza_test['X'][:i] )
            accuracy=etr.score(self.baza_test['X'][:i],self.baza_test['y'][:i],sample_weight=None)
            accuracyETR.append(accuracy)
        print("Accuracy ETR: ")
        print(accuracy)

        ########### Linear regression
        print("Linear Regression")
        accuracyLR=[]
        for i in range(min_liczba_probek,len(self.baza_trening['X'])):
            lr = linear_model.LinearRegression()
            model_dane=lr.fit(self.baza_trening['X'][:i],self.baza_trening['y'][:i] )
            y_ = lr.fit(self.baza_trening['X'][:i],self.baza_trening['y'][:i] ).predict(self.baza_test['X'][:i] )
            accuracy=lr.score(self.baza_test['X'][:i],self.baza_test['y'][:i],sample_weight=None)
            accuracyLR.append(accuracy)
        print("Accuracy: ")
        print(accuracy)

        ########### Ridge regression
        print("Ridge regression")
        accuracyRR=[]
        for i in range(min_liczba_probek,len(self.baza_trening['X'])):
            lr2 = linear_model.RidgeCV()
            model_dane=lr2.fit(self.baza_trening['X'][:i],self.baza_trening['y'][:i] )
            y_ = lr2.fit(self.baza_trening['X'][:i],self.baza_trening['y'][:i] ).predict(self.baza_test['X'][:i] )
            accuracy=lr2.score(self.baza_test['X'][:i],self.baza_test['y'][:i],sample_weight=None)
            accuracyRR.append(accuracy)
        print("Accuracy: ")
        print(accuracy)

        """wykresy"""
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(range(min_liczba_probek,len(self.baza_trening['X'])), accuracyKNN)
        axs[0, 0].set_title('K-Nearest Neighbours Regression')
        axs[0, 1].plot(range(min_liczba_probek,len(self.baza_trening['X'])), accuracyETR, 'tab:orange')
        axs[0, 1].set_title('Extra Trees Regression')
        axs[1, 0].plot(range(min_liczba_probek,len(self.baza_trening['X'])), accuracyLR, 'tab:green')
        axs[1, 0].set_title('Linear Regression')
        axs[1, 1].plot(range(min_liczba_probek,len(self.baza_trening['X'])), accuracyRR, 'tab:red')
        axs[1, 1].set_title('Ridge Regression')

        for ax in axs.flat:
            ax.set(xlabel='size of training set', ylabel='accuracy')

        for ax in fig.get_axes():
            ax.label_outer()

        plt.show()

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
        self.baza_trening['X'], self.baza_test['X'], self.baza_trening['y'], self.baza_test['y'] = model_selection.train_test_split(
            self.baza['X'],
            self.baza['y'],
            test_size=0.1
        )

        logger.info('Liczebnosc grup zbioru treningowego: {}'
                    .format(self.liczebnosc_grup(self.baza_trening['y'])))
        logger.info('Liczebnosc grup zbioru testowego: {}'
                    .format(self.liczebnosc_grup(self.baza_test['y'])))

    def liczebnosc_grup(self, zbior_y):
        """Policz liczebnosc poszczegolnych grup."""
        liczebnosc_grupy = [0, 0, 0, 0, 0]
        for y in zbior_y:
            liczebnosc_grupy[y] += 1
        return liczebnosc_grupy
