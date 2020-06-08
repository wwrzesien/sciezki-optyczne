# Model wykorzystujący regresję K-najbliższych sąsiadów

from baza_danych import BazaDanych
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
warnings.filterwarnings("ignore", message="Reloaded modules: <module_name>")


GRUPA_ZAKRES = [17.04, 24.06, 29.55, 35.53]
GRUPA_ZAKRES_2 = [19.24, 27.05, 34.01, 48]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class KNeighbours:
    def __init__(self):
        self.baza_trening = {}
        self.baza_trening_skal = {}
        self.baza_test = {}
        self.baza = {}
        
        
    def analiza(self):
        """Przeprowadz analize."""
        self.przygotuj_baze()
        self.preprocessing()
        #dane_blad = 
        self.sasiedzi()

        # Wykres straty
        #self.wykres(dane_blad['loss'], dane_blad['val_loss'], 'Wykres straty', 'Strata')

        # Wykres dokladnosci
        #self.wykres(dane_blad['accuracy'], dane_blad['val_accuracy'],
         #           'Wykres dokladnosci', 'Dokladnosc')    
        
        
        
        
        
    def sasiedzi(self):
        """Przeprowadz analize."""
      
        
        print("kNearest")
        # Fit regression model
        n_neighbors = 5
    
        weights ='distance'
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        model_dane=knn.fit(self.baza_trening['X'],self.baza_trening['y'] )
        y_ = knn.fit(self.baza_trening['X'],self.baza_trening['y'] ).predict(self.baza_test['X'] )
        #print(y_)
        
        accuracy=knn.score(self.baza_test['X'],self.baza_test['y'],sample_weight=None)
        
        #print('Sciezka, prawdziwa kategoria, przewidziana kategoria')
       # for p, x, y in zip(y_, self.baza_test['X'], self.baza_test['y']):
        #    print(x, y, p)
        
        #plt.subplot(2, 1, 1)
        #print("x: ")
        #print(self.baza_trening['X'])
        #print("y: ")
        #print(self.baza_trening['y'])
        #plt.scatter(self.baza_trening['X'], self.baza_trening['y'], color='darkorange', label='data')
        #plt.plot(self.baza_test['X'], y_, color='navy', label='prediction')
        #plt.axis('tight')
        #plt.legend()
        #plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
        #                                                        weights))

        
        print("Accuracy: ")
        print(accuracy)
        
        
        
        
        
        print("Extra Tress Regresor")
    
      
        etr = ensemble.ExtraTreesRegressor(n_estimators=10, max_features=10, random_state=0)
        model_dane=etr.fit(self.baza_trening['X'],self.baza_trening['y'] )
        y_ = etr.fit(self.baza_trening['X'],self.baza_trening['y'] ).predict(self.baza_test['X'] )
        #print(y_)
        
        accuracy=etr.score(self.baza_test['X'],self.baza_test['y'],sample_weight=None)
        
        print("Accuracy: ")
        print(accuracy)
        
        
        
        
        
        print("Linear Regression")
    
        
        lr = linear_model.LinearRegression()
        model_dane=lr.fit(self.baza_trening['X'],self.baza_trening['y'] )
        y_ = lr.fit(self.baza_trening['X'],self.baza_trening['y'] ).predict(self.baza_test['X'] )
        #print(y_)
        
        accuracy=lr.score(self.baza_test['X'],self.baza_test['y'],sample_weight=None)
        
        print("Accuracy: ")
        print(accuracy)
        
        
        
       
        print("Ridge regression")
      
    
       
        lr2 = linear_model.RidgeCV()
        model_dane=lr2.fit(self.baza_trening['X'],self.baza_trening['y'] )
        y_ = lr2.fit(self.baza_trening['X'],self.baza_trening['y'] ).predict(self.baza_test['X'] )
        #print(y_)
        
        accuracy=lr2.score(self.baza_test['X'],self.baza_test['y'],sample_weight=None)
        
        print("Accuracy: ")
        print(accuracy)
        
  
    
    
        ############################################################################
        

        
        

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

        skaler = MinMaxScaler(feature_range=(0, 1))
        self.baza_trening_skal['X'] = skaler.fit_transform(self.baza_trening['X'])

        logger.info('Liczebnosc grup zbioru treningowego: {}'
                    .format(self.liczebnosc_grup(self.baza_trening['y'])))
        logger.info('Liczebnosc grup zbioru testowego: {}'
                    .format(self.liczebnosc_grup(self.baza_test['y'])))



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