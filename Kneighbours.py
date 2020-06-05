# Model wykorzystujący regresję K-najbliższych sąsiadów

from baza_danych import BazaDanych
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore", message="Reloaded modules: <module_name>")

class KNeighbours:
    def __init__(self):
        self.baza_trening = {}
        self.baza_test = {}
        self.baza = {}
        
        
    def analiza(self):
        """Przeprowadz analize."""
        self.przygotuj_baze()
        X=self.baza_trening['X'][100:]     
        y=self.baza_trening['y']
        Xtest=self.baza_test['X'][1:100]
        print
        print(Xtest)
        print(X)
        
    #     # Fit estimators
    #     ESTIMATORS = {
    #         "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
    #                                    random_state=0),
    #         "K-nn": KNeighborsRegressor(),
    #         "Linear regression": LinearRegression(),
    #         "Ridge": RidgeCV(),
    #         }

    #     y_test_predict = dict()
    #     for name, estimator,i in ESTIMATORS.items():
    #             estimator.fit(self.baza_trening['X'], self.baza_trening['y'])
    #             y_test_predict[name] = estimator.predict(self.baza_test['X'])
                
    #             plt.subplot(2, 1, i+1)
    #             plt.scatter(X, y, color='darkorange', label='data')
    #             plt.plot(T, y_, color='navy', label='prediction')
    #             plt.axis('tight')
    #             plt.legend()
    #             plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
    #                                                             weights))

    # plt.tight_layout()
    # plt.show()
        
    
        #############################################################################
        

        # Fit regression model
        n_neighbors = 5
    
        for i, weights in enumerate(['uniform', 'distance']):
            knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            y_ = knn.fit(X,y ).predict( Xtest)
            
            plt.subplot(2, 1, i + 1)
            plt.scatter(X, y, color='darkorange', label='data')
            plt.plot(X, y_, color='navy', label='prediction')
            plt.axis('tight')
            plt.legend()
            plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                    weights))
            
        plt.tight_layout()
        plt.show()
            
        

    def przygotuj_baze(self):
            """Przygotowanie bazy testowej i treningowej."""
            baza = BazaDanych('pol.xml')
            baza.utworz_baze()
            x = []
            y = []

            for sciezka in baza.sciezki:
                dane_x = [int(i) for i in sciezka['sciezka']]
                for i in range(len(sciezka['sciezka']), baza.najdluzsza_sciezka()):
                    dane_x.append(-1)
                    x.append(dane_x.copy())
                    y.append(int((sciezka['osnr']-0.001)/10))
        
            self.baza['X'] = np.array(x)
            self.baza['y'] = np.array(y)

            self.baza_trening['X'], self.baza_test['X'], self.baza_trening['y'], self.baza_test['y'] = model_selection.train_test_split(
            self.baza['X'], self.baza['y'], test_size=0.2)


            print(self.baza_trening['X'])