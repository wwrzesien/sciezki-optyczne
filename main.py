# Program 2 PSZT
# Temat: ML sciezki optyczne provisioning
# Autor: Marcin Gajewski, Wojciech Wrzesien

from ann import ANN
from Kneighbours import KNeighbours
import logging

logging.basicConfig(level=logging.INFO)

def main():

    sieci_neuronowe = ANN()
    sieci_neuronowe.analiza()

    K_neighbors = KNeighbours()
    K_neighbors.analiza()


if __name__ == "__main__":
    main()
