# Program 2 PSZT
# Temat: ML sciezki optyczne provisioning
# Autor: Marcin Gajewski, Wojciech Wrzesien

from baza_danych import BazaDanych


def main():

    baza = BazaDanych('pol.xml')
    baza.utworz_baze()


if __name__ == "__main__":
    main()
