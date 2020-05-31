"""Generowanie z pliku xml treningowej i testowej bazy danych."""

import xml.etree.ElementTree as ET
import math

BAZA_TRENING = 0.1


class BazaDanych():
    def __init__(self, nazwa_pliku):
        self.nazwa_pliku = nazwa_pliku
        self.tree = ET.parse(self.nazwa_pliku)
        self.root = self.tree.getroot()
        self.miasta = []
        self.polaczenia = []
        self.sciezki = []
        self.baza_testowa = []
        self.baza_treningowa = []

    def utworz_baze(self):
        """Generowanie bazy danych."""
        self.baza_miast()
        self.baza_polaczen()  # Nie jest potrzebna
        self.baza_sciezek()
        self.generuj_baze()

    def baza_miast(self):
        """Utworz liste miast.
        [{
            'id': '0',
            'miasto': 'Gdansk',
            'x': 18.6,
            'y': 54.2
        },
        ...]"""
        for count, node in enumerate(self.root[0][0]):
            wiersz = {
                'id': str(count),
                'miasto': node.attrib['id'],
                'x': float(node[0][0].text),
                'y': float(node[0][1].text)
            }
            self.miasta.append(wiersz.copy())

    def baza_polaczen(self):
        """Utworz liste polaczen oraz policz odleglosc na podstawie
        wspolrzednych x i y.
        [{
            'nazwa': 'Link_0_10',
            'start': '0',
            'meta': '10',
            'odleglosc': 3.12
        },
        ...]"""
        for node in self.root[0][1]:
            wiersz = {
                'nazwa': node.attrib['id'],
                'start': node.attrib['id'][node.attrib['id'].find('_') + 1:
                                           node.attrib['id'].find('_')+2],
                'meta': node.attrib['id'][node.attrib['id'].find('_')+3:]
            }
            odleglosc = self.oblicz_odleglosc(wiersz['start'], wiersz['meta'])
            wiersz['odleglosc'] = odleglosc

            self.polaczenia.append(wiersz.copy())

    def oblicz_odleglosc(self, start, meta):
        """Oblicz odleglosc miedzy miastami."""
        for miasto in self.miasta:
            if miasto['id'] == start:
                miasto_A = miasto
            elif miasto['id'] == meta:
                miasto_B = miasto

        try:
            odleglosc = math.sqrt(pow(miasto_A['x'] - miasto_B['x'], 2) +
                                  pow(miasto_A['y'] - miasto_B['y'], 2))
        except KeyError as err_k:
            print(err_k)
            odleglosc = None

        return round(odleglosc, 2)

    def baza_sciezek(self):
        """Utworz liste sciezek, oblicz jej dlugosc oraz koszt OSNR.
        [{
            'start': 'Warsaw',
            'meta': 'Wroclaw',
            'id': '461',
            'sciezka': ['10', '4', '8', '5', '0', '2', '9', '7', '11'],
            'odleglosc': 20.66,
            'osnr': 9.23
        },
        ...]"""
        count = 0
        # demand
        for node in self.root[1]:
            wiersz = {
                'start': node[0].text,
                'meta': node[1].text
            }
            # admissiblePat
            for path in node[3]:
                odleglosc = 0
                lista_miast = [self.id_miasta(wiersz['start'])]
                # linkId
                for link in path:
                    start = link.text[link.text.find('_') + 1:link.text.find('_')+2]
                    meta = link.text[link.text.find('_')+3:]
                    odleglosc += self.oblicz_odleglosc(start, meta)

                    if start not in lista_miast:
                        lista_miast.append(start)
                    if meta not in lista_miast:
                        lista_miast.append(meta)

                wiersz['id'] = str(count)
                wiersz['sciezka'] = lista_miast
                wiersz['odleglosc'] = round(odleglosc, 2)
                count += 1
                self.sciezki.append(wiersz.copy())

        max = self.odleglosc_max()
        min = self.odleglosc_min()

        for sciezka in self.sciezki:
            sciezka['osnr'] = self.koszt_osnr(sciezka['odleglosc'], max, min)

    def id_miasta(self, nazwa):
        """Zwroc id miasta."""
        for miasto in self.miasta:
            if miasto['miasto'] == nazwa:
                return miasto['id']
        return None

    def generuj_baze(self):
        """Podziel baze na testowa (90% bazy) i treningowa (10% bazy)."""
        count = 0
        for sciezka in self.sciezki:
            if count % (int(1/BAZA_TRENING)) == 0:
                self.baza_treningowa.append(sciezka.copy())
            else:
                self.baza_testowa.append(sciezka.copy())
            count += 1

    def koszt_osnr(self, odleglosc, max, min):
        """Oblicz OSNR, przedzial 0-50, im dluzsza sciezka tym mniejszy OSNR."""
        return round(abs(((odleglosc - min) * 50) / (max - min) - 50), 2)

    def odleglosc_min(self):
        """Zwroc najkrotsza odleglosc."""
        min = 999999
        for sciezka in self.sciezki:
            if sciezka['odleglosc'] < min:
                min = sciezka['odleglosc']
        return min

    def odleglosc_max(self):
        """Zwroc najdluzsza odleglosc."""
        max = -999999
        for sciezka in self.sciezki:
            if sciezka['odleglosc'] > max:
                max = sciezka['odleglosc']
        return max

    def najdluzsza_sciezka(self):
        """Zwraca rozmiar najdluzszej sciezki."""
        rozmiar = 0
        for sciezka in self.sciezki:
            if len(sciezka['sciezka']) > rozmiar:
                rozmiar = len(sciezka['sciezka'])
        return rozmiar
