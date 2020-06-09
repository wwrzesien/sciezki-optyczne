"""Generowanie bazy danych z pliku xml."""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import logging
import math
import random

logger = logging.getLogger('BazaDanych')


class BazaDanych():
    def __init__(self, nazwa_pliku):
        self.nazwa_pliku = nazwa_pliku
        self.tree = ET.parse(self.nazwa_pliku)
        self.root = self.tree.getroot()
        self.miasta = []
        self.sciezki = []

    def utworz_baze(self):
        """Generowanie bazy danych."""
        self.baza_miast()
        self.baza_sciezek()
        self.wyswietl()
        random.shuffle(self.sciezki)

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
            wiersz_2 = {
                'start': node[1].text,
                'meta': node[0].text
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

                # Przypadek np. Gdansk -> Warszawa
                wiersz['id'] = str(count)
                wiersz['sciezka'] = lista_miast
                wiersz['odleglosc'] = round(odleglosc, 2)
                count += 1
                self.sciezki.append(wiersz.copy())

                # Przypadek Warszawa -> Gdansk
                wiersz_2['id'] = str(count)
                wiersz_2['sciezka'] = lista_miast[::-1]
                wiersz_2['odleglosc'] = round(odleglosc, 2)
                count += 1
                self.sciezki.append(wiersz_2.copy())

        max = self.odleglosc_max()
        min = self.odleglosc_min()

        for sciezka in self.sciezki:
            sciezka['osnr'] = self.koszt_osnr(sciezka['odleglosc'], max, min)

        logger.debug('Rozmair zbioru: {}'.format(len(self.sciezki)))

    def id_miasta(self, nazwa):
        """Zwroc id miasta."""
        for miasto in self.miasta:
            if miasto['miasto'] == nazwa:
                return miasto['id']
        return None

    def wyswietl(self):
        self.sciezki.sort(key=lambda i: i['osnr'])
        y = [i['osnr'] for i in self.sciezki]
        x = [i for i in range(0, len(self.sciezki))]

        rozmiar_grupy = int(len(self.sciezki)/5)

        logger.debug("Rozmiar bazy: {r}, jednej grupy {g}".format(
            r=len(self.sciezki), g=rozmiar_grupy))
        logger.debug("Grupa I: {z}".format(z=self.sciezki[rozmiar_grupy]['osnr']))
        logger.debug("Grupa II: {z}".format(z=self.sciezki[2*rozmiar_grupy]['osnr']))
        logger.debug("Grupa III: {z}".format(z=self.sciezki[3*rozmiar_grupy]['osnr']))
        logger.debug("Grupa IV: {z}".format(z=self.sciezki[4*rozmiar_grupy]['osnr']))


    def koszt_osnr(self, odleglosc, max, min):
        """Oblicz OSNR, przedzial 0-50, im dluzsza sciezka tym mniejszy OSNR."""
        return round(abs(((odleglosc - min) * (50)) / (max - min) - 48), 2)

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
