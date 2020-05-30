"""Generowanie z pliku xml treningowej i testowej bazy danych."""

import xml.etree.ElementTree as ET


class BazaDanych():
    def __init__(self, nazwa_pliku):
        self.nazwa_pliku = nazwa_pliku
        self.tree = ET.parse(self.nazwa_pliku)
        self.root = self.tree.getroot()
        self.miasta = []

    def utworz_baze(self):
        """Generowanie bazy danych."""
        self.baza_miast()
        self.baza_polaczen()
        # for node in self.root[0][0]:
        #     print(node.attrib)
        #     print(node.attrib['id'])
        # print(self.root[0][0].tag)

    def baza_miast(self):
        """Utworz slownik miast."""
        for count, node in enumerate(self.root[0][0]):
            wezel = {
                'id': count,
                'miasto': node.attrib['id'],
                'x': node[0][0].text,
                'y': node[0][1].text
            }
            self.miasta.append(wezel)

    def baza_polaczen(self):
        """Utworz slownik polaczen oraz policz odleglosc na podstawie
        wspolrzednych x i y."""
