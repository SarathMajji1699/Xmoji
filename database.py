# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:27:20 2020

@author: sarath
"""


import sqlite3

conn = sqlite3.connect('database.sqlite')
cur = conn.cursor()

cur.execute('CREATE TABLE Logins (username TEXT, password TEXT)')
cur.execute('INSERT INTO Logins (username, password) VALUES (?, ?)',  ('Sarath', '12345'))


cur.execute('CREATE TABLE Register (username TEXT, password TEXT, email TEXT, mobile INTEGER)')


cur.execute('CREATE TABLE Admin (username TEXT, password TEXT)')
cur.execute('INSERT INTO Admin (username, password) VALUES (?, ?)',  ('Sarath', 'password'))

conn.commit()
conn.close()