#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载用于预训练的国际象棋棋谱（PGN）"""
import urllib.request
import os

os.makedirs('data/pgn', exist_ok=True)

players = ['Capablanca', 'Alekhine', 'Botvinnik', 'Tal', 'Fischer', 'Karpov', 'Kasparov', 'Anand']

for p in players:
    url = f'https://www.pgnmentor.com/players/{p}.zip'
    dst = f'data/pgn/{p}.zip'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(dst, 'wb') as f:
            f.write(data)
        print(f'OK  {p}: {len(data)} bytes')
    except Exception as e:
        print(f'FAIL {p}: {e}')
