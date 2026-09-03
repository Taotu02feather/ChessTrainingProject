#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载用于预训练的国际象棋棋谱（PGN）——扩充版，覆盖更多世界冠军和顶级大师"""
import urllib.request
import os

os.makedirs('data/pgn', exist_ok=True)

# 世界冠军 + 顶级大师（按 pgnmentor 的文件名）
players = [
    # 已下载的 8 位
    'Capablanca', 'Alekhine', 'Botvinnik', 'Tal', 'Fischer', 'Karpov', 'Kasparov', 'Anand',
    # 世界冠军
    'Carlsen', 'Kramnik', 'Topalov', 'Petrosian', 'Spassky', 'Smyslov', 'Euwe', 'Steinitz', 'Lasker',
    # 顶级大师
    'Caruana', 'Aronian', 'Nakamura', 'Ding', 'Keres', 'Korchnoi', 'Rubinstein', 'Morphy',
    'Bronstein', 'Keres', 'Gelfand', 'Ivanchuk', 'Grischuk', 'Mamedyarov', 'So', 'Nepomniachtchi',
]

# 去重，保持顺序
seen = set()
players = [p for p in players if not (p in seen or seen.add(p))]

success = 0
failed = 0

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
        success += 1
    except Exception as e:
        print(f'FAIL {p}: {e}')
        failed += 1

print(f'\n下载完成：成功 {success} 个，失败 {failed} 个')

