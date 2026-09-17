# -*- coding: utf-8 -*-
"""Подключить полные клетки к приложению: преобразовать в формат загрузчика
(ключ atoms, atoms_count, длительность_сек) и записать на штатные пути с бэкапом."""
import os, json, shutil
КОРЕНЬ=os.path.dirname(os.path.abspath(__file__))
КЛ=os.path.join(КОРЕНЬ,"данные","клеточки")
ПОЛН=os.path.join(КЛ,"клеточки_полные")
БЭК=os.path.join(КЛ,"_BAK_8полей"); os.makedirs(БЭК,exist_ok=True)

d=json.load(open(os.path.join(КЛ,"каталог.json"),encoding="utf-8"))
items=d if isinstance(d,list) else next((v for v in d.values() if isinstance(v,list)),[])
ok=0; пропуск=0
for it in items:
    путь_атом=it.get("атомы"); _id=it.get("id")
    if not путь_атом or not _id: пропуск+=1; continue
    полн_f=os.path.join(ПОЛН,_id+".json")
    орig=os.path.join(КЛ,путь_атом)
    if not os.path.exists(полн_f): пропуск+=1; continue
    full=json.load(open(полн_f,encoding="utf-8"))
    # длительность из оригинала (если есть) — чтоб не сдвинуть тайминги
    дл=full.get("parent_duration_sec")
    if os.path.exists(орig):
        try: дл=json.load(open(орig,encoding="utf-8")).get("длительность_сек",дл)
        except: pass
        # бэкап оригинала (плоско по id)
        shutil.cop2 if False else shutil.copy2(орig, os.path.join(БЭК,_id+".json"))
    новый={
        "version":"2.0-full","длительность_сек":дл,
        "atoms_count":full.get("atoms_count"),
        "atoms":full.get("атомы"),                      # ← ключ как ждёт приложение
        "parent_params_full":full.get("parent_params_full"),  # 104 на клетке
    }
    os.makedirs(os.path.dirname(орig),exist_ok=True)
    json.dump(новый,open(орig,"w",encoding="utf-8"),ensure_ascii=False)
    ok+=1

print(f"подключено: {ok} клеток · пропущено {пропуск}")
print(f"бэкап 8-польных: данные/клеточки/_BAK_8полей/")
# проверка одной
import glob
пример=os.path.join(КЛ,items[0]["атомы"])
v=json.load(open(пример,encoding="utf-8"))
a=v["atoms"][0]
print(f"проверка [{items[0]['id']}]: ключ atoms есть={'atoms' in v}, атомов={v['atoms_count']}, полей атома={len(a)}, паспорт={len(v.get('parent_params_full',{}))}")
