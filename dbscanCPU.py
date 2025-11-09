import sys, subprocess, os, logging

def ensure(pkgs):
    import importlib
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])
ensure(["pandas", "numpy", "networkx", "scikit-learn", "gensim", "tqdm"])

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from itertools import combinations
import hashlib
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CSV_PATH = os.getenv("FLIGHTS_CSV_PATH", "csv/RESULT/MergedResult-cleared-unify-caps.csv")

def norm_str(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    if s in {"0", "0.0", "nan", "NaN", ""}: return ""
    return s

def to_dt(date_col, time_col):
    d = norm_str(date_col)
    t = norm_str(time_col) or "00:00"
    if not d: return pd.NaT
    return pd.to_datetime(f"{d} {t}", errors="coerce")

def passenger_id(row):
    key = "|".join([
        norm_str(row.get("lastname")).upper(),
        norm_str(row.get("name")).upper(),
        norm_str(row.get("PassengerBirthDate")),
        norm_str(row.get("PassengerDocument")).replace(" ", "")
    ])
    if not key.strip("|"):
        key = "|".join([
            norm_str(row.get("TicketNumber")),
            norm_str(row.get("FlightNumber")),
            norm_str(row.get("FlightDate"))
        ])
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def flight_id(row):
    fn = norm_str(row.get("FlightNumber")).upper() or norm_str(row.get("Code")).upper()
    fd = norm_str(row.get("FlightDate"))
    return f"{fn}_{fd}"

def load_df(path):
    if os.path.exists(path):
        logging.info(f"Чтение CSV: {path}")
        df = pd.read_csv(path)
    else:
        logging.info("CSV не найден, используется demo-выборка.")
        from io import StringIO
        demo = """AgentInfo,ArrivalCity,ArrivalCountry,ArrivalDate,ArrivalTime,Baggage,BaggageStarus,BookingCode,Code,CodeShare,DepartureCity,DepartureCountry,Dest,FlightDate,FlightNumber,FlightTime,From,Loyality,Meal,PassengerBirthDate,PassengerDocument,Sex,TicketNumber,TrvCls,lastname,name
Travelgenio,MOSCOW,RUSSIA,2017-01-01,14:30,1PC,0,PBTDYJ,YFLXWW,Own,SALEKHARD,RUSSIA,SVO,2017-01-01,SU1491,13:20,SLY,DT 353568614,ORML,1978-07-17,0018 818437,Female,1647854967612619.0,Y,MOSKVINA,VLADA
0,MOSCOW,RUSSIA,0,0,0,0,FLWXKE,YFLXDN,Own,ORENBURG,RUSSIA,SVO,2017-01-01,SU6170,05:05,REN,DT 873909674,0,1979-01-04,0093 770708,Male,5123834104775340.0,Y,KASATKIN,EMIL
Travelgenio,MOSCOW,RUSSIA,2017-01-01,23:40,0,Registered,QDZUUC,YSTNOD,Own,SOCHI,RUSSIA,SVO,2017-01-01,SU1141,21:05,AER,SU 551179199,VLML,1994-06-18,0206 773384,Female,1138129192341931.0,Y,GORDEEVA,ALENA
Go2See,MOSCOW,RUSSIA,2017-01-04,13:55,0,0,GFTEAN,YSTNSQ,Own,BELGOROD,RUSSIA,SVO,2017-01-04,SU1371,12:35,EGO,SU 555858874,0,1988-12-22,3703 779581,Male,3356803479343771.0,Y,VERESHCHAGIN,ROMAN
OZON.travel,MOSCOW,RUSSIA,2017-01-02,19:05,0,0,BCGOLZ,YFLXRL,Own,SAMARA,RUSSIA,SVO,2017-01-02,SU1213,18:05,KUF,KE 482181852,VGML,0,8846 854445,Female,2696570973220838.0,Y,ISAEVA,TAYSIYA
OneTwoTrip,MOSCOW,RUSSIA,2017-01-02,20:25,0PC,Delayed,XIGYAQ,PGRPLZ,Own,BARNAUL,RUSSIA,SVO,2017-01-02,SU1433,19:55,BAX,FB 55024361,0,1985-05-23,2922 452959,Male,4384410429098700.0,P,GORYACHEV,VLADIMIR
KupiBilet,MOSCOW,RUSSIA,2017-01-03,21:15,0,0,LGHXLR,JRSTED,Own,ROSTOV,RUSSIA,SVO,2017-01-03,SU1161,19:10,ROV,DT 726035684,STML,0,8819 962216,Male,9679408162555746.0,J,KOROLKOV,SAVVA
Aeroflot,MOSCOW,RUSSIA,2017-01-01,07:55,1PC,0,AUNSLO,YFLXGG,Own,KHANTY-MANSIYSK,RUSSIA,SVO,2017-01-01,SU1383,06:40,HMA,KE 72029401,VGML,0,7189 025309,Male,7241598764878272.0,Y,ZINOVEV,IVAN
OneTwoTrip,MOSCOW,RUSSIA,2017-01-01,15:15,0,Registered,VUUMAI,YFLXJN,Own,SOCHI,RUSSIA,SVO,2017-01-01,SU1123,12:45,AER,KE 531109940,VGML,1978-12-07,7838 061967,Female,7191506660173498.0,Y,BOGDANOVA,ELMIRA
Go2See,MOSCOW,RUSSIA,2017-01-01,13:55,0PC,0,BXWGFJ,YSTNIL,Own,MAGNETIOGORSK,RUSSIA,SVO,2017-01-01,SU1435,13:15,MQF,FB 852426376,0,2000-12-15,1001 759045,0,9051259106326872.0,Y,RUSANOV,DANILA
OZON.travel,MOSCOW,RUSSIA,2017-01-02,07:25,0PC,0,XSMAOW,YRSTIL,Own,KEMEROVO,RUSSIA,SVO,2017-01-02,SU1451,06:55,KEJ,DT 158689506,0,1985-10-20,6695 468980,Female,6674877020615909.0,Y,PARFENOVA,ALINA
Go2See,MOSCOW,RUSSIA,2017-01-01,12:35,0,0,SPHYUL,YRSTCQ,Own,SOCHI,RUSSIA,SVO,2017-01-01,SU1139,09:50,AER,DT 95355643,0,1993-09-15,2064 623486,Male,1804782809401583.2,Y,KIREEV,DEMID
City.Travel,MOSCOW,RUSSIA,2017-01-01,18:15,0,Transit,QXOXGV,YRSTVR,Own,SARANSK,RUSSIA,SVO,2017-01-01,SU1465,16:40,SKX,KE 22302027,KSML,1984-03-10,3394 326380,Male,9265978798599682.0,Y,KORSHUNOV,DAVID
0,MOSCOW,RUSSIA,0,0,0,Transit,WMEVQN,YSTNVC,Own,NOVYJ URENGOJ,RUSSIA,SVO,2017-01-01,SU1523,09:35,NUX,KE 977609292,0,1971-06-22,9273 123111,Female,2407112421262518.0,Y,ZELENINA,ULYANA
Aerobilet,MOSCOW,RUSSIA,2017-01-02,00:05,0PC,Delayed,0,ARSTDE,Own,ROSTOV,RUSSIA,SVO,2017-01-01,SU1169,21:00,ROV,SU 564595900,KSML,1983-12-06,6522 786406,Female,6581689342413446.0,A,KARASEVA,KAMILLA
Tickets.ru,KHABAROVSK,RUSSIA,2017-01-01,13:20,0,0,ICLZGO,YFLXAY,Operated,NAN,NAN,KHV,2017-01-01,SU4606,12:10,OHH,KE 472161685,0,1979-06-15,1630 121892,Male,451344792163574.0,Y,DRUZHININ,RODYON
        """
        df = pd.read_csv(StringIO(demo))
    return df

df = load_df(CSV_PATH)

logging.info("Очистка и подготовка данных")
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).replace({"0":"", "0.0":"", "nan":"", "NaN":""})

df["origin"] = df["From"].apply(norm_str).str.upper()
df["dest"] = df["Dest"].apply(norm_str).str.upper()
df["flight_date"] = df["FlightDate"].apply(norm_str)
df["flight_time"] = df["FlightTime"].apply(norm_str)
df["flight_dt"] = [to_dt(d, t) for d, t in zip(df["flight_date"], df["flight_time"])]
df["arr_dt"] = [to_dt(d, t) for d, t in zip(df["ArrivalDate"], df["ArrivalTime"])]

df["pid"] = df.apply(passenger_id, axis=1)
df["fid"] = df.apply(flight_id, axis=1)
df = df[(df["origin"]!="") & (df["dest"]!="") & (df["fid"]!="") & (df["pid"]!="")]

G = nx.Graph()

logging.info("Построение узлов графа")
for _, r in tqdm(df.iterrows(), total=len(df), desc="Узлы"):
    pid = r["pid"]
    if pid not in G:
        G.add_node(pid, name=norm_str(r.get("name")).title(), lastname=norm_str(r.get("lastname")).title(),
                   doc=norm_str(r.get("PassengerDocument")), birth=norm_str(r.get("PassengerBirthDate")))

by_flight = df.groupby("fid")
by_route_day = df.groupby(["origin", "dest", "flight_date"])

def add_edge(a, b, w_add, f_add=0, r_add=0):
    if a == b: return
    if G.has_edge(a, b):
        G[a][b]["weight"] += w_add
        G[a][b]["co_flights"] += f_add
        G[a][b]["co_routes"] += r_add
    else:
        G.add_edge(a, b, weight=w_add, co_flights=f_add, co_routes=r_add)

logging.info("Добавление рёбер по общим рейсам")
for fid, g in tqdm(by_flight, total=len(by_flight), desc="Рейсы"):
    pids = list(g["pid"].unique())
    for u, v in combinations(pids, 2):
        add_edge(u, v, w_add=3.0, f_add=1, r_add=0)

logging.info("Добавление рёбер по совпадению маршрута в день")
for (o, d, day), g in tqdm(by_route_day, total=len(by_route_day), desc="Маршруты/день"):
    pids = list(g["pid"].unique())
    for u, v in combinations(pids, 2):
        add_edge(u, v, w_add=1.0, f_add=0, r_add=1)

if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    print("Недостаточно данных для построения графа.")
    sys.exit(0)

logging.info(f"Граф: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")
rng = np.random.default_rng(42)
def random_walk(G, start, length=40):
    walk = [start]
    cur = start
    for _ in range(length-1):
        nbrs = list(G.neighbors(cur))
        if not nbrs: break
        weights = np.array([G[cur][n].get("weight",1.0) for n in nbrs], dtype=float)
        p = weights / weights.sum()
        cur = rng.choice(nbrs, p=p)
        walk.append(cur)
    return walk

logging.info("Генерация случайных прогулок (Node2Vec-подобно)")
nodes = list(G.nodes())
walks = []
walks_per_node = 10
walk_len = 40
pbar_walks = tqdm(total=walks_per_node*len(nodes), desc="Прогулки")
for _ in range(walks_per_node):
    rng.shuffle(nodes)
    for n in nodes:
        walks.append([str(x) for x in random_walk(G, n, walk_len)])
        pbar_walks.update(1)
pbar_walks.close()

if len(walks) == 0:
    print("Недостаточно данных для обучения эмбеддингов.")
    sys.exit(0)

class TQDMCallback(CallbackAny2Vec):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Word2Vec")
    def on_epoch_end(self, model):
        self.pbar.update(1)
    def on_train_end(self, model):
        self.pbar.close()

logging.info("Обучение эмбеддингов узлов")
epochs = 10
w2v = Word2Vec(
    sentences=walks,
    vector_size=64,
    window=10,
    min_count=1,
    sg=1,
    negative=10,
    workers=1,
    epochs=epochs,
    callbacks=[TQDMCallback(epochs)]
)
emb = np.vstack([w2v.wv[str(n)] for n in nodes])

logging.info("Кластеризация DBSCAN")
cl = DBSCAN(eps=0.6, min_samples=3, metric="euclidean").fit(emb)
labels = cl.labels_
node2label = {n: int(l) for n, l in zip(nodes, labels)}

clusters = {}
for n, lbl in node2label.items():
    if lbl == -1: continue
    clusters.setdefault(lbl, []).append(n)

def cluster_stats(members):
    sub = G.subgraph(members)
    n = len(members)
    if n < 2: 
        return {"size": n, "avg_w": 0, "avg_co_flights": 0, "edges": 0}
    weights = []
    cofl = []
    for u, v, d in sub.edges(data=True):
        weights.append(d.get("weight", 0.0))
        cofl.append(d.get("co_flights", 0))
    m = len(weights)
    avg_w = float(np.mean(weights)) if m else 0.0
    avg_cf = float(np.mean(cofl)) if m else 0.0
    return {"size": n, "avg_w": avg_w, "avg_co_flights": avg_cf, "edges": m}

scores = {}
for cid, members in clusters.items():
    st = cluster_stats(members)
    score = st["avg_w"] * 0.6 + st["avg_co_flights"] * 0.4
    scores[cid] = (score, st)

SCORE_THRESHOLD = 1.5
MIN_SIZE = 3

suspicious = []
for cid, (score, st) in scores.items():
    if st["size"] >= MIN_SIZE and score >= SCORE_THRESHOLD:
        suspicious.append((cid, score, st))
suspicious.sort(key=lambda x: (-x[1], -x[2]["size"]))

print("Найденные группы с высокой связанностью (возможные ко-тревелеры):")
for cid, score, st in suspicious:
    members = clusters[cid]
    print(f"\nКластер {cid}: size={st['size']}, score={score:.2f}, avg_w={st['avg_w']:.2f}, avg_co_flights={st['avg_co_flights']:.2f}")
    for pid in members:
        n = G.nodes[pid]
        print(f"- {n.get('lastname','')} {n.get('name','')} | {n.get('birth','')} | {n.get('doc','')} | pid={pid[:8]}")

for cid, _, st in suspicious:
    sub = G.subgraph(clusters[cid])
    edges_sorted = sorted(sub.edges(data=True), key=lambda e: (e[2].get("co_flights",0), e[2].get("weight",0)), reverse=True)
    print(f"\nТоп связей внутри кластера {cid}:")
    for u, v, d in edges_sorted[:10]:
        a = G.nodes[u]; b = G.nodes[v]
        print(f"{a.get('lastname','')}/{a.get('name','')} <-> {b.get('lastname','')}/{b.get('name','')}: co_flights={d.get('co_flights',0)}, weight={d.get('weight',0):.1f}")
