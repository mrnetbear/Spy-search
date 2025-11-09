import os
import warnings
warnings.filterwarnings("ignore")

import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from cuml.preprocessing import StandardScaler
from cuml.decomposition import PCA
from cuml.manifold import UMAP
from cuml.cluster import DBSCAN

from tqdm.auto import tqdm


def load_data(path):
    force_str = {
        'TicketNumber': 'str',
        'PassengerDocument': 'str',
        'FlightNumber': 'str',
        'FlightTime': 'str',
        'FlightDate': 'str',
        'From': 'str',
        'Dest': 'str',
        'DepartureCity': 'str',
        'ArrivalCity': 'str',
        'DepartureCountry': 'str',
        'ArrivalCountry': 'str',
        'AgentInfo': 'str',
        'CodeShare': 'str',
        'TrvCls': 'str',
        'Meal': 'str',
        'Sex': 'str',
        'PassengerBirthDate': 'str',
        'lastname': 'str',
        'name': 'str',
        'Baggage': 'str'
    }
    return cudf.read_csv(path, dtype=force_str)


def clean_and_engineer(df):
    cols_exist = {c.lower(): c for c in df.columns}

    def col(name, default=None):
        ln = name.lower()
        if ln in cols_exist:
            return cols_exist[ln]
        df[name] = '' if default is None else default
        return name

    c_from = col('From'); c_dest = col('Dest'); c_flightno = col('FlightNumber')
    c_flightdate = col('FlightDate'); c_dep_city = col('DepartureCity'); c_arr_city = col('ArrivalCity')
    c_dep_country = col('DepartureCountry'); c_arr_country = col('ArrivalCountry')
    c_flighttime = col('FlightTime'); c_baggage = col('Baggage'); c_codeshare = col('CodeShare')
    c_agent = col('AgentInfo'); c_trvcls = col('TrvCls'); c_meal = col('Meal'); c_sex = col('Sex')
    c_birth = col('PassengerBirthDate'); c_ticket = col('TicketNumber'); c_lastname = col('lastname'); c_name = col('name')

    def to_str(s): return s.astype('str').fillna('').str.strip()
    for c in [c_from, c_dest, c_flightno, c_flightdate, c_dep_city, c_arr_city, c_dep_country,
              c_arr_country, c_flighttime, c_baggage, c_codeshare, c_agent, c_trvcls,
              c_meal, c_sex, c_birth, c_ticket, c_lastname, c_name]:
        df[c] = to_str(df[c])

    def is_na_like(s):
        sl = s.astype('str').str.lower()
        return (s == '') | (s == '0') | (s == '0.0') | (sl == 'nan') | (sl == 'none') | (sl == 'null')

    for c in [c_from, c_dest, c_flightno, c_dep_city, c_arr_city, c_dep_country, c_arr_country,
              c_agent, c_trvcls, c_meal, c_sex, c_codeshare]:
        df.loc[is_na_like(df[c]), c] = 'Unknown'

    df['route_code'] = (df[c_from] + '_' + df[c_dest]).fillna('Unknown_Unknown')
    df['flight_code'] = (df[c_flightno] + '_' + df[c_flightdate]).fillna('Unknown_Unknown')

    def parse_time_to_hour(s):
        ss = s.astype('str').fillna('').str.strip()
        m = ss.str.extract(r'^\s*(\d{1,2})\s*:\s*(\d{2})', expand=True)
        h = m[0].astype('float32')
        return h.fillna(0)

    df['dep_hour'] = parse_time_to_hour(df[c_flighttime])

    def parse_date(s):
        ss = s.astype('str').fillna('1970-01-01').str.strip()
        mask = ss.str.contains(r'^\d{4}-\d{2}-\d{2}$', regex=True)
        ss = ss.copy()
        ss[~mask] = '1970-01-01'
        return cudf.to_datetime(ss, format='%Y-%m-%d')

    fdate = parse_date(df[c_flightdate]); bdate = parse_date(df[c_birth])

    df['flight_dow'] = fdate.dt.weekday.fillna(0).astype('int32')
    df['age_at_flight'] = ((fdate - bdate).dt.days / 365.25).fillna(0).clip(lower=0, upper=110).astype('float32')

    def baggage_to_num(s):
        digits = s.astype('str').str.extract(r'(\d+)', expand=False)
        return digits.fillna('0').astype('float32')

    df['baggage_count'] = baggage_to_num(df[c_baggage])

    cat_cols = ['route_code','flight_code', c_agent, c_codeshare, c_trvcls, c_meal, c_sex,
                c_from, c_dest, c_dep_city, c_arr_city, c_dep_country, c_arr_country]
    for c in cat_cols:
        if c not in df.columns: df[c] = 'Unknown'
        else: df[c] = df[c].astype('str')
    code_cols = []
    for c in cat_cols:
        codes, _ = df[c].factorize()
        df[c + '_code'] = codes.astype('int32'); code_cols.append(c + '_code')

    num_cols = ['dep_hour', 'flight_dow', 'age_at_flight', 'baggage_count']
    feat_cols = code_cols + num_cols
    X = df[feat_cols].astype('float32')

    keep_cols = [c_ticket, c_lastname, c_name, c_from, c_dest, c_dep_city, c_arr_city,
                 c_dep_country, c_arr_country, c_flightno, c_flightdate, c_flighttime,
                 'route_code', 'flight_code', c_agent, c_trvcls, c_meal, c_sex, c_baggage]
    keep_cols = [c for c in keep_cols if c in df.columns]
    meta = df[keep_cols].copy()
    return X, meta, df


def cluster_gpu(X, random_state=42):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(30, int(Xs.shape[1])))
    Xp = pca.fit_transform(Xs)

    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.05, init='spectral', random_state=random_state)
    emb2d = umap.fit_transform(Xp)

    db = DBSCAN(eps=1.5, min_samples=3)
    labels = db.fit_predict(Xp)
    return labels, emb2d


def compute_suspicion(meta_df, labels):
    labels_series = labels if isinstance(labels, cudf.Series) else cudf.Series(labels)
    gdf = meta_df.copy()
    gdf['cluster'] = labels_series
    gdf = gdf[gdf['cluster'] != -1]
    if gdf.shape[0] == 0:
        return gdf.assign(suspicion_score=0.0, cluster_size=0), cudf.DataFrame({'cluster': [], 'suspicion_score': [], 'cluster_size': []})

    counts = gdf[['cluster', 'flight_code']].groupby(['cluster', 'flight_code']).size().reset_index(name='n')
    counts['pairs'] = (counts['n'] * (counts['n'] - 1)) / 2.0

    sizes = gdf.groupby('cluster').size().reset_index(name='cluster_size')
    sizes['total_pairs'] = (sizes['cluster_size'] * (sizes['cluster_size'] - 1)) / 2.0

    num_pairs_cluster = counts.groupby('cluster')['pairs'].sum().reset_index()
    stats = sizes.merge(num_pairs_cluster, on='cluster', how='left').fillna({'pairs': 0.0})
    stats['suspicion_score'] = (stats['pairs'] / stats['total_pairs']).fillna(0.0)
    stats.loc[stats['total_pairs'] == 0, 'suspicion_score'] = 0.0

    gdf = gdf.merge(stats[['cluster', 'suspicion_score', 'cluster_size']], on='cluster', how='left')
    return gdf, stats[['cluster', 'suspicion_score', 'cluster_size']]


def select_suspicious(gdf, min_size=3, min_score=0.2):
    return gdf[(gdf['cluster_size'] >= min_size) & (gdf['suspicion_score'] >= min_score)]


def plot_clusters(emb2d, labels, out_path='clusters_plot.png', sample_size=1000, random_state=42):
    np.random.seed(random_state)

    # Приводим emb2d к NumPy
    if isinstance(emb2d, cudf.DataFrame) or isinstance(emb2d, cudf.Series):
        emb = emb2d.to_numpy()
    elif isinstance(emb2d, cp.ndarray):
        emb = cp.asnumpy(emb2d)
    else:
        emb = np.asarray(emb2d)

    # Приводим labels к NumPy 1D
    if isinstance(labels, cudf.Series):
        lab = labels.to_numpy().ravel()
    elif isinstance(labels, cp.ndarray):
        lab = cp.asnumpy(labels).ravel()
    else:
        lab = np.asarray(labels).ravel()

    n = emb.shape[0]
    if sample_size is not None and n > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        emb = emb[idx]
        lab = lab[idx]

    plt.figure(figsize=(10, 7))
    unique = np.unique(lab)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))
    for c, u in zip(colors, unique):
        m = lab == u
        if u == -1:
            plt.scatter(emb[m, 0], emb[m, 1], s=10, c=[(0.6, 0.6, 0.6, 0.5)], label='noise', marker='x')
        else:
            plt.scatter(emb[m, 0], emb[m, 1], s=14, c=[c], label=f'cluster {u}', alpha=0.85)

    plt.title('Кластеры пассажиров (UMAP, DBSCAN)')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
def main(
    input_csv='flights.csv',
    suspicious_csv='suspicious_groups.csv',
    clusters_plot='clusters_plot.png',
    min_cluster_size=3,
    min_suspicion_score=0.2,
    random_state=42
):
    pbar = tqdm(total=7, desc='Процесс', ncols=100)
    pbar.set_postfix_str('Загрузка данных')
    df = load_data(input_csv)
    pbar.update(1)

    pbar.set_postfix_str('Очистка и фичи')
    X, meta, _ = clean_and_engineer(df)
    pbar.update(1)

    pbar.set_postfix_str('Кластеризация на GPU')
    labels, emb2d = cluster_gpu(X, random_state=random_state)
    pbar.update(1)

    pbar.set_postfix_str('Подсчёт подозрительности')
    grouped, stats = compute_suspicion(meta, labels)
    pbar.update(1)

    pbar.set_postfix_str('Фильтрация подозрительных групп')
    suspicious = select_suspicious(grouped, min_size=min_cluster_size, min_score=min_suspicion_score)
    pbar.update(1)

    pbar.set_postfix_str('Сохранение CSV')
    suspicious.to_csv(suspicious_csv, index=False)
    pbar.update(1)

    pbar.set_postfix_str('Построение графиков')
    plot_clusters(emb2d, labels, out_path=clusters_plot, sample_size=1500, random_state=random_state)
    pbar.update(1)

    pbar.close()
    print(f'Готово. Подозрительные группы: {int(suspicious.shape[0])} строк. CSV: {os.path.abspath(suspicious_csv)}. Плот: {os.path.abspath(clusters_plot)}')


if __name__ == '__main__':
    main(
        input_csv='data/MergedResult-cleared-unify-caps.csv',
        suspicious_csv='suspicious_groups.csv',
        clusters_plot='clusters_plot.png',
        min_cluster_size=3,
        min_suspicion_score=0.2,
        random_state=42
    )