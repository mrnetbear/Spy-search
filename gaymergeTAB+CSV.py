import re
import sys
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, total=None, desc=None):
        return it

in_path = 'MergedData.csv'
out_dups = 'dublicate_TicketNumber.csv'
out_dedup = 'MergedData_TicketDedup.csv'

rus_passport_re = re.compile(r'^\d{4}\s\d{6}$')

def read_clean(path):
    df = pd.read_csv(path, dtype=str)
    obj = df.select_dtypes(include='object')
    df[obj.columns] = obj.apply(lambda s: s.str.strip())
    df = df.replace({'': pd.NA})
    return df

def first_non_na(series):
    for v in series:
        if pd.notna(v):
            return v
    return pd.NA

def pick_russian_passport(series):
    vals = [v for v in series if pd.notna(v)]
    for v in vals:
        if rus_passport_re.match(v or ''):
            return v
    return vals[0] if vals else pd.NA

print('Чтение данных...', flush=True)
df = read_clean(in_path)
cols = list(df.columns)

print('Поиск дубликатов TicketNumber...', flush=True)
valid_ticket = df['TicketNumber'].notna() & (df['TicketNumber'] != '0')
dup_mask = valid_ticket & df.duplicated(subset=['TicketNumber'], keep=False)
dups_df = df[dup_mask].copy().sort_values(['TicketNumber'])
dups_df.to_csv(out_dups, index=False)

df_out = df.copy()

if not dups_df.empty:
    print(f'Обработка {dups_df["TicketNumber"].nunique()} групп дубликатов...', flush=True)
    grouped = dups_df.groupby('TicketNumber', sort=False)
    to_append = []
    to_remove_idx = set()

    for t, grp in tqdm(grouped, total=dups_df['TicketNumber'].nunique(), desc='Дедупликация'):
        docs = grp['PassengerDocument'].dropna().unique()
        if len(docs) > 1:
            chosen_doc = pick_russian_passport(grp['PassengerDocument'])
            if pd.notna(chosen_doc):
                order_idx = list(grp[grp['PassengerDocument'] == chosen_doc].index) + \
                            [i for i in grp.index if grp.at[i, 'PassengerDocument'] != chosen_doc]
                ordered = grp.loc[order_idx]
            else:
                ordered = grp

            combined = ordered.apply(first_non_na, axis=0)
            if pd.notna(chosen_doc):
                combined['PassengerDocument'] = chosen_doc
            combined = combined.reindex(cols)

            to_remove_idx.update(df_out[df_out['TicketNumber'] == t].index.tolist())
            to_append.append(combined)

    if to_remove_idx:
        df_out = df_out.drop(index=list(to_remove_idx))
    if to_append:
        df_out = pd.concat([df_out, pd.DataFrame(to_append, columns=cols)], ignore_index=True)

print('Сохранение результатов...', flush=True)
df_out.to_csv(out_dedup, index=False)
print('Готово.', flush=True)