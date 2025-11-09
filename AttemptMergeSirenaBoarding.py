import pandas as pd

sirena_path = 'Sirena-export-fixed/Sirena-export-fixed-UNIFY-cleaned.csv'
bd_path = 'BoardingData/BoardingData-UNIFY-cleaned.csv'
out_merged = 'MergedData.csv'
out_fail_data = 'FailData.csv'
out_fail_match = 'FailMatch.csv'
out_find_lost_ticket = 'FindLostTicketNumber.csv'

NOT_PRESENTED = 'Not presented'
NAME_FIELDS = {'name', 'lastname'}

def read_clean(path):
    df = pd.read_csv(path, dtype=str)
    obj = df.select_dtypes(include='object')
    df[obj.columns] = obj.apply(lambda s: s.str.strip())
    df = df.replace({'': pd.NA})
    return df

def normalize_bd_birthdate(df):
    if 'PassengerBirthDate' in df.columns:
        df['PassengerBirthDate'] = pd.to_datetime(
            df['PassengerBirthDate'], format='%m/%d/%Y', errors='coerce'
        ).dt.strftime('%Y-%m-%d')
    return df

def is_missing_ticket(x):
    return pd.isna(x) or x == '0'

def is_empty_booking(v):
    return pd.isna(v) or v == NOT_PRESENTED

def is_empty_birthdate(v):
    return pd.isna(v) or v == '0'

def coalesce(row, col):
    v_bd = row.get(f'{col}_bd', pd.NA)
    v_sir = row.get(f'{col}_sir', pd.NA)
    if col == 'TicketNumber':
        if pd.notna(v_bd) and v_bd != '0':
            return v_bd
        return v_sir
    if col == 'BookingCode':
        if is_empty_booking(v_bd) and not is_empty_booking(v_sir):
            return v_sir
        if not is_empty_booking(v_bd):
            return v_bd
        return v_sir if not is_empty_booking(v_sir) else v_bd
    if col == 'PassengerBirthDate':
        if not is_empty_birthdate(v_bd):
            return v_bd
        if not is_empty_birthdate(v_sir):
            return v_sir
        return v_bd if pd.notna(v_bd) else v_sir
    if col in NAME_FIELDS:
        if pd.notna(v_bd):
            return v_bd
        return v_sir
    if pd.notna(v_bd):
        return v_bd
    return v_sir

def coalesce_frame(df_merged, all_cols):
    out = pd.DataFrame(index=df_merged.index)
    for c in all_cols:
        c_bd = f'{c}_bd'
        c_sir = f'{c}_sir'
        if c_bd in df_merged.columns or c_sir in df_merged.columns:
            out[c] = df_merged.apply(lambda r: coalesce(r, c), axis=1)
        elif c in df_merged.columns:
            out[c] = df_merged[c]
        else:
            out[c] = pd.NA
    return out

def find_mismatches(df_merged, common_cols, exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    exclude_cols |= NAME_FIELDS  # name/lastname не считаем несостыковкой
    diffs = []
    for c in common_cols:
        if c in exclude_cols:
            continue
        cb, cs = f'{c}_bd', f'{c}_sir'
        if cb in df_merged.columns and cs in df_merged.columns:
            if c == 'BookingCode':
                left = df_merged[cb].where(~df_merged[cb].isin([NOT_PRESENTED]), pd.NA)
                right = df_merged[cs].where(~df_merged[cs].isin([NOT_PRESENTED]), pd.NA)
                d = (left.notna() & right.notna()) & (left != right)
            elif c == 'PassengerBirthDate':
                left = df_merged[cb].where(~df_merged[cb].isin(['0']), pd.NA)
                right = df_merged[cs].where(~df_merged[cs].isin(['0']), pd.NA)
                d = (left.notna() & right.notna()) & (left != right)
            else:
                d = (df_merged[cb].fillna(pd.NA) != df_merged[cs].fillna(pd.NA))
            diffs.append((c, d))
    if not diffs:
        n = len(df_merged)
        return pd.Series([False]*n, index=df_merged.index), pd.Series(['']*n, index=df_merged.index)
    any_diff = diffs[0][1].copy()
    for _, d in diffs[1:]:
        any_diff |= d
    def cols_list(i):
        return ';'.join([c for c, d in diffs if bool(d.iat[i])])
    mismatch_cols_series = pd.Series([''] * len(df_merged), index=df_merged.index, dtype=object)
    if any_diff.any():
        mismatch_cols_series = pd.Series(
            [cols_list(i) if any_diff.iat[i] else '' for i in range(len(df_merged))],
            index=df_merged.index, dtype=object
        )
    return any_diff, mismatch_cols_series

sirena = read_clean(sirena_path)
bd = read_clean(bd_path)

bd = normalize_bd_birthdate(bd)

sirena['__sir_idx'] = range(len(sirena))
bd['__bd_idx'] = range(len(bd))

common_cols = sorted(list(set(sirena.columns).intersection(set(bd.columns))))
common_cols = [c for c in common_cols if not c.startswith('__')]

ticket_present_mask = ~bd['TicketNumber'].map(is_missing_ticket)
bd_with_ticket = bd[ticket_present_mask].copy()
bd_no_ticket = bd[~ticket_present_mask].copy()

merge_ticket = bd_with_ticket.merge(
    sirena, on=['PassengerDocument', 'TicketNumber'], how='left', suffixes=('_bd', '_sir')
)

matched_ticket_mask = merge_ticket['__sir_idx'].notna()
matched_ticket = merge_ticket[matched_ticket_mask].copy()
unmatched_ticket = merge_ticket[~matched_ticket_mask].copy()

mismatch_mask, mismatch_cols_series = find_mismatches(
    matched_ticket, common_cols, exclude_cols={'PassengerDocument', 'TicketNumber'}
)
fail_data = matched_ticket[mismatch_mask].copy()
if not fail_data.empty:
    fail_data = fail_data.assign(MismatchColumns=mismatch_cols_series[mismatch_mask].values)
    fail_data.to_csv(out_fail_data, index=False)
else:
    pd.DataFrame().to_csv(out_fail_data, index=False)

all_cols_union = sorted(list(set(sirena.columns).union(set(bd.columns))))
all_cols_union = [c for c in all_cols_union if not c.startswith('__')]

final_parts = []

if not matched_ticket.empty:
    final_ticket = coalesce_frame(matched_ticket, all_cols_union)
    final_parts.append(final_ticket)

booking_present_mask = bd_no_ticket['BookingCode'].notna() & (~bd_no_ticket['BookingCode'].isin([NOT_PRESENTED]))
bd_no_ticket_with_booking = bd_no_ticket[booking_present_mask].copy()
bd_no_ticket_without_booking = bd_no_ticket[~booking_present_mask].copy()

keys_with_booking = ['PassengerDocument', 'FlightDate', 'FlightTime', 'FlightNumber', 'CodeShare', 'BookingCode']
keys_without_booking = ['PassengerDocument', 'FlightDate', 'FlightTime', 'FlightNumber', 'CodeShare']

merge_no_ticket_with_booking = bd_no_ticket_with_booking.merge(
    sirena, on=keys_with_booking, how='left', suffixes=('_bd', '_sir')
)
merge_no_ticket_without_booking = bd_no_ticket_without_booking.merge(
    sirena, on=keys_without_booking, how='left', suffixes=('_bd', '_sir')
)

matched_fields_with_booking_mask = merge_no_ticket_with_booking['__sir_idx'].notna()
matched_fields_without_booking_mask = merge_no_ticket_without_booking['__sir_idx'].notna()

matched_fields_with_booking = merge_no_ticket_with_booking[matched_fields_with_booking_mask].copy()
matched_fields_without_booking = merge_no_ticket_without_booking[matched_fields_without_booking_mask].copy()

if not matched_fields_with_booking.empty:
    final_fields_wb = coalesce_frame(matched_fields_with_booking, all_cols_union)
    final_parts.append(final_fields_wb)
if not matched_fields_without_booking.empty:
    final_fields_wob = coalesce_frame(matched_fields_without_booking, all_cols_union)
    final_parts.append(final_fields_wob)

# Строки с восстановленным TicketNumber
lost_frames = []
if not matched_fields_with_booking.empty:
    lost_wb = matched_fields_with_booking[
        (matched_fields_with_booking['TicketNumber_bd'].isna() | (matched_fields_with_booking['TicketNumber_bd'] == '0')) &
        (matched_fields_with_booking['TicketNumber_sir'].notna())
    ].copy()
    if not lost_wb.empty:
        lost_frames.append(coalesce_frame(lost_wb, all_cols_union))
if not matched_fields_without_booking.empty:
    lost_wob = matched_fields_without_booking[
        (matched_fields_without_booking['TicketNumber_bd'].isna() | (matched_fields_without_booking['TicketNumber_bd'] == '0')) &
        (matched_fields_without_booking['TicketNumber_sir'].notna())
    ].copy()
    if not lost_wob.empty:
        lost_frames.append(coalesce_frame(lost_wob, all_cols_union))

if lost_frames:
    pd.concat(lost_frames, ignore_index=True).to_csv(out_find_lost_ticket, index=False)
else:
    pd.DataFrame(columns=all_cols_union).to_csv(out_find_lost_ticket, index=False)

matched_bd_idxs = set()
if not matched_ticket.empty:
    matched_bd_idxs.update(matched_ticket['__bd_idx'].tolist())
if not matched_fields_with_booking.empty:
    col = '__bd_idx_bd' if '__bd_idx_bd' in matched_fields_with_booking.columns else '__bd_idx'
    matched_bd_idxs.update(matched_fields_with_booking[col].dropna().astype(int).tolist())
if not matched_fields_without_booking.empty:
    col = '__bd_idx_bd' if '__bd_idx_bd' in matched_fields_without_booking.columns else '__bd_idx'
    matched_bd_idxs.update(matched_fields_without_booking[col].dropna().astype(int).tolist())

unmatched_bd = bd[~bd['__bd_idx'].isin(matched_bd_idxs)].copy()
if not unmatched_bd.empty:
    unmatched_bd.to_csv(out_fail_match, index=False)
else:
    pd.DataFrame().to_csv(out_fail_match, index=False)

if not unmatched_bd.empty:
    missing_cols_from_sir = [c for c in sirena.columns if c not in unmatched_bd.columns and not c.startswith('__')]
    for c in missing_cols_from_sir:
        unmatched_bd[c] = pd.NA
    unmatched_bd = unmatched_bd[[c for c in all_cols_union]]
    final_parts.append(unmatched_bd)

matched_sir_idxs = set()
if not matched_ticket.empty:
    matched_sir_idxs.update(matched_ticket['__sir_idx'].dropna().astype('Int64').tolist())
if not matched_fields_with_booking.empty and '__sir_idx' in matched_fields_with_booking.columns:
    matched_sir_idxs.update(matched_fields_with_booking['__sir_idx'].dropna().astype('Int64').tolist())
if not matched_fields_without_booking.empty and '__sir_idx' in matched_fields_without_booking.columns:
    matched_sir_idxs.update(matched_fields_without_booking['__sir_idx'].dropna().astype('Int64').tolist())

unmatched_sirena = sirena[~sirena['__sir_idx'].isin(matched_sir_idxs)].copy()
if not unmatched_sirena.empty:
    missing_cols_from_bd = [c for c in bd.columns if c not in unmatched_sirena.columns and not c.startswith('__')]
    for c in missing_cols_from_bd:
        unmatched_sirena[c] = pd.NA
    unmatched_sirena = unmatched_sirena[[c for c in all_cols_union]]
    final_parts.append(unmatched_sirena)

if final_parts:
    final = pd.concat(final_parts, ignore_index=True, sort=False)
else:
    final = pd.DataFrame(columns=all_cols_union)

for helper in ['__bd_idx', '__sir_idx']:
    if helper in final.columns:
        final = final.drop(columns=[helper])

final.to_csv(out_merged, index=False)