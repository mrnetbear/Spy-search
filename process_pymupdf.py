# Skyteam_Timetable.pdf -> SkyteamPdf_jsons/best_pair1{N}.json

import pymupdf
import json
from tqdm.auto import tqdm
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_data_path = CURRENT_DIR + "/Airlines/Skyteam_Timetable.pdf"


def extract_half(page: pymupdf.Page, clip):
    text = page.get_text("text", clip=clip)
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines()]

    data = {
        "from": None,
        "to": None,
        "columns": None,
        "rows": None
    }

    # если есть заголовок
    if "FROM:" in lines[0]:
        # если нет записей
        if lines[-1] == "Consult your travel agent for details":
            return None

        data["from"] = str.join(" ", lines[0:2])
        data["to"] = str.join(" ", lines[2:5])
        data["columns"] = str.join(" ", lines[5:15])
        data["rows"] = lines[15:]
    # если продолжение предыдущей страницы
    else: 
        data["rows"] = lines

    return data


def parse_pdf(path: str, start_page: int):

    pair_counter = 0
    results = [None, None]

    with pymupdf.open(path) as pdf:
        n_pages = len(list(pdf.pages()))
        print(f"Всего страниц: {n_pages}")

        # все страницы одинаковые
        page = pdf.load_page(start_page)
        width, height = page.rect.width, page.rect.height
        left_clip = pymupdf.Rect(0, 0, width / 2, height)
        right_clip = pymupdf.Rect(width / 2, 0, width, height)

        pages_tqdm = tqdm(
            pdf.pages(start_page, n_pages), 
            total=n_pages - start_page, 
            desc="Обработка страниц", 
            unit="стр"
        )
        for page in pages_tqdm:
            left = extract_half(page, left_clip)
            right = extract_half(page, right_clip)

            # если новый перелёт
            if left and right and left["columns"] and right["columns"]:
                # сохраняем предыдущую пару, если есть
                if results[0] and results[1]:
                    with open(
                        CURRENT_DIR + f"/SkyteamPdf_jsons/best_pair{pair_counter}.json", 
                        "w", 
                        encoding="utf-8"
                    ) as f:
                        json.dump(results, f, ensure_ascii=False)

                pair_counter += 1
                results[0] = left
                results[1] = right

            # начала записей на страницах синхронны, продолжения - нет
            elif left or right:
                if left:
                    results[0]["rows"].extend(left["rows"])
                if right:
                    results[1]["rows"].extend(right["rows"])
        
        # сохранить последнюю пару после завершения цикла
        if results[0] and results[1]:
            with open(
                CURRENT_DIR + f"/SkyteamPdf_jsons/dest_pair{pair_counter}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(results, f, ensure_ascii=False)

    return results


parse_pdf(
    pdf_data_path, 
    start_page=4 # pdf page - 1
)


# для тестов и анализа структуры результата
# with pymupdf.open(pdf_data_path) as pdf:
#     page = pdf.load_page(20681)

#     width, height = page.rect.width, page.rect.height
#     left_clip = pymupdf.Rect(0, 0, width / 2, height)
#     right_clip = pymupdf.Rect(width / 2, 0, width, height)

#     text = page.get_text("text", clip=left_clip)
#     print(text)


# Сценарии под pymupdf

# Начало таблицы:
# ```
# FROM: Aalborg, Denmark
# AAL
# TO:
# Amsterdam , Netherlands
# AMS
# Validity
# Days
# Dep
# Time
# Arr
# Time
# Flight
# Aircraft
# Travel
# Time
# 01 Nov  -  31 Jan
# 1234567 06:00
# 07:25
# KL1328
# 73W
# 1H25M
# 01 Nov  -  31 Jan
# 1234567 12:10
# 13:35
# KL1334
# 73W
# 1H25M
# ```

# Продолжение таблицы:
# ```
# 01 Nov  -  31 Jan
# 1234567 12:10
# 13:35
# KL1334
# 73W
# 1H25M
# 01 Nov  -  23 Dec
# 1234567 18:15
# 19:35
# KL1336
# EQV
# 1H20M
# ```

# Одна таблица может продолжаться, в другой может быть пустота

# Отсутствие записей:
# ```
# FROM: Abbotsford , Canada
# YXX
# TO:
# Calgary , Canada
# YYC
# Validity
# Days
# Dep
# Time
# Arr
# Time
# Flight
# Aircraft
# Travel
# Time
# Consult your travel agent for details
# ```

# Также возможен вариант с Operated by:
# ```
# FROM: Naples , Italy
# NAP
# TO:
# Bucharest , Romania
# OTP
# Validity
# Days
# Dep
# Time
# Arr
# Time
# Flight
# Aircraft
# Travel
# Time
# 07 Nov  -  30 Jan
#       4
# 19:25
# 22:20
# AZ7400*
# 738
# 1H55M
# Operated by:  Airline 0B
# ```