import pdfplumber
import json
from tqdm.auto import tqdm
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_data_path = CURRENT_DIR + "/Airlines/Skyteam_Timetable.pdf"


def extract_half(page: pdfplumber.pdf.Page, bbox):
    half = page.crop(bbox)
    text = half.extract_text()
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
    if (
        len(lines) > 1 # может быть продолжение предыдущей в 1 строку
        and lines[1] == "FROM:"
    ):
        # если нет записей
        if lines[6] == "Consult your travel agent for details":
            return None

        data["from"] = lines[0]
        data["to"] = lines[2]
        data["columns"] = lines[4]
        data["rows"] = lines[6:]
    # если продолжение предыдущей страницы
    else: 
        data["rows"] = lines

    return data


def parse_pdf(path: str, start_page: int):

    pair_counter = 0
    results = [None, None]

    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        print(f"Всего страниц: {n_pages}")

        # все страницы одинаковые
        page = pdf.pages[start_page]
        width, height = page.width, page.height
        left_bbox = (0, 0, width / 2, height)
        right_bbox = (width / 2, 0, width, height)

        pages_tqdm = tqdm(
            pdf.pages[start_page:], 
            total=n_pages - start_page, 
            desc="Обработка страниц", 
            unit="стр"
        )
        for page in pages_tqdm:
            left = extract_half(page, left_bbox)
            right = extract_half(page, right_bbox)

            # если новый перелёт
            if left and right and left["columns"] and right["columns"]:
                # сохраняем предыдущую пару, если есть
                if results[0] and results[1]:
                    with open(
                        CURRENT_DIR + f"/SkyteamPdf_jsons/dest_pair{pair_counter}.json", 
                        "w", 
                        encoding="utf-8"
                    ) as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

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
                json.dump(results, f, ensure_ascii=False, indent=2)

    return results


parse_pdf(
    pdf_data_path, 
    start_page=4 # pdf page - 1
) 


# для тестов и анализа структуры результата
# with pdfplumber.open(pdf_data_path) as pdf:
#     page = pdf.pages[23835]

#     width, height = page.width, page.height
#     left_bbox = (0, 0, width / 2, height)
#     right_bbox = (width / 2, 0, width, height)

#     left = page.crop(left_bbox)
#     text = left.extract_text()

#     lines = [l.strip() for l in text.splitlines()]
#     print(lines[1])


# Сценарии под pdfplumber

# Начало таблицы:
# ```
# Aalborg, Denmark AAL
# FROM:
# Amsterdam , Netherlands AMS
# TO:
# Validity Days Dep Arr Flight Aircraft Travel
# Time Time Time
# 01 Nov - 31 Jan 1234567 06:00 07:25 KL1328 73W 1H25M
# 01 Nov - 31 Jan 1234567 12:10 13:35 KL1334 73W 1H25M
# ```

# Продолжение таблицы:
# ```
# 01 Nov - 31 Jan 1234567 12:10 13:35 KL1334 73W 1H25M
# 01 Nov - 23 Dec 1234567 18:15 19:35 KL1336 EQV 1H20M
# ```

# Одна таблица может продолжаться, в другой может быть пустота

# Отсутствие записей:
# ```
# Aalborg, Denmark AAL
# FROM:
# Graz, Austria GRZ
# TO:
# Validity Days Dep Arr Flight Aircraft Travel
# Time Time Time
# Consult your travel agent for details
# ```

# Также возможен вариант с Operated by:
# ```
# 09 Nov - 16 Nov 6 09:00 10:25 DL7272* 73H 1H25M
# Operated by: V Australia
# 10 Nov - 26 Jan 7 09:00 10:25 DL7274* EQV 1H25M
# Operated by: V Australia
# ```