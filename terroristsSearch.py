import pandas as pd
from airports import Airport
import csv
from datetime import datetime, timedelta
import os

def get_airport_info(iata_code):
    """Получить информацию об аэропорте по IATA коду"""
    try:
        airport = Airport(iata_code)
        if airport:
            return {
                'city': airport.city,
                'country': airport.country
            }
    except:
        pass
    return {'city': None, 'country': None}

def unify_airport_data():
    """Восстановление данных об аэропортах"""
    print("Задача 1: Восстановление данных об аэропортах...")
    
    # Чтение исходного файла
    input_file = "csv/RESULT/MergedResult-cleared.csv"
    output_file = "csv/RESULT/MergedResult-cleared-unify.csv"
    
    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} записей")
    
    # Создаем словари для кэширования результатов запросов
    airport_cache = {}
    
    # Функция для обработки одного аэропорта
    def process_airport(iata_code, current_city, current_country, field_type):
        if pd.isna(current_city) or current_city in ['0', '0.0', 0] or current_city == '':
            if iata_code not in airport_cache:
                airport_cache[iata_code] = get_airport_info(iata_code)
            return airport_cache[iata_code][field_type]
        return current_city
    
    # Восстанавливаем данные для DepartureCity и DepartureCountry
    print("Обработка аэропортов вылета...")
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Обработано {i}/{len(df)} записей")
        
        departure_airport = str(row['From']).strip()
        if departure_airport and departure_airport != '0':
            df.at[i, 'DepartureCity'] = process_airport(
                departure_airport, row['DepartureCity'], row['DepartureCountry'], 'city'
            )
            df.at[i, 'DepartureCountry'] = process_airport(
                departure_airport, row['DepartureCountry'], row['DepartureCountry'], 'country'
            )
    
    # Восстанавливаем данные для ArrivalCity и ArrivalCountry
    print("Обработка аэропортов прибытия...")
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Обработано {i}/{len(df)} записей")
        
        arrival_airport = str(row['Dest']).strip()
        if arrival_airport and arrival_airport != '0':
            df.at[i, 'ArrivalCity'] = process_airport(
                arrival_airport, row['ArrivalCity'], row['ArrivalCountry'], 'city'
            )
            df.at[i, 'ArrivalCountry'] = process_airport(
                arrival_airport, row['ArrivalCountry'], row['ArrivalCountry'], 'country'
            )
    
    # Сохраняем результат
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"Файл сохранен: {output_file}")
    return df

def find_passengers_in_terror_events():
    """Поиск пассажиров, связанных с террористическими актами"""
    print("\nЗадача 2: Поиск пассажиров, связанных с террористическими актами...")
    
    # Чтение файла с террористическими актами
    terracts_file = "csv/RESULT/Data/Terracts.csv"
    terracts_df = pd.read_csv(terracts_file)
    print(f"Загружено {len(terracts_df)} событий")
    
    # Чтение унифицированного файла с пассажирами
    passengers_file = "csv/RESULT/MergedResult-cleared-unify.csv"
    passengers_df = pd.read_csv(passengers_file)
    
    # Преобразование дат в datetime
    passengers_df['FlightDate'] = pd.to_datetime(passengers_df['FlightDate'])
    passengers_df['ArrivalDate'] = pd.to_datetime(passengers_df['ArrivalDate'], errors='coerce')
    
    # Создаем папку для результатов, если её нет
    output_dir = "csv/RESULT/Terract_Passengers"
    os.makedirs(output_dir, exist_ok=True)
    
    # Обрабатываем каждое событие
    for idx, event in terracts_df.iterrows():
        print(f"Обработка события {idx+1}/{len(terracts_df)}: {event['city']}, {event['country']}")
        
        # Парсим даты события
        start_date = pd.to_datetime(event['startDate'])
        end_date = pd.to_datetime(event['endDate'])
        
        # Определяем временной диапазон (2 дня до начала + сам день + 2 дня после завершения)
        search_start = start_date - timedelta(days=2)
        search_end = end_date + timedelta(days=2)
        
        # Получаем код аэропорта события
        event_airport = event['airport']
        if pd.isna(event_airport) or event_airport == '':
            print(f"  Пропуск события: отсутствует код аэропорта")
            continue
        
        # Ищем пассажиров, которые прилетели или вылетели из этого аэропорта в указанный период
        # Прилеты (Dest = аэропорт события)
        arrivals = passengers_df[
            (passengers_df['Dest'] == event_airport) & 
            (passengers_df['FlightDate'] >= search_start) & 
            (passengers_df['FlightDate'] <= search_end)
        ].copy()
        
        # Вылеты (From = аэропорт события)
        departures = passengers_df[
            (passengers_df['From'] == event_airport) & 
            (passengers_df['FlightDate'] >= search_start) & 
            (passengers_df['FlightDate'] <= search_end)
        ].copy()
        
        # Добавляем тип события
        arrivals['EventType'] = 'Arrival'
        departures['EventType'] = 'Departure'
        
        # Объединяем результаты
        combined = pd.concat([arrivals, departures], ignore_index=True)
        
        if not combined.empty:
            # Добавляем информацию о событии
            combined['Terract_City'] = event['city']
            combined['Terract_Country'] = event['country']
            combined['Terract_Airport'] = event['airport']
            combined['Terract_StartDate'] = event['startDate']
            combined['Terract_EndDate'] = event['endDate']
            combined['Terract_Classification'] = event['classification']
            
            # Сохраняем в отдельный файл
            event_filename = f"{event['city']}_{event['country']}_{event['startDate']}".replace(' ', '_').replace('/', '-')
            output_file = f"{output_dir}/{event_filename}.csv"
            
            combined.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
            print(f"  Найдено {len(combined)} пассажиров. Сохранено в: {output_file}")
        else:
            print(f"  Пассажиры не найдены для события в {event['city']}, {event['country']}")
    
    print("Задача 2 завершена!")

def main():
    """Основная функция"""
    print("Начало обработки данных...")
    
    try:
        # Задача 1: Восстановление данных об аэропортах
        unified_df = unify_airport_data()
        
        # Задача 2: Поиск пассажиров, связанных с террористическими актами
        find_passengers_in_terror_events()
        
        print("\nВсе задачи успешно завершены!")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()