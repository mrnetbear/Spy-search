import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Загрузка данных
df1 = pd.read_csv('FrequentFlyer71993_flights.csv')
df2 = pd.read_csv('FrequentFlyer29897_flights.csv')

# Добавляем идентификатор для каждого пользователя
df1['user'] = 'FrequentFlyer71993 (Головина Олеся)'
df2['user'] = 'FrequentFlyer29897'

# Объединяем данные
df = pd.concat([df1, df2], ignore_index=True)

# Удаляем дубликаты перелётов
df_unique = df.drop_duplicates(['user', 'DATE', 'DEPARTURE_CITY', 'ARRIVAL_CITY'])

# Словарь для преобразования названий стран в ISO-3 коды
country_to_iso3 = {
    'Russian Federation': 'RUS',
    'China': 'CHN',
    'Iran': 'IRN',
    'Ko': 'KOR',  # Южная Корея
    'Saudi Arabia': 'SAU',
    'Czech Republic': 'CZE',
    'United Arab Emirates': 'ARE',
    'Lebanon': 'LBN',
    'Italy': 'ITA',
    'Albania': 'ALB',
    'Spain and Canary Islands': 'ESP',
    'Russian': 'RUS',  # Исправляем для Saint Petersburg
}

# Добавляем ISO-3 коды в DataFrame
df_unique['ARRIVAL_COUNTRY_ISO'] = df_unique['ARRIVAL_COUNTRY'].map(country_to_iso3)
df_unique['DEPARTURE_COUNTRY_ISO'] = df_unique['DEPARTURE_COUNTRY'].map(country_to_iso3)

# Создаем базовую карту с точками прибытия (используя ISO-3 коды)
fig = px.scatter_geo(df_unique, 
                     locations="ARRIVAL_COUNTRY_ISO", 
                     locationmode='ISO-3',
                     color="user",
                     hover_name="ARRIVAL_CITY",
                     hover_data=["DATE", "FLIGHT", "DEPARTURE_CITY"],
                     projection="natural earth",
                     title="Маршруты перелётов Frequent Flyers",
                     size_max=15)

# Увеличиваем размер точек
fig.update_traces(marker=dict(size=8))
fig.show()

# Создаем детальную карту с маршрутами
print("Создаем детальную карту с маршрутами...")

# Словарь координат для стран (долгота, широта)
country_coords = {
    'RUS': [37.618423, 55.751244],  # Москва, Россия
    'CHN': [116.407396, 39.904211],  # Пекин, Китай
    'IRN': [51.3890, 35.6892],  # Тегеран, Иран
    'KOR': [126.977969, 37.566535],  # Сеул, Южная Корея
    'SAU': [46.675296, 24.713552],  # Эр-Рияд, Саудовская Аравия
    'CZE': [14.437800, 50.075539],  # Прага, Чехия
    'ARE': [55.296249, 25.276987],  # Дубай, ОАЭ
    'LBN': [35.5018, 33.8938],  # Бейрут, Ливан
    'ITA': [12.496366, 41.902782],  # Рим, Италия
    'ALB': [19.8189, 41.3275],  # Тирана, Албания
    'ESP': [-3.703790, 40.416775],  # Мадрид, Испания
}

# Создаем новую карту с линиями маршрутов
fig2 = go.Figure()

# Цвета для пользователей
colors = {
    'FrequentFlyer71993 (Головина Олеся)': 'blue', 
    'FrequentFlyer29897': 'red'
}

# Для каждого пользователя
for user in df_unique['user'].unique():
    user_data = df_unique[df_unique['user'] == user].sort_values('DATE')
    
    # Добавляем маршруты
    first_route = True  # Флаг для отображения в легенде
    for idx, row in user_data.iterrows():
        dep_iso = row['DEPARTURE_COUNTRY_ISO']
        arr_iso = row['ARRIVAL_COUNTRY_ISO']
        
        if dep_iso in country_coords and arr_iso in country_coords:
            dep_lon, dep_lat = country_coords[dep_iso]
            arr_lon, arr_lat = country_coords[arr_iso]
            
            # Добавляем линию маршрута
            fig2.add_trace(go.Scattergeo(
                lon = [dep_lon, arr_lon],
                lat = [dep_lat, arr_lat],
                mode = 'lines',
                line = dict(width = 3, color = colors[user]),
                opacity = 0.8,
                name = user,
                legendgroup = user,
                showlegend = first_route,  # Показывать легенду только для первого маршрута
                hoverinfo = 'text',
                text = f"{row['DEPARTURE_CITY']} → {row['ARRIVAL_CITY']}<br>Дата: {row['DATE']}<br>Рейс: {row['FLIGHT']}",
                hoverlabel = dict(bgcolor=colors[user], font_color='white')
            ))
            first_route = False  # После первого маршрута не показывать в легенде

# Добавляем точки городов
visited_countries = set()
first_city = True  # Флаг для отображения в легенде
for user in df_unique['user'].unique():
    user_data = df_unique[df_unique['user'] == user]
    for country_iso in user_data['ARRIVAL_COUNTRY_ISO'].unique():
        if country_iso in country_coords and country_iso not in visited_countries:
            visited_countries.add(country_iso)
            lon, lat = country_coords[country_iso]
            
            # Получаем оригинальное название страны
            country_name = [k for k, v in country_to_iso3.items() if v == country_iso][0]
            
            fig2.add_trace(go.Scattergeo(
                lon = [lon],
                lat = [lat],
                mode = 'markers',
                marker = dict(size=10, color='green', symbol='circle'),
                name = 'Города',
                legendgroup = 'cities',
                showlegend = first_city,  # Только для первого города
                hoverinfo = 'text',
                text = country_name,
                hoverlabel = dict(bgcolor='green', font_color='white')
            ))
            first_city = False

# Настраиваем карту
fig2.update_layout(
    title_text = '✈️ Детальная карта маршрутов перелётов',
    showlegend = True,
    geo = dict(
        scope = 'world',
        projection_type = 'equirectangular',
        showland = True,
        landcolor = 'rgb(240, 240, 240)',
        countrycolor = 'rgb(200, 200, 200)',
        coastlinecolor = 'rgb(160, 160, 160)',
        showocean = True,
        oceancolor = 'rgb(200, 230, 255)',
        showcountries = True,
        countrywidth = 1,
    ),
    width = 1200,
    height = 700,
    legend = dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

fig2.show()

# Выводим подробную статистику
print("\nДЕТАЛЬНАЯ СТАТИСТИКА ПЕРЕЛЁТОВ:")
print("=" * 60)

for user in df_unique['user'].unique():
    user_data = df_unique[df_unique['user'] == user].sort_values('DATE')
    
    # Собираем все уникальные страны и города
    all_countries = set(user_data['DEPARTURE_COUNTRY_ISO']).union(set(user_data['ARRIVAL_COUNTRY_ISO']))
    all_cities = set(user_data['DEPARTURE_CITY']).union(set(user_data['ARRIVAL_CITY']))
    
    print(f"\n {user}:")
    print(f"Всего перелётов: {len(user_data)}")
    print(f"Период путешествий: {user_data['DATE'].min()} - {user_data['DATE'].max()}")
    print(f"Посещено стран: {len(all_countries)}")
    print(f"Уникальные города: {len(all_cities)}")
    
    # Анализ codeshare рейсов
    codeshare_flights = user_data[user_data['CODESHARE'] == True]
    print(f"Codeshare рейсов: {len(codeshare_flights)}")