import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from pathlib import Path


st.set_page_config(page_title='Предсказание цен на автомобили',
                   page_icon='🚘',
                   layout='wide')



BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / 'pipelines' / 'ridge_pipeline.pkl'
DATA_PATH = BASE_DIR / 'data' / 'train_car_prices.csv'
NUM_FEATURES = ['year', 'km_driven']
CAT_FEATURES = ['fuel', 'seller_type', 'transmission', 'owner']
FEATURES_WITH_TEXT = ['mileage', 'engine', 'max_power']
SEATS_COL = ['seats']
NAME_COL = ['name']
ALL_COLS = NUM_FEATURES + CAT_FEATURES + FEATURES_WITH_TEXT + SEATS_COL + NAME_COL

def extract_brand_name(X):
  X = pd.DataFrame(X).copy()
  X.iloc[:, 0] = X.iloc[:, 0].apply(lambda x: x.lower().split()[0])
  return X

def convert_dtypes(X, data_type):
  X = pd.DataFrame(X).copy()
  if isinstance(data_type, list):
    for i, col in enumerate(X.columns):
      X[col] = pd.to_numeric(X[col], downcast=data_type[i], errors='coerce')
    return X

  for col in X.columns:
      X[col] = pd.to_numeric(X[col], downcast=data_type, errors='coerce')
  return X

def extract_number(X):
  X = pd.DataFrame(X).copy()
  for col in X.columns:
    if X[col].dtype == 'object':
      X[col] = X[col].str.replace(r'[^\d\.]', '', regex=True)
  return X

@st.cache_resource
def load_pipeline():
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

@st.cache_data
def load_data(file):
    return pd.read_csv(file)



st.title('Предсказание цен на автомобили')

try:
   PIPELINE = load_pipeline()
except Exception as e:
   st.error(f'❌ Ошибка при загрузке пайплайна: {e}')
   st.stop()


tab1, tab2 = st.tabs(['**EDA и коэффициенты модели**',
                      '**Применение модели**'])

with tab1:
    df_train = load_data(DATA_PATH)
    st.header('🔹 Тренировочные данные')
    st.dataframe(df_train.head())

    st.markdown('''## 🔹Описание признаков
- **name** — модель автомобиля
- **year** — год выпуска
- **selling_price** — цена автомобиля (целевая переменная)
- **km_driven** — пробег
- **fuel** — вид топлива
- **seller_type** — продавец
- **transmission** — тип коробки передач
- **owner** — предыдущие владельцы
- **mileage** — топливная экономичность двигателя
- **engine** — объём двигателя
- **max_power** — максимальная мощность двигателя
- **torque** — крутящий момент двигателя
- **seats** — количество мест в автомобиле            
''')
    

    st.header('🔹 Основные визуализации')

    col_1, col_2 = st.columns(2)
    col_3, col_4 = st.columns(2)
    
    with col_1:
       fig_1 = px.scatter(df_train, x='max_power', y='selling_price',
                         title='Цена (selling_price) — максимальная мощность (max_power)')
       st.plotly_chart(fig_1, use_container_width=True)

    with col_2:
       fig_2 = px.scatter(df_train, x='max_power', y='engine',
                         title='Объём двигателя (engine) — максимальная мощность (max_power)')
       st.plotly_chart(fig_2, use_container_width=True)
    
    with col_3:
       fig_3 = px.scatter(df_train, x='max_power', y='mileage',
                         title='Пробег (mileage) — максимальная мощность (max_power)')
       st.plotly_chart(fig_3, use_container_width=True)

    with col_4:
       fig_4 = px.scatter(df_train, x='mileage', y='engine',
                         title='Объём двигателя (engine) — пробег (mileage)')
       st.plotly_chart(fig_4, use_container_width=True)
   

    median_value = df_train['selling_price'].median()
    mean_value = df_train['selling_price'].mean()
    
    fig_5 = px.histogram(df_train, x='selling_price', nbins=40,
                         title='Распределение целевой переменной (selling_price)')
    
    fig_5.add_vline(x=median_value, line_color='pink', annotation_position='top left',
                    annotation_text=f'Медиана: {median_value:.2f}')
    
    fig_5.add_vline(x=mean_value, line_color='orange', annotation_position='top right',
                    annotation_text=f'Среднее: {mean_value:.2f}')
    
    st.plotly_chart(fig_5, use_container_width=True)


    fig_6 = px.box(df_train, x='fuel', y='selling_price', height=800,
                   title='Распределение цен автомобилей (selling price) в зависимости от типа топлива (fuel)')
    
    st.plotly_chart(fig_6, use_container_width=True)


    RIDGE_COEFFS = PIPELINE.named_steps['regressor'].coef_
    FEATURE_NAMES = PIPELINE.named_steps['preprocessor'].get_feature_names_out()

    fig_7 = px.histogram(x=RIDGE_COEFFS, nbins=40,
                         title='Распределение коэффициентов модели')
    
    st.plotly_chart(fig_7, use_container_width=True)

    st.write('**Коэффициенты модели:**')
    st.dataframe(pd.Series(RIDGE_COEFFS, index=FEATURE_NAMES, name='weights').sort_values(key=abs, ascending=False))
     


with tab2:
    input_method = st.radio('Выберите способ ввода данных:',
                            ('Загрузка файла', 'Ручной ввод'))
    if input_method == 'Загрузка файла':
       delimiter = st.selectbox('Выберите разделитель в CSV файле:',
                                (',', ';', ':', r'\t', '|'))
       uploaded_file = st.file_uploader('Загрузите CSV файл', type=['csv'])
       if uploaded_file:
          try:
             df_from_csv = pd.read_csv(uploaded_file, sep=delimiter)
          except Exception as e:
             st.error(f'❌ Ошибка при чтении CSV файла: {e}')
             st.stop()
          else:
             df_from_csv.columns = df_from_csv.columns.str.lower()
             if not set(ALL_COLS).issubset(df_from_csv.columns):
                st.error(f'❌ В CSV файле отсутствуют столбцы: {set(ALL_COLS).difference(df_from_csv.columns)}')
                st.stop()
             try:
                predictions = PIPELINE.predict(df_from_csv)
             except Exception as e:
                st.error(f'❌ Ошибка при обработке данных: {e}')
                st.stop()
             else:
                st.success('🔎 **Результаты**')
                df_from_csv['predicted_price'] = predictions
                st.dataframe(df_from_csv[['name', 'predicted_price']])


    if input_method == 'Ручной ввод':
       input_data = {}

       with st.form('prediction_form'):
          input_data['name'] = st.text_input('name', value='Car model', key='name')
          for col in NUM_FEATURES + SEATS_COL + ['engine']:
             val = df_train[col].median().astype(int)
             input_data[col] = st.number_input(col, value=val, min_value=1, key=f'{col}')
 
          for col in ('mileage', 'max_power'):
             val = df_train[col].median()
             input_data[col] = st.number_input(col, value=val, min_value=1.0, key=f'{col}')

          for col in CAT_FEATURES:
             unique_vals = df_train[col].unique()
             input_data[col] = st.selectbox(col, unique_vals, key=f'{col}')
         
          submitted = st.form_submit_button('Спрогнозировать цену')
          
          if submitted:
             if not input_data['name']:
                st.warning('❗ Введите название модели автомобиля')
                st.stop()
             try:
                df_from_inp = pd.DataFrame([input_data])
                prediction = PIPELINE.predict(df_from_inp)
             except Exception as e:
                st.error(f'❌ Ошибка при обработке данных: {e}')
                st.stop()
             else:
                 st.success('🔎 **Результаты**')
                 df_from_inp['predicted_price'] = prediction
                 st.dataframe(df_from_inp[['name', 'predicted_price']])

