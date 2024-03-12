# Задание

Даны 2 датасета: train и test (лежат в папке data).

Поля датасета:
* search_id - айди поиска (айди группы, которая связана с набором объектов)
* features_0-78 - признаки этих объектов
* target - целевая переменная (допустим клик по объекту)

Задача:
* обучить модель для релевантной выдачи (можно выбрать любой алгоритм, библиотеку или фреймворк)
* посчитать метрику NDCG (для всех документов, не конкретного топа) на тестовом датасете
* выложить код на github отдельным проектом и поделиться ссылкой в поле для ответа

Будет плюсом, если решение завернуть в docker.

# Реализация

## Шаги решения
1. Посмотрел на данные, убрал частей признаков и постарался разобраться с дисбалансом классов
2. Сделал полносвязную сеть на `Pytorch Lightning` с BCE лоссом, обучил её
3. Добавил логирование модели
4. Прогнал модель с разными гипер параметрами, максимальный на тесте NDCG = 0.57
5. Сделал API сервис для модели
6. Завернул его в контейнер
7. Добавил всякое полезное для сервиса (см. технологии)

## Структура проекта
<pre>
.
├── Dockerfile (<- docker для запуска API модели)
├── Makefile 
├── README.md (<- вы тут)
├── app (<- backend сервис)
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── model.pt
│   ├── requirements.txt  (<- серверные зависимости для запуска в докере)
│   ├── scaler_info.json
│   └── schema.py
├── model (<- ML модуль)
│   ├── config.py (<- конфиг с настройками)
│   ├── data
│   │   ├── custom_test_df.csv (<- test датасет после EDA)
│   │   ├── custom_train_df.csv (<- train датасет после EDA)
│   │   ├── test_df.csv
│   │   └── train_df.csv
│   ├── dataset.py
│   ├── eda.ipynb (<- тетрадка с eda)
│   ├── model.py (<- модел)
│   ├── model_training.ipynb (<- тетрадка с обучением модели)
│   └── scaler_serializator.py
└── requirements.txt (<- обязательные зависимости)
</pre>

## Технологии
* `PyTorch` и `PyTorch Lightning` для создания модели
* `TensorBoard` для анализа модели
* `MLFlow` для менеджмента моделей
* `FastAPI` для API модели
* `Docker` для контейризации
* и еще всякое сверху: `Makefile`, `Swagger`, `dotenv`, etc...

## Запускаем!
1. Клонируем проект: `git clone https://github.com/evlko/VK-MLE.git`
2. Подтягиваем зависимости: `pip install -r requirements.txt`
3. Билдим: `make docker_build`
4. Запускаем: `make docker_run`
5. Проверяем: `localhost:3000/health`
6. Swagger: `localhost:3000/docs`
7. Model API: `localhost:3000/predict`