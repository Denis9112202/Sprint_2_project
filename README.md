data_utils.py:
функция preprocess_dataset осуществляет предобработку данных (с сиспользованием функции clean_text), токенизацию и сохранение датасетов в csv файлы
сохраняю чистые текстовые датасеты для предобученного трансформера и токенизированные датасеты для LSTM модели

lstm_model.py
описание класса LSTM модели LSTMTextGenerator
в ней кроме метода forward описан метод generate для генерации следующего токена

next_token_dataset.py
описание класса NextTokenDataset
padding здесь не реализую, он будет в  collate_fn (см. lstm_train.py)

lstm_train.py
кастомная функция collate_fn для DataLoader
осуществляет паддинг по максимальной длине тензора в батче

функция train_lstm_model
в ней осуществляется
          - обучение модели на датасете TRAIN,
          - расчет LOSS, метрик ROUGE на датасете VAL (после каждой эпохи),
          - 10 случайных примеров из датасета TEST (после каждой эпохи).

eval_lstm.py
функция evaluate_lstm реализует формирование входных данных для модели (из тензора берем первые 3/4), формируется список ответов(targets) и с помощью метода generate модели рассчитывается список предсказаний predictions
функция calculate_rouge_scores рассчитывает метрики rouge (для стандартной из evaluate нужны списки слов (в данном случае работаю с токенами))

eval_transformer_pipeline.py
функция evaluate_transformer реализует генерацию текстов с помощью модели 'distilgpt2' по алгоритму как и evaluate_lstm
метрики rouge здесь рассчитываю штатным методом из библиотеки evaluate


ВЫВОДЫ:
Судя по метрикам да и по наглядным примерам немного лидирует модель 'distilgpt2' 
Но по наглядным примерам на последних эпохах обучения LSTM модель начинает предсказывать более точно. Даже в малой выборке примеров попадаются точные предсказания:
--- Пример 8 из 2 эпохи---
Вход: its that whole roid thing that makes *** me cringe
Предсказание: its that whole roid thing that makes *** me smile at work

--- Пример 7 из 7 эпохи ---
Вход: mission statement or something like that psyk *** ##oidcom
Предсказание: mission statement or something like that psyk *** ##oidcom

--- Пример 4 из 8 эпохи ---
Вход: its gone morning how are *** you today
Предсказание: its gone morning how are *** you doing

--- Пример 8 из 8 эпохи ---
Вход: i cud never buy those toys somehow my dad thought it was a wastehe used *** to get me robots and cars
Предсказание: i cud never buy those toys somehow my dad thought it was a wastehe used *** to play

--- Пример 5 из 10 эпохи ---
Вход: i still dont want to go *** to school
Предсказание: i still dont want to go *** back to work
