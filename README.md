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
Для LSTM:
Epoch [1/10], TrainLoss: 5.1616
rouge1: 0.0306 rouge2: 0.0025 rougeL: 0.0306
Epoch [2/10], TrainLoss: 4.7214
rouge1: 0.0406 rouge2: 0.0042 rougeL: 0.0406
Epoch [3/10], TrainLoss: 4.6088
rouge1: 0.0379 rouge2: 0.0019 rougeL: 0.0378
Epoch [4/10], TrainLoss: 4.5458
rouge1: 0.0392 rouge2: 0.0030 rougeL: 0.0390
Epoch [5/10], TrainLoss: 4.5034
rouge1: 0.0403 rouge2: 0.0038 rougeL: 0.0402
Epoch [6/10], TrainLoss: 4.4722
rouge1: 0.0400 rouge2: 0.0037 rougeL: 0.0399
Epoch [7/10], TrainLoss: 4.4478
rouge1: 0.0411 rouge2: 0.0037 rougeL: 0.0411
Epoch [8/10], TrainLoss: 4.4281
rouge1: 0.0400 rouge2: 0.0027 rougeL: 0.0399
Epoch [9/10], TrainLoss: 4.4117
rouge1: 0.0437 rouge2: 0.0050 rougeL: 0.0436
Epoch [10/10], TrainLoss: 4.3978
rouge1: 0.0397 rouge2: 0.0035 rougeL: 0.0397

Для 'distilgpt2': 
rouge1: 0.0536 rouge2: 0.0048 rougeL: 0.0504
rougeLsum: 0.0505

Но по наглядным примерам на последних эпохах обучения LSTM модель начинает предсказывать более точно по сравнению с превыми эпохами. Даже в малой выборке примеров попадаются точные предсказания:
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
