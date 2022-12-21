# Deep Learning in Audio. Домашнее задание по HiFiGAN

## Инструкция по запуску

1. Клонируем репозиторий.
    ```
    git clone https://github.com/Markfryazino/HiFiGAN.git
    ```

1. Поднимаем контейнер. `WANDB_API_KEY` можно не указывать. Все операции далее, разумеется происходят внутри контейнера.
    ```
    cd TTS
    docker build --network host -t tts .
    docker run -it --network host -e WANDB_API_KEY=YOUR_API_KEY -v ./TTS:/repos/tts_project tts
    ```

1. Скачиваем данные (LJSpeech и MarkovkaSpeech - 3 аудио для инференса). Там же генерируем мелспеки для MarkovkaSpeech.
    ```
    bash scripts/prepare_data.sh
    ```

1. Скачиваем чекпоинт модели.
    ```
    python3 scripts/download_model.py
    ```

1. Теперь можно запустить модель. Для этого есть скрипт [test.py](./test.py). У него там много всяких параметров, но достаточно сделать просто:
    ```
    python3 test.py
    ```
    Этот запуск создаст папку `data/MarkovkaSpeech/generated` и положит туда вавки для примеров из MarkovkaSpeech.

1. Чтобы вопроизвести обучение, можно запустить [run_training.py](./run_training.py).
    ```
    python3 run_training.py
    ```

## Немного про содержание репозитория

Пакет с моделью и всеми содержательными классами лежит в папке [src](./src). Там есть папочка [models](./src/models) с моделями, файлик [config.py](./src/config.py) со всеми параметрами и файлик [train.py](./src/train.py) с обучением. Всё остальное менее интересно.