import logging
import tensorflow as tf
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import ParseMode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Вставь свой токен бота
TOKEN = "8194961676:AAENyYv0SV_oi31MzAGAsd8XRSp5tMyFE_M"

# Настройки бота
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot)

# Вставь путь до модели
model = tf.keras.models.load_model(r"путь до модели")

# Настройки токенизатора
VOCAB_SIZE = 20000
MAX_LEN = 25

# Создаем токенизатор 
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(["Hello!", "I am happy", "I feel sad", "What a surprise!"])

# Классы
dataset_labels = ["joy", "sadness", "anger", "fear", "love", "surprise"]

# Объединение классов в 4 настроения
label_map = {
    "Радостный": ["joy", "love"],
    "Грустный": ["sadness"],
    "Негативный": ["anger", "fear"],
    "Нейтральный": ["surprise"]
}

# Функция предсказания настроения
def predict_sentiment(text):
    # Токенизация и предсказание
    sequence = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding="post")
    prediction = model.predict(sequence)[0]  # Получаем массив вероятностей

    # Преобразуем вероятности 6 классов в 4
    sentiment_probs = {sentiment: sum(prediction[dataset_labels.index(class_name)] for class_name in class_group)
                       for sentiment, class_group in label_map.items()}

    # Определяем самый вероятный класс
    predicted_sentiment = max(sentiment_probs, key=sentiment_probs.get)

    # Формируем текст с вероятностями
    probs_text = "\n".join([f"{sentiment}: {round(prob * 100, 2)}%" for sentiment, prob in sentiment_probs.items()])

    return predicted_sentiment, probs_text

# Обработчик команды /start
@dp.message_handler(commands=["start"])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь мне любое сообщение, и я определю его настроение!")

# Обработчик текстовых сообщений
@dp.message_handler()
async def analyze_message(message: types.Message):
    sentiment, probs = predict_sentiment(message.text)
    await message.reply(f"🌟 <b>Предсказанный класс:</b> {sentiment}\n\n📊 <b>Вероятности:</b>\n{probs}")

# Запуск бота
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    executor.start_polling(dp, skip_updates=True)
