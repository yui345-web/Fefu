import logging
import tensorflow as tf
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import ParseMode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Вставьте сюда свой токен бота
TOKEN = "твой токен"

# Настройки бота
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot)

# Укажите свой путь до обученной модели
model = tf.keras.models.load_model(r"путь до модели")

# Настройки токенизатора
VOCAB_SIZE = 20000
MAX_LEN = 25

# Создаем токенизатор 
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(["Hello!", "I am happy", "I feel sad", "What a surprise!"])

# Классы эмоций 
label_map = {0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "love", 5: "surprise"}

# Функция предсказания
def predict_sentiment(text):
    sequence = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding="post")
    prediction = model.predict(sequence)
    predicted_class = np.argmax(prediction)
    return label_map[predicted_class]

# Обработчик команды /start
@dp.message_handler(commands=["start"])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь мне любое сообщение, и я определю его настроение!")

# Обработчик текстовых сообщений
@dp.message_handler()
async def analyze_message(message: types.Message):
    sentiment = predict_sentiment(message.text)
    await message.reply(f"🌟 <b>Предсказанное настроение:</b> {sentiment}")

# Запуск бота
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    executor.start_polling(dp, skip_updates=True)
