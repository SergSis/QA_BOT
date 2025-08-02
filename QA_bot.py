import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import csv
from typing import List, Dict
import os

from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate


import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- 1. ПАРСИНГ И СОХРАНЕНИЕ ДАННЫХ ---
# -----------------------------------------------------------------------------

def add_space_between_case(text: str) -> str:
    """Добавляет пробел между строчной и заглавной буквой."""
    return re.sub(r'([a-zа-яё])([A-ZА-ЯЁ])', r'\1 \2', text)

def extract_text_from_element(element) -> str:
    """Извлекает весь текст из элемента и его дочерних тегов p и li."""
    texts = []
    
    # Добавляем текст самого элемента
    if element.name and element.name not in ['p', 'li', 'h1', 'h2', 'h3', 'h4', 'span']:
        text_content = element.get_text(strip=True)
        if text_content:
            texts.append(add_space_between_case(text_content))

    # Находим все параграфы и элементы списка внутри
    child_elements = element.find_all(['p', 'li', 'span'], recursive=True)
    for child in child_elements:
        text_content = child.get_text(strip=True)
        if text_content:
            texts.append(add_space_between_case(text_content))

    return ' '.join(texts)

def parse_by_headings(soup: BeautifulSoup) -> Dict[str, str]:
    """Извлекает заголовки и соответствующий текст со страницы."""
    data = {}
    sections = soup.find_all(['h1', 'h2', 'h3', 'h4'])
    
    for section in sections:
        title = add_space_between_case(section.get_text(strip=True))
        if not title:
            continue
            
        content_text = []
        next_element = section.find_next_sibling()
        
        while next_element and next_element.name not in ['h1', 'h2', 'h3', 'h4', 'script', 'style']:
            if isinstance(next_element, NavigableString):
                text = next_element.strip()
                if text:
                    content_text.append(add_space_between_case(text))
            elif isinstance(next_element, Tag):
                extracted_content = extract_text_from_element(next_element)
                if extracted_content:
                    content_text.append(extracted_content)
            
            next_element = next_element.find_next_sibling()
        
        full_content = ' '.join(content_text)
        if full_content:
            data[title] = full_content
            
    return data

def parse_by_classes(soup: BeautifulSoup, classes_to_parse: List[str]) -> Dict[str, str]:
    """Извлекает данные из элементов с указанными CSS-классами."""
    data = {}
    
    for class_name in classes_to_parse:
        sections = soup.find_all(class_=class_name)
        for section in sections:
            title_tag = section.find(['h1', 'h2', 'h3', 'h4'], recursive=True)
            title = add_space_between_case(title_tag.get_text(strip=True)) if title_tag else f"Секция по классу: {class_name}"
            
            content = extract_text_from_element(section)
            if content and content != title:
                data[title] = content
                
    return data

def parse_and_save_data(urls: List[str], classes_to_parse: List[str], output_file: str):
    """Выполняет парсинг и сохраняет данные в CSV."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Заголовок', 'Текст'])

        for url in urls:
            print(f"Парсинг URL: {url}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                combined_data = parse_by_headings(soup)
                combined_data.update(parse_by_classes(soup, classes_to_parse))

                if combined_data:
                    for title, content in combined_data.items():
                        writer.writerow([title, content])
                    print(f"Парсинг {url} завершен. Найдено {len(combined_data)} уникальных секций.")
                else:
                    print(f"На URL {url} не найдено подходящих данных.")

            except requests.exceptions.RequestException as e:
                print(f"Ошибка при получении данных с URL {url}: {e}")

TELEGRAM_BOT_TOKEN = "8359159245:AAGF0BF1OISB0r3aPHaXMhBLyYows8stWkc"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка базы данных и создание цепочки
print("Инициализация бота...")
csv_file = 'program_itmo_combined_full.csv'
documents = load_data_from_csv(csv_file)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index_itmo", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# --- 2. ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ ДЛЯ LANGCHAIN ---
# -----------------------------------------------------------------------------

def load_data_from_csv(file_path: str) -> List[Document]:
    """Загружает данные из CSV и создает объекты Document."""
    documents = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1]:
                title, content = row[0], row[1]
                full_content = f"{title}. {content}" if title else content
                documents.append(Document(page_content=full_content, metadata={"source": file_path, "title": title}))
    return documents
# --- ОСНОВНАЯ ЛОГИКА ---
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    urls = [
        "https://abit.itmo.ru/program/master/ai",
        "https://abit.itmo.ru/program/master/ai_product"
    ]
    classes_to_parse = [
        'Background_background__im0jM',
        'Information_block',
        'row',
        'Directions_directions',
        'AboutProgram_aboutProgram',
        'Team_team',
        'Admission_admission',
        'Opportunities_opportunities_',
        'Accordion_accordion__title'
    ]
    csv_file = 'program_itmo_combined_full.csv'

    # Шаг 1: Парсинг и сохранение данных
    parse_and_save_data(urls, classes_to_parse, csv_file)

    # Шаг 2: Загрузка данных в LangChain
    documents = load_data_from_csv(csv_file)
    print(f"\nЗагружено документов: {len(documents)}")

    # Шаг 3: Генерация эмбеддингов
    if not documents:
        print("Невозможно создать векторную базу данных, так как список документов пуст.")
        exit()

    print("Генерация эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Шаг 4: Создание векторной базы данных FAISS
    print("Создание векторной базы данных FAISS...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss_index_itmo")
    print("Векторная база данных успешно создана и сохранена в 'faiss_index_itmo'.")

    # Шаг 5: Создание цепочки LangChain
    # --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
    
    # Путь к новой модели Mistral-7B-Instruct
    model_path = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели не найден по пути: {model_path}")
        print("Пожалуйста, скачайте модель 'mistral-7b-instruct-v0.1.Q4_K_M.gguf' с Hugging Face и укажите правильный путь.")
        exit()
    
    print(f"Загрузка модели Mistral-7B-Instruct из {model_path}...")
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=2048,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=True
    )
    
    # Новый шаблон промпта для Mistral-7B-Instruct
    # Модель Mistral-Instruct ожидает промпт в формате: 
    # [INST] Запрос [/INST]
    prompt_template = """
    [INST] Используй следующий контекст, чтобы ответить на вопрос в конце.
    Если ты не знаешь ответа, просто скажи, что ты не знаешь, не пытайся придумать ответ.
    Отвечай на русском языке.

    {context}

    Вопрос: {question} [/INST]
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    retriever = db.as_retriever(search_kwargs={"k": 5})

    print("Создание цепочки RetrievalQA...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )

    # Шаг 6: Запуск цепочки
    print("\nЦепочка готова. Задавайте свои вопросы по программам ИТМО.")
    print("Введите 'выход' для завершения.")
    while True:
        query = input("Ваш вопрос: ")
        if query.lower() == 'выход':
            break
        
        result = qa_chain({"query": query})
        
        print("\n--- Ответ ---")
        print(result['result'])
        
        print("\n--- Источники ---")
        for i, doc in enumerate(result['source_documents']):
            title = doc.metadata.get('title', 'Без заголовка')
            content_preview = doc.page_content.replace('\n', ' ')[:150]
            print(f"{i+1}. Заголовок: {title}")
            print(f"   Текст: {content_preview}...")
        print("-" * 50)



qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False, # Для бота лучше не показывать источники
    verbose=False
)
print("Бот готов к работе!")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /start."""
    await update.message.reply_text(
        "Привет! Я бот-помощник для абитуриентов магистратуры ИТМО. "
        "Я могу ответить на твои вопросы по программам"
        "Спрашивай!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения пользователя."""
    query = update.message.text
    # Отправляем "печатает..."
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        result = qa_chain({"query": query})
        response = result['result']
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text("Извини, произошла ошибка. Попробуй ещё раз.")

def main() -> None:
    """Запускает бота."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == "__main__":
    main()
