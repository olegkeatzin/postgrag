import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

from unstructured.partition.auto import partition
from unstructured.file_utils.filetype import detect_filetype

from tqdm import tqdm

import numpy as np
import psycopg2
from psycopg2 import sql
from pgvector.psycopg2 import register_vector
from typing import List, Dict
import os

class RAG:
    def __init__(self, conn, 
                llm='mistral-nemo', #Загружается через ollama
                parser='unstructured', 
                bi_encoder='ai-forever/sbert_large_nlu_ru', #huggingface: AutoTokenizer, AutoModel
                cross_encoder='DiTy/cross-encoder-russian-msmarco', #Sentence-Transformers CrossEncoder
                text_splitter=('symbol', 2500, 10), #Метод разделения, размер чанка, перекрытие
                index = 'hnsw'): #Метод индексации в БД
        #Соединение с БД
        self.conn = conn
        self.cursor = conn.cursor() #для отправки SQL запросов
        register_vector(conn) #для работы pgvector
        
        # Конфигурация
        self.config = {
            'llm': llm,
            'parser': parser,
            'text_splitter': text_splitter,
            'bi_encoder': bi_encoder,
            'cross_encoder': cross_encoder,
            'index': index
        }

        #Инициализация функций по конфигу
        if self.config['parser'] == 'unstructured':
            self.get_text = self.get_text_unstructured
        if self.config['text_splitter'][0] == 'symbol':
            self.split_text = self.split_text_symbol
        if self.config['index'] == 'hnsw':
            self.create_index = self.create_hnsw_index

        
        #Инициализация моделей
        #bi encoder
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['bi_encoder'])
        self.model_bi_encoder = AutoModel.from_pretrained(self.config['bi_encoder'])

        #cross encoder
        self.model_cross_encoder = CrossEncoder(self.config['cross_encoder'], max_length=512, device='cuda')

    # def get_text_unstructured(self, path_to_pdf): #Парсер текстов с помощью библиотеки unstructured
    #     elements = partition_pdf(path_to_pdf)
    #     text_elements = [element for element in elements if element.category in ["Title", "NarrativeText"]] #ОСТАВЛЯЕМ ТОЛЬКО ТЕКСТ, УБИРАЕМ ТАБЛИЦЫ И ИЗОБРАЖЕНИЯ
    #     text = "\n".join([element.text for element in text_elements])
    #     # Заменяем переносы строк на пробелы
    #     text = text.replace("\n", " ")
        
    #     # Убираем лишние пробелы (если они образовались после замены)
    #     text = " ".join(text.split())
    #     return text
    def get_text_unstructured(self, path_to_file):
        # Определяем тип файла
        filetype = detect_filetype(path_to_file)
        
        # Парсим файл в зависимости от его типа
        elements = partition(filename=path_to_file, filetype=filetype)
        
        # Оставляем только текстовые элементы (заголовки и основной текст)
        text_elements = [element for element in elements if element.category in ["Title", "NarrativeText"]]
        
        # Объединяем текстовые элементы в одну строку
        text = "\n".join([element.text for element in text_elements])
        
        # Заменяем переносы строк на пробелы
        text = text.replace("\n", " ")
        
        # Убираем лишние пробелы (если они образовались после замены)
        text = " ".join(text.split())
    
        return text
    def split_text_symbol(self, text): #Деление текстов по числу символов в чанке
        chunk_size = self.config['text_splitter'][1]
        overlap_size = self.config['text_splitter'][2]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap_size  # Учитываем перекрытие
        return chunks
    
    #Получение эмбеддингов

    #Mean Pooling - Take attention mask into account for correct averaging - нужно для get_emb
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_emb(self, chunks):
        #Sentences we want sentence embeddings for
        # sentences = ['Привет! Как твои дела?',
        #              'А правда, что 42 твое любимое число?']

        #Tokenize sentences
        encoded_input = self.tokenizer(chunks, padding=True, truncation=True, max_length=512, return_tensors='pt')

        #Compute token embeddings
        with torch.no_grad():
            model_output = self.model_bi_encoder(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return np.squeeze(np.array(sentence_embeddings.tolist()))
    
    def create_hnsw_index(self):
        self.cursor.execute('DROP INDEX IF EXISTS chunks_hnsw_idx; CREATE INDEX chunks_hnsw_idx ON chunks USING hnsw (chunk_embedding vector_cosine_ops);') #Задаём HNSW индекс после заполнения БД        

    # Загрузка файлов из папки в базу данных
    def get_file_paths(self, folder_path):
        file_paths = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                file_paths.append(full_path)
        return file_paths
    


    def insert_files_in_db(self, folder_path):
        paths = self.get_file_paths(folder_path)
        for path in tqdm(paths, desc="Обработка файлов"):  # Открываем файл
            self.cursor.execute("INSERT INTO files (file_path) VALUES (%s) RETURNING id;", (path,))
            path_id = self.cursor.fetchone()[0]
            text = self.get_text(path)  # Парсинг файла
            chunks = self.split_text(text)  # Делим текст на чанки
            embs = self.get_emb(chunks)  # Получаем эмбеддинги чанков
            for i in range(len(chunks)):
                self.cursor.execute("INSERT INTO chunks (chunk_text, chunk_embedding, fk_file_of_chunks) VALUES (%s, %s, %s);", (chunks[i], embs[i], path_id))  # Заносим чанк и эмбеддинг в таблицу чанков в соответствии к принадлежности к файлу.
        self.conn.commit()
        print('Все файлы загружены, настраиваем индексацию...')
        self.create_index()  # Задаём индекс после заполнения БД
        print('Выполнено!!!')


    
    # Векторный поиск
    def search(self,
        query_embedding: List[float], 
        table_name: str = "chunks", 
        vector_column: str = "chunk_embedding", 
        text_column: str = "chunk_text", 
        limit: int = 10
    ) -> List[Dict]:
        """
        Выполняет векторный поиск с использованием индекса HNSW.

        :param cursor: Курсор базы данных.
        :param query_embedding: Эмбеддинг запроса (список float).
        :param table_name: Имя таблицы.
        :param vector_column: Имя столбца с векторами.
        :param text_column: Имя столбца с текстом.
        :param limit: Количество возвращаемых результатов.
        :return: Список словарей с результатами поиска.
        """
        try:
            # Выполняем поиск с использованием оператора <=> (косинусная схожесть)
            self.cursor.execute(f"""
                SELECT {text_column}, {vector_column} <=> %s AS similarity
                FROM {table_name}
                ORDER BY similarity
                LIMIT %s
            """, (query_embedding, limit))
            
            # Формируем результат
            results = [
                {"text": row[0], "similarity": float(row[1])}
                for row in self.cursor.fetchall()
            ]
            return results

        except Exception as e:
            print(f"Ошибка при выполнении поиска: {e}")
            return []
    
    #Reranking
    # def rerank(self, results, text_query):
    #     bi_results = []
    #     for result in results:
    #         bi_results.append(result['text'])
    #     cross_results = self.model_cross_encoder.rank(text_query, bi_results)
    #     for cross_result in cross_results:
    #         results[cross_result['corpus_id']]['cross_score'] = cross_result['score']
    #     results = sorted(results, key=lambda x: x["cross_score"], reverse=True)
    #     return results
    def rerank(self, results, text_query):
        bi_results = []
        for result in results:
            bi_results.append(result['text'])
        
        # Получаем ранжированные результаты от кросс-энкодера
        cross_results = self.model_cross_encoder.rank(text_query, bi_results)
        
        # Добавляем cross_score в исходные результаты
        for cross_result in cross_results:
            results[cross_result['corpus_id']]['cross_score'] = cross_result['score']
        
        # Сортируем результаты по cross_score в порядке убывания
        results = sorted(results, key=lambda x: x["cross_score"], reverse=True)
        
        # Удаляем результаты с cross_score < 0.1
        results = [result for result in results if result['cross_score'] >= 0.1]
        
        return results
        
        

class DATABASE:
    def __init__(self, DB_HOST = "localhost", DB_USER = "your_user", DB_PASSWORD = "your_password", DEFAULT_DB = "postgres"):
        # Параметры подключения к серверу PostgreSQL
        self.DB_HOST = DB_HOST
        self.DB_USER = DB_USER
        self.DB_PASSWORD = DB_PASSWORD
        self.DEFAULT_DB = DEFAULT_DB  # База по умолчанию для начального подключения
        print('Подключение к серверу...')
        self.conn_server = self.connect_to_database(self.DEFAULT_DB) #Подключение к серверу
        print('Сервер подключен.')
        self.conn_db = None #Подключение к БД

        #Настройка БД
        self.prebuild_code = """
        CREATE EXTENSION vector;
        CREATE TABLE files
        (
        	id serial PRIMARY KEY,
        	file_path text NOT NULL
        );
        
        CREATE TABLE chunks (
        	id serial PRIMARY KEY,
        	chunk_embedding vector(1024) NOT NULL,
        	chunk_text text NOT NULL,
        	fk_file_of_chunks int REFERENCES files(id)
        );
        """
    def connect_to_database(self, db_name):
        """Подключается к указанной базе данных."""
        print(f'Подключение к базе данных {db_name} ...')
        conn = psycopg2.connect(
            host=self.DB_HOST,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            dbname=db_name
        )
        conn.autocommit = True
        print(f'Подключение к базе данных {db_name} выполнено!')
        return conn
    
    def create_database(self, db_name):
        """Создает базу данных."""
        print('Создаём БД...')
        with self.conn_server.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(db_name)))
            print(f"База данных '{db_name}' создана.")
        print('Создаём conn_db...')
        self.conn_db = self.connect_to_database(db_name)
        print('conn_db создан.')

    def setup_database(self, conn_db):
        """Настраивает базу данных: создает таблицы, добавляет данные."""
        with conn_db.cursor() as cur:
        
            # Вставляем тестовые данные
            cur.execute(self.prebuild_code)
            print("БД настроена.")
    def use_database(self, conn, command):
        """Использует базу данных: выполняет запросы."""
        with conn.cursor() as cur:
            # Пример запроса: выборка данных
            cur.execute(command)
            rows = cur.fetchall()
            print("Вывод после выполнения программы:")
            for row in rows:
                print(row)
    def logout_database(self, db_name):
        with self.conn_server.cursor() as cur:
            # Завершаем все активные соединения с базой данных
            cur.execute(
                sql.SQL("""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = %s
                    AND pid <> pg_backend_pid();
                """), [db_name]
            )
            print(f"Все активные соединения с базой '{db_name}' завершены.")
    def drop_database(self, db_name): #именно conn_server!!! потому что соединение с удаляемой бд(conn_db) должно быть закрыто
        """Удаляет базу данных, завершая все активные соединения."""
        with self.conn_server.cursor() as cur:
            self.logout_database(db_name) #Закрываем все соединения
            # Удаляем базу данных
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {};").format(sql.Identifier(db_name)))
            print(f"База данных '{db_name}' удалена.")
    
    def rename_database(self, current_name, future_name):
        with self.conn_server.cursor() as cur:
            self.logout_database(current_name)  # Закрываем все соединения
            query = sql.SQL("ALTER DATABASE {} RENAME TO {};").format(
                sql.Identifier(current_name),
                sql.Identifier(future_name)
            )
            cur.execute(query)
            print(f"База данных '{current_name}' переименована в '{future_name}'")

