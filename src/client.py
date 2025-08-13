
from rag_lib import RAG
from rag_lib import DATABASE
import ollama
class Client:
    def __init__(self):
        #Значения по умолчанию
        llm='qwen2.5' #Загружается через ollama
        parser='unstructured'
        bi_encoder='ai-forever/sbert_large_nlu_ru' #huggingface: AutoTokenizer, AutoModel
        cross_encoder='DiTy/cross-encoder-russian-msmarco' #Sentence-Transformers CrossEncoder
        text_splitter=['symbol', 2500, 10] #Метод разделения, размер чанка, перекрытие
        index = 'hnsw'

        database_name = True # Имя базы данных, для временной базы данных используется имя temp

        llmplus = False # Для поиска использовать ответ модели без RAG
        rerank = True # Использовать reranking?
        prellm = llm
        u_input = input("Введите llm(из ollama) для поиска /mistral-nemo: ")
        if u_input != '':
            llm = u_input

        u_input = input("Прегенерировать ответ на запрос для повышения качества ответов? (займёт больше времени): Y/N ")
        if u_input== 'Y':
            llmplus = True
            u_input = input("Введите модель для прегенерации: / либо использовать основную модель")
            if u_input != '':
                prellm = u_input
        elif u_input == 'N':
            llmplus = False

        u_input = input("Использовать реранкинг? (займёт больше времени): Y/N ")
        if u_input== 'Y':
            rerank = True
        elif u_input == 'N':
            rerank = False

        print("Введите параметры для разделения документов:")
        u_input = input("Введите размер чанка в символах: /2500")
        if u_input.strip() != '':
            text_splitter[1] = int(u_input)
    
        u_input = input("Введите размер перекрытия в символах: /10")
        if u_input.strip() != '':
            text_splitter[2] = int(u_input)
        

        print("Введите параметры базы данных:")
        flag = True
        while flag:
            try:
                DB_HOST = input('DB_HOST: ')
                DB_USER = 'postgres'
                # DB_USER = input('DB_USER: ')
                DB_PASSWORD = input('PASSWORD: ')
                DEFAULT_DB = "postgres" 
                database = DATABASE(DB_HOST, DB_USER, DB_PASSWORD, DEFAULT_DB)
                flag = False
            except Exception as e:
                flag = True
                print(f"Произошла ошибка: {e}")
        
        u_input = input("Если хотите использовать готовую базу данных, введите её название: ")
        if u_input != '':
            database_name = u_input
            database.conn_db = database.connect_to_database(database_name)
            r = RAG(conn = database.conn_db,
                llm = llm, 
                parser = parser, 
                bi_encoder = bi_encoder, 
                cross_encoder = cross_encoder, 
                text_splitter = text_splitter, 
                index = index)
        else:
            database_name = 'temp'
            database.drop_database('temp') #Пытаемся удалить БД, если не удалилась в прошлом
            database.create_database('temp')
            database.setup_database(database.conn_db)
            
            r = RAG(conn = database.conn_db,
                    llm = llm, 
                    parser = parser, 
                    bi_encoder = bi_encoder, 
                    cross_encoder = cross_encoder, 
                    text_splitter = text_splitter, 
                    index = index)
            
            flag = True
            while flag:
                try:
                    path = input('Введите путь к папке с документами: ')
                    r.insert_files_in_db(folder_path=path)
                    flag = False
                except:
                    flag = True
        flag = True
        while flag:
            text_query = input('Введите запрос (пропустите для выхода): ')
            if text_query.strip() == '':
                # flag = False
                break
            if llmplus == False:
                # Получаем эмбеддинг
                emb = r.get_emb(text_query)
            if llmplus == True:
                llm_answer = ollama.generate(model=prellm, prompt=text_query)['response']
                emb = r.get_emb(llm_answer)
            # Выполняем поиск
            results = r.search(emb, limit=15)
            print('--------------------Поиск по косинусному расстоянию--------------------')
            print(f'Топ {len(results)} результатов:')
            for result in results:
                print(f'{'-'*50}')
                print(f"Text: {result['text']}\n{'-'*50}")
                print(f"Similarity: {result['similarity']:.4f}")
                print(f'{'-'*50}')
            if rerank == True:
                results = r.rerank(results, text_query)
            print('------------------Реранкинг с помощью кросс энкодера-------------------')
            print(f'После реранкинга осталось топ {len(results)} результатов:')
            for result in results:
                print(f'{'-'*50}')
                print(f"Text: {result['text']}")
                print(f"Cross_score: {result['cross_score']}")
                print(f"Similarity: {result['similarity']:.4f}")
                print(f'{'-'*50}')
            res = [results[i]['text'] for i in range(0,len(results))]
            
            ##Финальный вывод:
            if llmplus == True:
                print('Ответ модели без RAG: ')
                print(llm_answer)
            print('РЕЗУЛЬТАТ: ')
            answer = ollama.generate(system = 'Ты — помощник, который отвечает на запрос пользователя, используя только предоставленный контекст.', 
                            model='mistral-nemo', 
                            prompt=f"""
                            Контекст:
                            {res}
                            Запрос:
                            {text_query} 
                            Ответ:
                            """
                            )
            print(answer['response'])
        if database_name == 'temp':
            u_input = input("Удалить БД?: Y/N: ")
            if u_input == 'Y':
                database.drop_database(database.conn_server, 'temp')
            else:
                u_input = input("Введите новое название для БД: ")
                database.rename_database('temp', u_input)

if __name__ == "__main__":
    Client()