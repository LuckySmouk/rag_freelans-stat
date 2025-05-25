import os
import time
import json
import hashlib
import requests
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
from typing import List, Dict, Tuple, Optional
import threading

# Configuration
CONFIG = {
    "ollama_host": "http://localhost:11434",
    "embedding_model": "nomic-embed-text:latest",
    "llm_model": "qwen3:latest",
    "csv_path": "dataset/freelancer_earnings_bd.csv",
    "log_file": "freelancer_rag_test.log",
    "max_context_length": 4096,
    "top_k_results": 3,
    "cache_dir": "cache",
    "max_workers": 2,
    "batch_size": 20,
    "request_delay": 0.1
}


class FreelancerRAG:
    def __init__(self):
        self._setup_logging()
        self._check_ollama_server()
        self._check_models()

        self.cache_hash = self._get_file_hash(CONFIG["csv_path"])
        self._init_cache_dirs()

        # Thread-safe операции
        self._lock = threading.Lock()

        if self._check_cache_exists():
            self._load_from_cache()
        else:
            self._process_csv_data()
            self._save_to_cache()

    def _setup_logging(self):
        logger.add(
            CONFIG["log_file"],
            rotation="10 days",
            level="INFO",
            format="{time} | {level} | {message}"
        )

    def _check_ollama_server(self):
        print("🔄 Проверяю соединение с Ollama...")
        try:
            response = requests.get(
                f"{CONFIG['ollama_host']}/api/version", timeout=10)
            response.raise_for_status()
            print("✅ Ollama сервер доступен")
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama сервер недоступен: {e}")
            print("Запустите Ollama командой: ollama serve")
            sys.exit(1)

    def _check_models(self):
        print("🔄 Проверяю доступность моделей...")
        try:
            response = requests.get(
                f"{CONFIG['ollama_host']}/api/tags", timeout=10)
            available_models = [model["name"]
                                for model in response.json().get("models", [])]

            missing_models = []
            for model in [CONFIG["embedding_model"], CONFIG["llm_model"]]:
                if model not in available_models:
                    missing_models.append(model)

            if missing_models:
                print(f"❌ Отсутствующие модели: {missing_models}")
                print("Скачайте их командами:")
                for model in missing_models:
                    print(f"  ollama pull {model}")
                sys.exit(1)

            print("✅ Все модели доступны")

        except Exception as e:
            print(f"❌ Ошибка проверки моделей: {e}")
            sys.exit(1)

    def _init_cache_dirs(self):
        os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    def _get_file_hash(self, filepath):
        if not os.path.exists(filepath):
            print(f"❌ CSV файл не найден: {filepath}")
            sys.exit(1)

        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _check_cache_exists(self):
        cache_files = {
            'text': os.path.join(CONFIG["cache_dir"], f"text_cache_{self.cache_hash}.json"),
            'embeddings': os.path.join(CONFIG["cache_dir"], f"embeddings_cache_{self.cache_hash}.npy"),
            'metadata': os.path.join(CONFIG["cache_dir"], f"metadata_cache_{self.cache_hash}.json")
        }

        if all(os.path.exists(f) for f in cache_files.values()):
            self.text_cache_file = cache_files['text']
            self.embeddings_cache_file = cache_files['embeddings']
            self.metadata_cache_file = cache_files['metadata']
            return True
        return False

    def _load_from_cache(self):
        print("🔄 Загружаю данные из кеша...")
        try:
            with open(self.text_cache_file, 'r', encoding='utf-8') as f:
                self.text_data = json.load(f)

            with open(self.metadata_cache_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            self.embeddings = np.load(self.embeddings_cache_file)

            print(f"✅ Загружено {len(self.text_data)} записей из кеша")
            print(
                f"📊 Статистика: {self.metadata.get('total_records', 0)} исходных записей")

        except Exception as e:
            print(f"❌ Ошибка загрузки кеша: {e}")
            self._process_csv_data()
            self._save_to_cache()

    def _save_to_cache(self):
        print("💾 Сохраняю данные в кеш...")
        try:
            cache_files = {
                'text': os.path.join(CONFIG["cache_dir"], f"text_cache_{self.cache_hash}.json"),
                'embeddings': os.path.join(CONFIG["cache_dir"], f"embeddings_cache_{self.cache_hash}.npy"),
                'metadata': os.path.join(CONFIG["cache_dir"], f"metadata_cache_{self.cache_hash}.json")
            }

            with open(cache_files['text'], 'w', encoding='utf-8') as f:
                json.dump(self.text_data, f, ensure_ascii=False, indent=2)

            with open(cache_files['metadata'], 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

            np.save(cache_files['embeddings'], self.embeddings)
            print("✅ Данные сохранены в кеш")

        except Exception as e:
            print(f"❌ Ошибка сохранения кеша: {e}")

    def _process_csv_data(self):
        print("📊 Обрабатываю CSV данные...")
        try:
            # разные кодировки
            encodings = ['utf-8', 'cp1251', 'iso-8859-1', 'utf-16']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(CONFIG["csv_path"], encoding=encoding)
                    print(f"✅ CSV загружен с кодировкой: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise Exception(
                    "Не удалось загрузить CSV с поддерживаемыми кодировками")

            self.df = df
            print(f"📋 Загружено {len(self.df)} записей")
            print(f"📋 Столбцы: {list(self.df.columns)}")

            # Создаем улучшенное текстовое представление данных
            self.text_data = []

            # Подготавливаем метаданные
            self.metadata = {
                'total_records': len(self.df),
                'columns': list(self.df.columns),
                'processing_time': time.time()
            }

            # Получаем статистики для числовых полей
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            stats = {}
            for col in numeric_cols:
                if self.df[col].notna().any():
                    stats[col] = {
                        'mean': float(self.df[col].mean()),
                        'std': float(self.df[col].std()),
                        'min': float(self.df[col].min()),
                        'max': float(self.df[col].max())
                    }

            self.metadata['stats'] = stats

            # Создаем более подробное и структурированное описание каждой записи
            for idx, row in self.df.iterrows():
                # Создаем естественное описание фрилансера
                description_parts = []

                # Основная информация
                if 'Job_Category' in row and pd.notna(row['Job_Category']):
                    description_parts.append(
                        f"Категория работы: {row['Job_Category']}")

                if 'Platform' in row and pd.notna(row['Platform']):
                    description_parts.append(f"Платформа: {row['Platform']}")

                if 'Experience_Level' in row and pd.notna(row['Experience_Level']):
                    description_parts.append(
                        f"Уровень опыта: {row['Experience_Level']}")

                if 'Client_Region' in row and pd.notna(row['Client_Region']):
                    description_parts.append(
                        f"Регион клиента: {row['Client_Region']}")

                # Финансовая информация
                if 'Hourly_Rate' in row and pd.notna(row['Hourly_Rate']):
                    rate = row['Hourly_Rate']
                    if 'Hourly_Rate' in stats:
                        if rate >= stats['Hourly_Rate']['mean'] + stats['Hourly_Rate']['std']:
                            rate_category = "высокая"
                        elif rate <= stats['Hourly_Rate']['mean'] - stats['Hourly_Rate']['std']:
                            rate_category = "низкая"
                        else:
                            rate_category = "средняя"
                        description_parts.append(
                            f"Почасовая ставка: ${rate} ({rate_category})")
                    else:
                        description_parts.append(f"Почасовая ставка: ${rate}")

                if 'Earnings_USD' in row and pd.notna(row['Earnings_USD']):
                    earnings = row['Earnings_USD']
                    if 'Earnings_USD' in stats:
                        if earnings >= stats['Earnings_USD']['mean'] + stats['Earnings_USD']['std']:
                            earnings_category = "высокие"
                        elif earnings <= stats['Earnings_USD']['mean'] - stats['Earnings_USD']['std']:
                            earnings_category = "низкие"
                        else:
                            earnings_category = "средние"
                        description_parts.append(
                            f"Доходы: ${earnings} ({earnings_category})")
                    else:
                        description_parts.append(f"Доходы: ${earnings}")

                # Дополнительная информация
                if 'Job_Success_Rate' in row and pd.notna(row['Job_Success_Rate']):
                    success_rate = row['Job_Success_Rate']
                    if success_rate >= 90:
                        success_category = "отличный"
                    elif success_rate >= 70:
                        success_category = "хороший"
                    else:
                        success_category = "низкий"
                    description_parts.append(
                        f"Рейтинг успеха: {success_rate}% ({success_category})")

                if 'Job_Duration_Days' in row and pd.notna(row['Job_Duration_Days']):
                    duration = row['Job_Duration_Days']
                    if duration <= 7:
                        duration_category = "краткосрочный"
                    elif duration <= 30:
                        duration_category = "среднесрочный"
                    else:
                        duration_category = "долгосрочный"
                    description_parts.append(
                        f"Продолжительность проекта: {duration} дней ({duration_category})")

                if 'Payment_Method' in row and pd.notna(row['Payment_Method']):
                    description_parts.append(
                        f"Способ оплаты: {row['Payment_Method']}")

                if 'Job_Completed' in row and pd.notna(row['Job_Completed']):
                    description_parts.append(
                        f"Завершенных работ: {row['Job_Completed']}")

                # Остальные поля
                for col, val in row.items():
                    if (pd.notna(val) and
                        col not in ['Job_Category', 'Platform', 'Experience_Level', 'Client_Region',
                                    'Hourly_Rate', 'Earnings_USD', 'Job_Success_Rate', 'Job_Duration_Days',
                                    'Payment_Method', 'Job_Completed']):
                        description_parts.append(f"{col}: {val}")

                # Объединяем все в естественное описание
                if description_parts:
                    full_description = f"Фрилансер #{idx + 1}: " + \
                        ". ".join(description_parts) + "."
                    self.text_data.append(full_description)
                else:
                    # Если нет данных, создаем базовое описание
                    self.text_data.append(
                        f"Фрилансер #{idx + 1}: Запись с ограниченными данными.")

            print(f"✅ Создано {len(self.text_data)} текстовых описаний")

            # Показать пример обработанных данных для проверки
            if self.text_data:
                print(f"📝 Пример обработанной записи:")
                print(f"   {self.text_data[0][:200]}...")

            self._generate_embeddings_parallel()

        except Exception as e:
            print(f"❌ Ошибка обработки CSV: {e}")
            logger.error(f"CSV processing error: {e}")
            sys.exit(1)

    def _generate_embeddings_parallel(self):
        """Параллельная генерация эмбеддингов с использованием ThreadPoolExecutor"""
        print(
            f"🔄 Генерирую эмбеддинги параллельно в {CONFIG['max_workers']} потоках...")

        # Предварительно выделяем память
        embeddings = [None] * len(self.text_data)
        total = len(self.text_data)
        completed = 0

        # Разбиваем данные на батчи для стабильной работы
        batch_size = CONFIG['batch_size']

        with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_texts = self.text_data[batch_start:batch_end]

                # Создаем задачи для текущего батча
                future_to_index = {}
                for i, text in enumerate(batch_texts):
                    global_index = batch_start + i
                    future = executor.submit(
                        self._generate_embedding_safe, text, global_index)
                    future_to_index[future] = global_index

                # Собираем результаты батча
                for future in as_completed(future_to_index):
                    global_index = future_to_index[future]
                    try:
                        result = future.result(
                            timeout=120)  # Таймаут на задачу
                        if result is not None:
                            embeddings[global_index] = result

                        with self._lock:  # Thread-safe обновление счетчика
                            completed += 1
                            if completed % 10 == 0 or completed == total:
                                progress = (completed / total) * 100
                                print(
                                    f"📈 Прогресс: {completed}/{total} ({progress:.1f}%)")

                    except Exception as e:
                        logger.error(
                            f"Error processing embedding for index {global_index}: {e}")
                        with self._lock:
                            completed += 1

                # Пауза между батчами для стабильности
                if batch_end < total:
                    time.sleep(1)

        # Фильтруем успешные результаты
        valid_embeddings = []
        valid_texts = []

        for i, (text, embedding) in enumerate(zip(self.text_data, embeddings)):
            if embedding is not None:
                valid_embeddings.append(embedding)
                valid_texts.append(text)
            else:
                logger.warning(f"Missing embedding for text {i}")

        if not valid_embeddings:
            print("❌ Не удалось сгенерировать ни одного эмбеддинга")
            sys.exit(1)

        self.embeddings = np.array(valid_embeddings)
        self.text_data = valid_texts

        print(
            f"✅ Сгенерировано {len(self.embeddings)} эмбеддингов из {total} записей")
        if len(self.embeddings) < total:
            print(
                f"⚠️ Потеряно {total - len(self.embeddings)} записей при генерации эмбеддингов")

    def _generate_embedding_safe(self, text: str, index: int) -> Optional[np.ndarray]:
        """Thread-safe версия генерации эмбеддинга"""
        max_retries = 3
        base_delay = CONFIG['request_delay']

        for attempt in range(max_retries):
            try:
                # Добавляем случайную задержку для избежания перегрузки сервера
                time.sleep(base_delay * (1 + np.random.random() * 0.5))

                truncated_text = text[:CONFIG["max_context_length"]]

                response = requests.post(
                    f"{CONFIG['ollama_host']}/api/embed",
                    json={
                        "model": CONFIG["embedding_model"],
                        "input": truncated_text
                    },
                    timeout=90
                )
                response.raise_for_status()

                result = response.json()
                if "embeddings" in result and result["embeddings"]:
                    return np.array(result["embeddings"][0])
                else:
                    logger.error(
                        f"No embeddings in response for index {index}: {result}")
                    return None

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * \
                        (1 + np.random.random())
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Timeout after {max_retries} attempts for index {index}")
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Error generating embedding for index {index} after {max_retries} attempts: {e}")
                    return None

        return None

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Генерация одного эмбеддинга для поиска"""
        return self._generate_embedding_safe(text, -1)

    def _find_similar(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if top_k is None:
            top_k = CONFIG["top_k_results"]

        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []

        if len(self.embeddings) == 0:
            return []

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for i in top_indices:
            if i < len(self.text_data):
                results.append({
                    'text': self.text_data[i],
                    'similarity': similarities[i],
                    'index': i
                })

        return results

    def generate_answer(self, query: str) -> Tuple[str, float]:
        start_time = time.time()

        similar_items = self._find_similar(query, CONFIG["top_k_results"])
        if not similar_items:
            return "Не удалось найти релевантную информацию.", time.time() - start_time

        # Создаем более подробный контекст
        context_parts = [f"Найденная информация (релевантность: {item['similarity']:.3f}):\n{item['text']}"
                         for item in similar_items[:3]]
        context = "\n\n".join(context_parts)

        # Добавляем статистическую информацию если есть
        if hasattr(self, 'metadata') and 'stats' in self.metadata:
            stats_info = []
            for field, stat in self.metadata['stats'].items():
                if any(keyword in field.lower() for keyword in ['rate', 'earning', 'success']):
                    stats_info.append(
                        f"{field}: среднее={stat['mean']:.2f}, мин={stat['min']:.2f}, макс={stat['max']:.2f}")

            if stats_info:
                context += f"\n\nСтатистическая информация по данным:\n" + \
                    "\n".join(stats_info)

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты эксперт-аналитик данных о фрилансерах. Анализируй ТОЛЬКО предоставленные данные. "
                    "Если точной информации нет - скажи об этом честно. "
                    "Используй конкретные цифры и факты из контекста. "
                    "Отвечай структурированно и по существу. "
                    "Если видишь тенденции в данных - укажи их. "
                    "Всегда основывай выводы на представленных фактах."
                )
            },
            {
                "role": "user",
                "content": f"Данные для анализа:\n{context}\n\nВопрос: {query}\n\nПроанализируй данные и дай ответ:"
            }
        ]

        try:
            response = requests.post(
                f"{CONFIG['ollama_host']}/api/chat",
                json={
                    "model": CONFIG["llm_model"],
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            if "message" in result and "content" in result["message"]:
                answer = result["message"]["content"]
            else:
                answer = "Не удалось получить ответ от модели."

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"Ошибка при генерации ответа: {str(e)}"

        return answer, time.time() - start_time

    def debug_info(self):
        """Выводит отладочную информацию о загруженных данных"""
        print("\n" + "="*60)
        print("🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ")
        print("="*60)

        if hasattr(self, 'metadata'):
            print(
                f"📊 Всего записей в исходном CSV: {self.metadata.get('total_records', 'Неизвестно')}")
            print(f"📋 Столбцы: {', '.join(self.metadata.get('columns', []))}")

        print(
            f"📝 Текстовых описаний: {len(self.text_data) if hasattr(self, 'text_data') else 0}")
        print(
            f"🔢 Эмбеддингов: {len(self.embeddings) if hasattr(self, 'embeddings') else 0}")

        if hasattr(self, 'text_data') and self.text_data:
            print(f"\n📄 Первые 3 примера обработанных данных:")
            for i, text in enumerate(self.text_data[:3]):
                print(f"{i+1}. {text[:150]}...")

        if hasattr(self, 'metadata') and 'stats' in self.metadata:
            print(f"\n📈 Статистика по числовым полям:")
            for field, stats in self.metadata['stats'].items():
                print(
                    f"  {field}: мин={stats['min']:.2f}, макс={stats['max']:.2f}, среднее={stats['mean']:.2f}")

        print("="*60 + "\n")

    def run_sample_queries(self):
        """Запуск тестовых запросов из задания"""
        sample_queries = [
            "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?",
            "Как распределяется доход фрилансеров в зависимости от региона проживания?",
            "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?",
            "Какая средняя почасовая ставка у веб-разработчиков разного уровня опыта?",
            "Какие платформы обеспечивают наибольший доход фрилансерам?",
            "Как влияет рейтинг успеха на размер заработка фрилансера?"
        ]

        print("\n" + "="*60)
        print("🔬 ДЕМОНСТРАЦИЯ СИСТЕМЫ - ТЕСТОВЫЕ ЗАПРОСЫ")
        print("="*60)

        for i, query in enumerate(sample_queries, 1):
            print(f"\n📋 Запрос {i}: {query}")
            print("-" * 50)

            answer, duration = self.generate_answer(query)
            print(f"💡 Ответ: {answer}")
            print(f"⏱️ Время: {duration:.2f} сек")

            if i < len(sample_queries):
                input("\nНажмите Enter для следующего запроса...")

    def run(self):
        def signal_handler(sig, frame):
            print("\n👋 Завершение работы...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        print("\n" + "="*60)
        print("🚀 СИСТЕМА АНАЛИЗА ДОХОДОВ ФРИЛАНСЕРОВ")
        print("="*60)
        print("💡 Команды:")
        print("   • Задавайте вопросы о данных фрилансеров")
        print("   • 'demo' для демонстрации тестовых запросов")
        print("   • 'debug' для отладочной информации")
        print("   • 'exit' или 'quit' для выхода")
        print("   • 'help' для примеров вопросов")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("❓ Ваш вопрос: ").strip()

                if user_input.lower() in ['exit', 'quit', 'выход']:
                    print("👋 До свидания!")
                    break

                if user_input.lower() in ['demo', 'демо']:
                    self.run_sample_queries()
                    continue

                if user_input.lower() in ['debug', 'отладка']:
                    self.debug_info()
                    continue

                if user_input.lower() in ['help', 'помощь']:
                    print("\n📋 Примеры вопросов:")
                    print("  • Какая средняя почасовая ставка у веб-разработчиков?")
                    print("  • Сколько зарабатывают фрилансеры в Австралии?")
                    print("  • Какие платформы самые популярные?")
                    print("  • Как опыт влияет на доходы?")
                    print("  • Какой рейтинг успеха у новичков?")
                    print("  • Сколько фрилансеров работает с мобильным банкингом?")
                    print("  • Насколько выше доходы при оплате криптовалютой?\n")
                    continue

                if not user_input:
                    continue

                print("🔍 Поиск ответа...")
                answer, duration = self.generate_answer(user_input)

                print(f"\n💡 {answer}")
                print(f"⏱️  Время: {duration:.2f} сек\n")

            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                logger.error(f"Runtime error: {e}")
                print("❌ Произошла ошибка. Попробуйте еще раз.\n")


def main():
    """Главная функция для запуска программы"""
    try:
        print("🚀 Запуск системы анализа доходов фрилансеров...")
        rag_system = FreelancerRAG()
        rag_system.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Критическая ошибка: {e}")
        input("Нажмите Enter для выхода...")
        sys.exit(1)


if __name__ == "__main__":
    main()
