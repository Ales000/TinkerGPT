import json
import os

DATASET_PATH = 'conversations_dataset.json'

def load_data(filepath):
    """Загружает данные из JSON файла или создает новую структуру, если файла нет."""
    if os.path.exists(filepath):
        print(f"Найден существующий файл '{filepath}'. Загружаем данные...")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"Файл '{filepath}' не найден. Будет создан новый.")
        return []

def save_data(filepath, data):
    """Сохраняет данные в JSON файл с красивым форматированием."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Данные успешно сохранены в '{filepath}'.")

def handle_delete(dataset):
    """Обрабатывает логику удаления записи из данных."""
    if not dataset:
        print("База данных пуста. Нечего удалять.")
        return dataset

    print("\n--- Режим удаления ---")
    # Выводим нумерованный список всех вопросов
    for i, item in enumerate(dataset):
        print(f"  {i+1}: {item['question']}")
    
    try:
        choice_str = input("Введите номер вопроса для удаления (или 'отмена' для возврата): ")
        if choice_str.lower() == 'отмена':
            print("Удаление отменено.")
            return dataset

        choice_index = int(choice_str) - 1

        if 0 <= choice_index < len(dataset):
            item_to_delete = dataset[choice_index]
            print("\nВы выбрали для удаления:")
            print(f"  Вопрос: {item_to_delete['question']}")
            print(f"  Ответы: {', '.join(item_to_delete['answers'])}")
            
            confirm = input("Вы уверены, что хотите удалить эту запись? (да/нет): ").lower()
            if confirm == 'да':
                deleted_item = dataset.pop(choice_index)
                print(f"Запись '{deleted_item['question']}' успешно удалена.")
            else:
                print("Удаление отменено.")
        else:
            print("Неверный номер. Пожалуйста, выберите номер из списка.")
    
    except (ValueError, IndexError):
        print("Некорректный ввод. Пожалуйста, введите числовой номер из списка.")
    
    return dataset

def main():
    """Основной цикл программы для добавления и удаления данных."""
    dataset = load_data(DATASET_PATH)
    
    print("\n--- Редактор Базы Знаний ---")
    print("Введите 'выход' для сохранения и выхода.")
    print("Введите '/delete' для перехода в режим удаления.")

    while True:
        # Обновляем карту вопросов на каждой итерации на случай удаления
        questions_map = {item['question']: i for i, item in enumerate(dataset)}

        # Шаг 1: Получаем ввод
        user_input = input("\nВведите ВОПРОС или команду: ").strip()

        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
        
        if user_input.lower() == '/delete':
            dataset = handle_delete(dataset)
            continue # Возвращаемся к началу цикла

        if not user_input:
            print("Вопрос не может быть пустым. Попробуйте снова.")
            continue

        question = user_input

        # Шаг 2: Получаем ответ
        answer = input(f"Введите ОТВЕТ на '{question}': ").strip()

        if not answer:
            print("Ответ не может быть пустым. Попробуйте снова.")
            continue
            
        # Шаг 3: Добавляем данные
        if question in questions_map:
            question_index = questions_map[question]
            if answer not in dataset[question_index]['answers']:
                dataset[question_index]['answers'].append(answer)
                print(f" -> Добавлен новый вариант ответа к существующему вопросу.")
            else:
                print(f" -> Такой ответ уже существует для этого вопроса.")
        else:
            new_entry = {"question": question, "answers": [answer]}
            dataset.append(new_entry)
            print(f" -> Добавлен новый вопрос и ответ.")
    
    # Шаг 4: Сохраняем все изменения
    save_data(DATASET_PATH, dataset)

if __name__ == "__main__":
    main()
