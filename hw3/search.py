import re
import json


def load_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    return content['inverted_index'], content['documents']


def parse_query(query_str):
    cleaned_query = query_str.upper().replace(' AND ', ' && ').replace(' OR ', ' || ').replace(' NOT ', ' ! ')
    return re.findall(r'\(|\)|\w+|\&\&|\|\||!', cleaned_query)


def to_postfix(tokens):
    priority = {'!': 3, '&&': 2, '||': 1}
    result = []
    ops = []

    for token in tokens:
        if token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                result.append(ops.pop())
            ops.pop()
        elif token in priority:
            while ops and ops[-1] != '(' and priority.get(ops[-1], 0) >= priority[token]:
                result.append(ops.pop())
            ops.append(token)
        else:
            result.append(token.lower())

    while ops:
        result.append(ops.pop())

    return result


def execute_postfix(postfix_expr, index, universe):
    stack = []

    for item in postfix_expr:
        if item == '&&':
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right)
        elif item == '||':
            right = stack.pop()
            left = stack.pop()
            stack.append(left | right)
        elif item == '!':
            operand = stack.pop()
            stack.append(universe - operand)
        else:
            stack.append(set(index.get(item, [])))

    return sorted(stack.pop()) if stack else []


def search_loop(index, documents):
    all_ids = set(documents.keys())

    while True:
        try:
            user_input = input("\nВведите запрос: ").strip()
            if user_input.lower() == 'q':
                print("Выход из поиска.")
                break

            tokens = parse_query(user_input)
            postfix = to_postfix(tokens)
            matching_ids = execute_postfix(postfix, index, all_ids)

            print(f"\nКоличество документов: {len(matching_ids)}")
            for doc_id in matching_ids:
                print(f"[{doc_id}] {documents[doc_id]}")

        except Exception as error:
            print(f"Ошибка: {error}")


if __name__ == "__main__":
    inverted_idx, doc_map = load_index('inverted_index.json')
    doc_map = {int(k): v for k, v in doc_map.items()}
    search_loop(inverted_idx, doc_map)
