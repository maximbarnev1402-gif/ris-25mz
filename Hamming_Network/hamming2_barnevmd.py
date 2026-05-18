import numpy as np

# ============================================================
#  СЕТЬ ХЭММИНГА - Коррекция опечаток
#
#  Идея: Чарли Гордон из романа "Цветы для Элджернона" пишет
#  слова с ошибками.
#  Я решил использовать Сеть Хэмминга для поиска правильного варианта написания слов в предложениях Чарли.
#
#  КАК РАБОТАЕТ СЕТЬ ХЭММИНГА:
#
#  1. Каждое слово кодируется в бинарный вектор.
#  2. Все правильные слова из словаря тоже закодированы и хранятся как образцы.
#  3. Слой 1 вычисляет схожесть входного слова с каждым образцом: чем больше совпадающих битов, тем выше балл.
#  4. Слой 2 (MAXNET) - конкурентный: нейроны подавляют друг друга, пока не останется один победитель - ближайший образец.
#  5. Победитель - это исправленное слово.
# ============================================================

# Каждое слово представляется вектором из 62 бит:
#    Биты  0–25 : есть ли буква a, b, c, ..., z в слов
#    Биты 26–51 : какая буква стоит первой
#    Биты 52–61 : длина слова от 1 до 10

VEC_SIZE = 62

DEFAULT_MISSPELLED = (
    'I am riting a riport becaus the teecher sed I shud show my progris '
    'and I wuz reely nervus about it'
)
# Правильная версия того же предложения (для сравнения результата):
DEFAULT_CORRECT = (
    'I am writing a report because the teacher said I should show my progress '
    'and I was really nervous about it'
)

#  Кодирование слова в бинарный вектор
def encode_word(word: str) -> np.ndarray:
    w = word.lower().strip()

    vec = np.zeros(VEC_SIZE, dtype=float)

    # Для каждой буквы слова ставим 1 на соответствующей позиции.
    for ch in w:
        if 'a' <= ch <= 'z':
            vec[ord(ch) - ord('a')] = 1.0

    # Если слово начинается на 'w', то бит 26+22=48 становится 1.
    vec[26 + ord(w[0]) - ord('a')] = 1.0

    # Слова длиной 1 - бит 52, длиной 2 - бит 53, ..., длиной 10+ - бит61.
    bucket = min(len(w), 10) - 1
    vec[52 + bucket] = 1.0

    return vec  # готовый бинарный вектор длиной 62



#  Расстояние Хэмминга между векторами
def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

#  Словарь образцов
DICTIONARY = [
    'progress', 'report', 'was', 'because', 'friend', 'nervous',
    'really', 'every', 'happiness', 'worried', 'teacher', 'should',
    'could', 'would', 'nothing', 'something', 'thought', 'said',
    'difficult', 'special', 'real', 'write', 'writing', 'young',
    'intelligent', 'important', 'remember', 'practice', 'learn',
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
    'her', 'one', 'our', 'had', 'his', 'how', 'may', 'new', 'now',
    'old', 'see', 'two', 'way', 'who', 'did', 'get', 'has', 'too',
    'any', 'day', 'also', 'back', 'call', 'came', 'come', 'does',
    'down', 'even', 'find', 'give', 'good', 'hand', 'have', 'here',
    'high', 'home', 'into', 'just', 'keep', 'kind', 'know', 'last',
    'left', 'life', 'like', 'line', 'live', 'long', 'look', 'made',
    'make', 'many', 'most', 'much', 'must', 'name', 'need', 'next',
    'only', 'open', 'over', 'part', 'play', 'read', 'right', 'same',
    'show', 'side', 'such', 'take', 'tell', 'than', 'them', 'then',
    'they', 'time', 'turn', 'very', 'walk', 'want', 'went', 'what',
    'when', 'with', 'word', 'work', 'year', 'your', 'about', 'again',
    'after', 'first', 'house', 'large', 'later', 'light', 'never',
    'night', 'other', 'place', 'plant', 'point', 'quite', 'short',
    'since', 'small', 'sound', 'space', 'stand', 'start', 'state',
    'still', 'story', 'study', 'table', 'their', 'there', 'thing',
    'think', 'those', 'three', 'today', 'under', 'until', 'watch',
    'water', 'where', 'which', 'white', 'whole', 'world', 'wrong',
    'hard', 'feel', 'miss', 'test', 'stop', 'smart',
    'i', 'a', 'my', 'am', 'it',
]
# Убирает возможные дублирующиеся слова, сохраняя порядок
DICTIONARY = list(dict.fromkeys(DICTIONARY))

#  Реализация сети Хэмминга
class HammingNetwork:

    def __init__(self, words: list[str]):
        # Список слов-образцов
        self.words = words

        # МИатрица образцов shape=(m, n), где m - количество слов, n - длина вектора (62)
        self.etalons = np.array([encode_word(w) for w in words])
        self.m, self.n = self.etalons.shape 

        # Матрица весов W = P / 2,  где P - матрица образцов
        # Порог b = n / 2
        self.W = self.etalons / 2
        self.b = self.n / 2

        # ε = 1/m - коэффициент взаимного торможения нейронов
        self.epsilon = 1.0 / self.m

    # Слой 1 - Вычисление схожести образцов
    def layer1(self, x: np.ndarray) -> np.ndarray:
        # Умножаем матрицу весов W на входной вектор x и добавляем порог b.
        # Результат - вектор из m чисел: y[i] тем больше, чем ближе x к образцу i.
        return self.W @ x + self.b

    # Слой 2 - конкурентный выбор победителя (MAXNET)
    def maxnet(self, y: np.ndarray, max_iter: int = 200) -> int:
        for _ in range(max_iter):
            # y_new = max(0, y - ε * (sum(y) - y))
            total = y.sum()
            y_new = np.maximum(0.0, y - self.epsilon * (total - y))
            # Условие остановки: остался не более одного ненулевого нейрона
            if (y_new > 1e-9).sum() <= 1:
                y = y_new
                break
            y = y_new
        # Возвращаем индекс нейрона-победителя (наибольшее значение)
        return int(np.argmax(y))

    # Поиск правильного слова
    def correct(self, word: str, top_k: int = 5):
        # Кодируем входное слово в бинарный вектор
        x = encode_word(word)

        # Прогоняем через слой 1
        y1 = self.layer1(x)

        # Слой 2 определяет ближайший образец
        winner = self.maxnet(y1.copy())

        # Вычисляем расстояния Хэмминга
        dists = np.array([hamming_distance(x, e) for e in self.etalons])

        # Сортируем по возрастанию и получаем топ-5 ближайших образцов
        order = np.argsort(dists)
        top = [(self.words[i], int(dists[i])) for i in order[:top_k]]

        # Возвращаем: образец-победитель, топ-k кандидатов, вектор входа
        return self.words[winner], top, x


if __name__ == '__main__':
    np.random.seed(0)  # фиксируем случайность для воспроизводимости

    # Загружаем образцы
    net = HammingNetwork(DICTIONARY)

    print('  СЕТЬ ХЭММИНГА - Коррекция опечаток Чарли Гордона')
    print(f'  Размер словаря : {len(DICTIONARY)} слов')
    print(f'  Размер вектора : {VEC_SIZE} бит')
    print()

    # Пользователь вводит предложение или нажимает Enter для дефолтного значения.
    raw_input = input('  Предложение с опечатками (Enter = дефолт):\n  > ').strip()
    use_default = not raw_input  # True, если пользователь ничего не ввёл
    misspelled_sentence = DEFAULT_MISSPELLED if use_default else raw_input

    if use_default:
        # При дефолтном входе автоматически берём и дефолтный правильный вариант
        print(f'  Используется дефолт: {DEFAULT_MISSPELLED}')
        correct_sentence = DEFAULT_CORRECT
        print(f'  Правильный вариант:  {DEFAULT_CORRECT}')
    else:
        # Если пользователь ввёл своё предложение - спрашиваем, хочет ли он задать правильный вариант для проверки точности
        print()
        want_correct = input('  Ввести правильный вариант для сравнения? (y/n):\n  > ').strip().lower()
        if want_correct == 'y':
            correct_sentence = input('  Правильное предложение:\n  > ').strip()
        else:
            correct_sentence = None  # режим без правильного варианта предложения - только поиск из словаря

    print()
    print('-' * 70)

    # Разбиваем предложение на отдельные слова
    misspelled_tokens = misspelled_sentence.split()

    # Сравнение результата с правильным предложением
    if correct_sentence is not None:
        correct_tokens = correct_sentence.split()

        # Предупреждаем, если длины предложений не совпадают
        if len(misspelled_tokens) != len(correct_tokens):
            print(f'  ВНИМАНИЕ: количество слов не совпадает '
                  f'({len(misspelled_tokens)} vs {len(correct_tokens)}). '
                  f'Сравнение по позиции до минимальной длины.')

        # Сравниваем только до конца более короткого предложения
        n_compare = min(len(misspelled_tokens), len(correct_tokens))

        print(f'  {"Исходное":>12}  {"Сеть нашла":>12}  {"Ожидалось":>12}  '
              f'{"Расст.":>6}  {"OK?":>4}')
        print('-' * 70)

        results = []
        for i in range(n_compare):
            input_word = misspelled_tokens[i].lower()   # слово с опечаткой
            expected   = correct_tokens[i].lower()      # правильное слово

            found, top5, _ = net.correct(input_word)

            # Расстояние Хэмминга между входным и найденным словом
            dist = hamming_distance(encode_word(input_word), encode_word(found))

            # Проверяем, совпало ли то, что нашла сеть, с ожидаемым словом
            ok = '+' if found == expected else '-'

            results.append((input_word, found, expected, dist, top5, ok))
            print(f'  {input_word:>12}  {found:>12}  {expected:>12}  {dist:>6}  {ok:>4}')

        # Подсчёт точности: сколько слов сеть нашла правильно
        n_correct = sum(1 for r in results if r[5] == '+')
        print()
        print(f'Точность: {n_correct}/{len(results)} = {n_correct/len(results)*100:.0f}%')

        # Разбор случаев, где программа ошиблась
        wrong = [r for r in results if r[5] == '-']
        if wrong:
            print()
            print('-' * 70)
            print('Слова, где сеть ошиблась:')
            for input_word, found, expected, dist, top5, _ in wrong:
                print()
                print(f'  "{input_word}" -> сеть: "{found}", ожидалось: "{expected}"')
                print(f'  Топ-5 ближайших по Хэммингу:')
                for w, d in top5:
                    mark = ' <- ожидалось' if w == expected else ''
                    bar  = '█' * max(0, 20 - d) + '░' * min(20, d)
                    print(f'    {w:>12} : d={d:>2}  {bar}{mark}')

    # Результат без сравнения с правильным предложением, только поиск праывильных слов
    else:
        print(f'  {"Исходное":>12}  {"Сеть нашла":>12}  {"Расст.":>6}')
        print('-' * 70)

        for token in misspelled_tokens:
            input_word = token.lower()
            found, _, _ = net.correct(input_word)
            dist = hamming_distance(encode_word(input_word), encode_word(found))
            print(f'  {input_word:>12}  {found:>12}  {dist:>6}')