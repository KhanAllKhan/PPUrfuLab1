"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    alf = 'aeiou'
    return sum(1 for i in text.lower() if i in alf)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    k = 0
    while number >= 10:
        res = 1
        for i in str(number):
            res *= int(i)
        number = res
        k += 1
    return k


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    res = sum((p - t) ** 2 for p, t in zip(predicted, expected))
    return res / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    res = ""
    delitel = 2
    while delitel * delitel <= number:
        if number % delitel == 0:
            count = 0
            while number % delitel == 0:
                number //= delitel
                count += 1
            res += f"({delitel}**{count})" if count > 1 else f"({delitel})"
        delitel += 1
    if number > 1:
        res += f"({number})"
    return res

def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    res = 0
    k = 1
    while res < cube_count:
        res += k**2
        if res == cube_count:
            return k
        k += 1
    return 'It is impossible'


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(digit) for digit in str(number)]
    total_length = len(digits)
    mid_index = total_length // 2

    if total_length % 2 != 0:
        left_side = digits[:mid_index]
        right_side = digits[mid_index + 1:]
    else:
        left_side = digits[:mid_index - 1]
        right_side = digits[mid_index + 1:]

    return sum(left_side) == sum(right_side)