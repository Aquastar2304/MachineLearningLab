import random


def count_pairs_with_sum(numbers, target):
    count = 0
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                count += 1
    return count


def find_range(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"
    return max(numbers) - min(numbers)


def multiply_matrices(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for k in range(inner):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matrix_power(matrix, m):
    result = matrix
    for _ in range(m - 1):
        result = multiply_matrices(result, matrix)
    return result


def highest_occurring_char(text):
    counts = {}
    for ch in text:
        if ch.isalpha():
            counts[ch] = counts.get(ch, 0) + 1
    top_char = max(counts, key=counts.get)
    return top_char, counts[top_char]


def find_mean(numbers):
    return sum(numbers) / len(numbers)


def find_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    return sorted_numbers[mid]


def find_mode(numbers):
    counts = {}
    for number in numbers:
        counts[number] = counts.get(number, 0) + 1
    return max(counts, key=counts.get)


def random_stats():
    numbers = [random.randint(1, 10) for _ in range(25)]
    return numbers, find_mean(numbers), find_median(numbers), find_mode(numbers)


given_list = [2, 7, 4, 1, 3, 6]
print("Q1: Pairs with sum 10 =", count_pairs_with_sum(given_list, 10))

range_list = [5, 3, 8, 1, 0, 4]
print("Q2: Range =", find_range(range_list))
print("Q2: Range (small list) =", find_range([5, 3]))

matrix_a = [[1, 2], [3, 4]]
power_m = 2
print("Q3: A^m =", matrix_power(matrix_a, power_m))

input_string = "hippopotamus"
char, occurrence = highest_occurring_char(input_string)
print("Q4: Highest occurring char =", char, "count =", occurrence)

numbers, mean_value, median_value, mode_value = random_stats()
print("Q5: Numbers =", numbers)
print("Q5: Mean =", mean_value, "Median =", median_value, "Mode =", mode_value)
