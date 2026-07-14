# Lab Session 01 - Set A (roll numbers ending with odd digits)
# Subject: 22AIE213

import random
import statistics


# Q1: Count pairs of elements with sum equal to 10
def count_pairs_with_sum(numbers, target):
    count = 0
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                count += 1
    return count


# Q2: Return the range (max - min) of a list. Error if less than 3 elements
def find_range(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"
    return max(numbers) - min(numbers)


# Q3: Return A raised to the power m (matrix multiplication)
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


# Q4: Find the highest occurring alphabet character and its count
def highest_occurring_char(text):
    counts = {}
    for ch in text:
        if ch.isalpha():
            counts[ch] = counts.get(ch, 0) + 1
    top_char = max(counts, key=counts.get)
    return top_char, counts[top_char]


# Q5: Generate 25 random numbers (1-10) and find mean, median, mode
def random_stats():
    numbers = [random.randint(1, 10) for _ in range(25)]
    mean_value = statistics.mean(numbers)
    median_value = statistics.median(numbers)
    mode_value = statistics.mode(numbers)
    return numbers, mean_value, median_value, mode_value


# Main program (all print statements are here)
if __name__ == "__main__":
    # Q1
    given_list = [2, 7, 4, 1, 3, 6]
    print("Q1: Pairs with sum 10 =", count_pairs_with_sum(given_list, 10))

    # Q2
    range_list = [5, 3, 8, 1, 0, 4]
    print("Q2: Range =", find_range(range_list))
    print("Q2: Range (small list) =", find_range([5, 3]))

    # Q3
    matrix_a = [[1, 2], [3, 4]]
    power_m = 2
    print("Q3: A^m =", matrix_power(matrix_a, power_m))

    # Q4
    input_string = "hippopotamus"
    char, occurrence = highest_occurring_char(input_string)
    print("Q4: Highest occurring char =", char, "count =", occurrence)

    # Q5
    numbers, mean_value, median_value, mode_value = random_stats()
    print("Q5: Numbers =", numbers)
    print("Q5: Mean =", mean_value, "Median =", median_value, "Mode =", mode_value)
