def coin(coins, x, ready, value):
    if x < 0:
        return 1000000
    if x == 0:
        return 0
    if ready[x]:
        return value[x]
    best = 10000000
    for c in coins:
        best = min(best, coin(coins, x - c, ready, value) + 1)
    ready[x] = 1
    value[x] = best
    return best


def coin2(coins, x):
    value[0] = 0
    first = [-1] * (x + 1)
    for i in range(1, x + 1):
        value[i] = 1000000
        for c in coins:
            if i - c >= 0 and value[i - c] + 1 < value[i]:
                value[i] = value[i - c] + 1
                first[i] = c
    for i in range(0, len(first)):
        print("i: ", i, "value: ", first[i])
    return value[x]


if __name__ == '__main__':
    input_arr = [1, 3, 4]
    target = 10
    value = [0] * (target + 1)
    ready = [0] * (target + 1)
    res = coin2(input_arr, target)
    print("res:", res)
