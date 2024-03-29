from math import factorial


def calculate_collision_probability(total_ids, total_machines):
    probability_no_collision = 1

    for i in range(total_machines):
        probability_no_collision *= (total_ids - i) / (total_ids)

    probability_collision = 1 - probability_no_collision
    return probability_collision


def calculate_collision_probability_2(total_ids, total_machines):
    return 1 - factorial(total_ids) / (factorial(total_ids - total_machines) * pow(total_ids, total_machines))


total_ids = 365
total_machines = 23

collision_probability_1 = calculate_collision_probability(total_ids, total_machines)
print(collision_probability_1)
collision_probability = calculate_collision_probability_2(total_ids, total_machines)
print(collision_probability)
