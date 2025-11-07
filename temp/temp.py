import random
# from pathlib import Path
# import numpy as np
# import SimpleITK as sitk
# import os
# from data_preprocessing.text_worker import (json_reader, yaml_reader,
#                                             add_info_logging)
count_steps = 1000000000
start_position = 0
count_flat = 0
count_auto = 0
count_hotel = 0
cycle = 0
count_new_chance = 0
count_step_forward = 0
count_step_back = 0
SETS = [0, 0, 0, 1, 10, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0]



def find_len_steps():
    return random.randint(1, 6) + random.randint(1, 6)

def check_new_round(position):
    global cycle
    if position >= len(SETS):
        cycle += 1
        return position - len(SETS), True
    return position, False

def steps_to_target(cur_pos, target_idx=4):
    """Число шагов по часовой от cur_pos до target_idx в циклическом поле."""
    # если target_idx == prev_pos => 0 шагов
    return (target_idx - cur_pos) % len(SETS)

def find_action(position, previous_position, new_round, cur_len, new_chance=False):
    global count_flat, count_auto, count_hotel, count_step_forward, count_step_back, count_new_chance
    needed = steps_to_target(position)
    if SETS[position] == 1 and not new_chance:
        count_flat += 1
        count_step_forward += 1
        return position + 1
    elif SETS[position] == 2 and not new_chance:
        count_flat += 1
        count_step_back += 1
        return position -1
    if SETS[position] == 10:
        count_flat += 1
        return position
    elif SETS[position] == 5:
        count_auto += 1
        return position
    elif SETS[position] == 7:
        count_hotel += 1
        return position
    elif SETS[position] == 0 and position > 4 and (previous_position < 4 or new_round) and not new_chance:
        count_new_chance += 1
        new_position = previous_position + find_len_steps()
        new_position, new_round = check_new_round(new_position)
        return find_action(new_position, previous_position, new_round, cur_len, new_chance=True)
    elif SETS[position] == 0 and cur_len == needed and not new_chance:
        count_new_chance += 1
        new_position = previous_position + find_len_steps()
        new_position, new_round = check_new_round(new_position)
        return find_action(new_position, previous_position, new_round, cur_len, new_chance=True)
    else:
        return position

def simulation_game(cur_len):
    global count_steps, start_position, count_flat, count_auto, count_hotel, cycle, count_step_forward, \
        count_step_back, count_new_chance
    previous_position = start_position
    for _ in range(count_steps):
        current_position = previous_position + find_len_steps()
        current_position, new_round = check_new_round(current_position)
        previous_position = find_action(current_position, previous_position, new_round, cur_len)

    print(f" процент получения билета на кв за круг {count_flat / cycle}")
    print(f" процент получения билета на авто за круг {count_auto / cycle}")
    print(f" процент получения билета на отель за круг {count_hotel / cycle}")
    print(f" процент использования шаг вперед на один билет кв {count_step_forward / count_flat}")
    print(f" процент использования шаг назад на один билет кв {count_step_back / count_flat}")
    print(f" процент использования перебросить на один билет кв {count_new_chance / count_flat}")
    print(f" процент использования перебросить за круг {count_new_chance / cycle}")
    print(f" процент использования шаг вперед за круг {count_step_forward / cycle}")
    print(f" процент использования шаг назад за круг {count_step_back / cycle}")
    print(f" количетсво бросков кубика за круг {count_steps / cycle}")


def find_probability():
    pass


def controller():
    global start_position, count_flat, count_auto, count_hotel, cycle, count_step_forward, \
        count_step_back, count_new_chance
    for cur_len in range(2, 13):
        start_position = 0
        count_flat = 0
        count_auto = 0
        count_hotel = 0
        cycle = 0
        count_new_chance = 0
        count_step_forward = 0
        count_step_back = 0
        print(f"=== cur_len = {cur_len} ===")
        simulation_game(cur_len)


if __name__ == "__main__":
    # data_path = "C:/Users/Kamil/Aortic_valve/data/"
    # controller_path = "C:/Users/Kamil/Aortic_valve/code_aortic_valve/controller.yaml"
    # controller_dump = yaml_reader(controller_path)
    controller()