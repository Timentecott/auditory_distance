import random

number_of_trials = 90
number_of_blocks = 3
trials_per_block = number_of_trials // number_of_blocks
trial_list = []
#trials list format: [output, stim_type]

for output in [0, 1]:  # 0 = speakers, 1 = headphones
    for stim_type in [0, 1, 2]:  # 0 = noise, 1 = speech, 2 = music
        trials_per_combination = number_of_trials // 6  
        for _ in range(trials_per_combination):
            trial_list.append([output, stim_type])

# Shuffle the trial list
random.shuffle(trial_list)

