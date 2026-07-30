#2afc 
#open pyscopy
#display prompt for participant number
#display practice instructions
#run practice trials with feedback (these will be easy)
#display main instructions
#run main trails without feedback

#each trial:
#display three clickable buttons '1' '2' and '3' and instructions "click the buttons to hear sounds. once you have decided which sounds furthest away,
# press the correstponding number on the keyboard (1, 2, or 3) to indicate your choice" 
#if the participant presses 1 2 or 3 before playing all sounds, display a message "please listen to all sounds before making a choice"
# allow the participant to click the buttons to play the sounds as many times as they want
#each button corresponds to a sound file
#when the participant clicks a button, play the corresponding sound file
#when the participant presses 1, 2, or 3 after listening to all sounds, record their choice and the time taken to make the choice 

#instructions:  which is the odd one out? which is furthest away? which is closest? which is quietest? which is loudest?  

#where the variable is loudness, the sounds will differ only on loudness with one sound louder than the other two
#where the variable is quietness, the sounds will differ only on loudness with one sound quieter than the other two
#where the variable is closeness, the sounds will differ only on distance with one sound further away than the other two
#where the variable is farness, the sounds will differ only on distance  with one sound further than the other two



#The question can be congruent, inconguent, or outlier. Where the question is congruent, the question will match the variable 
#(e.g. "which is loudest?" when the variable is loudness). 
#Where the question is incongruent, the question will not match the variable (e.g. "which is the quietest?" when all the variable is farness).
#for incongruent questions, the question can only be loudness in place of nearness, queitness in place of farness, nearness in place of loudness, and farness in place of quietness.
#where the question is outlier, it's just "which is the odd one out?" 

#this means that all possible trials are:
#question: outlier, variable: nearness (outlier)
#question: outlier, variable: farness (outlier)

#question: outlier, variable: loudness (outlier)
#question: outlier, variable: quietness (outlier)

#question: loudest, variable: loudness (congruent)
#question: quietest, variable: quietness (congruent)

#question: closest, variable: closeness (congruent)
#question: furthest, variable: farness (congruent)

#question: loudest, variable: closeness (incongruent)
#question: quietest, variable: farness (incongruent)

#question: closest, variable: loudness (incongruent)
#question: furthest, variable: quietness (incongruent)


#to start with all files will be "C:\Users\tim_e\source\repos\auditory_distance\experiment_3\stimuli\brown_noise_5s.wav"
#and there will only be one trial to begin with. the correct answer will be 3
#results will be saved to /results/[participant_number].csv including response time and whether the answer was correct or not. 

