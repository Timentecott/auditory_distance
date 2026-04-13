
#data extraction
data_folder = Path(r"C:\Users\tim_e\source\repos\auditory_distance\results\pilot_26_01")
output_folder = Path(r"C:\Users\tim_e\source\repos\auditory_distance\analysis_results")
import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM

#for every csv files in data_folder which have the ending _trial.csv
#import the file 
data_files = list(data_folder.glob('*_trial.csv'))
print



#presentation: 1=headphone, 2=loudspeaker
#stimulus 1=ISTS 2=environmental 3=noise 
dataframe = pd.DataFrame({'presentation':[],
						  'stimulus':[],
						  'accuracy':[]})
						  })
#calculate means

#complete repeated measures anova 