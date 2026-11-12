import numpy as np
import pandas as pd
import pandas as pn
import matplotlib.pyplot as plt

results = pd.read_csv('results.csv', index_col=[*range(4)])
results_till_end = results.query("Project>0") # Students that did not drop out in the middle

bins = np.arange(11) + 8.5
fig, ax = plt.subplots()
ax.hist( results_till_end.query("Group=='MEIecfs'")['Grade'], bins=bins, align='mid', alpha=0.8, label='MEIecfs' )
ax.hist( results_till_end.query("Group=='MEIswe'")['Grade'], bins=bins, align='mid', alpha=0.8, label='MEIswe' )
ax.set_xlabel('Grade', labelpad=8)
ax.set_ylabel('Number of students', labelpad=8)
ax.legend()
fig.tight_layout()
fig.savefig('figure/results.png', dpi=300)
