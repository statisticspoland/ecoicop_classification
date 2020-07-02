"""
The script below presents the way how to plot the histogram of the ecoicop categories
included in the main dataset and save it into image file.
"""
import pandas as pd
import matplotlib.pyplot as plt

# Plot the distribution of ecoicop categories from the main dataset
df = pd.read_excel("products_allshops_dataset.xlsx", names=['produkt', 'kategoria'])
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.41, right=0.99, top=0.80, left=0.08, wspace=0.2, hspace=0.2)
fig.suptitle('ECOICOP categories distribution')
df.groupby('kategoria').produkt.count().plot.bar(ylim=0, color='#39a257')
plt.xticks(fontsize=7, rotation=90)
fig.set_size_inches((11, 5), forward=False)
plt.savefig('ecoicop_histogram.png')
