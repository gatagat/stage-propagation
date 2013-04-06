import numpy as np
colors = range(256)
colors[:6] = (np.array([[214, 0, 0], [255, 145, 36], [246, 209, 60], [48, 194, 0], [0, 168, 224], [138, 49, 246]])/255.).tolist()
#colors[:6] = [ '#bbb12d', '#1480fa', '#bd2309', '#76BB2D', '#30D1C6', '#D130A1' ]
           #'#14fa2f', '#ea2e40', '#cdcdcd', '#577a4d', '#2e46c0', '#f59422',
           #'#219774', '#8086d9', '#000000' ]
colors[6:] = [ '#cccccc' ] * 250
import matplotlib.colors as col
import matplotlib.cm as cm
cm.register_cmap(cmap=col.ListedColormap(colors, 'classes'))
