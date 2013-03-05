colors = [ '#bbb12d', '#1480fa', '#bd2309' ]
           #'#faf214', '#2edfea', '#ea2ec4',
           #'#14fa2f', '#ea2e40', '#cdcdcd', '#577a4d', '#2e46c0', '#f59422',
           #'#219774', '#8086d9', '#000000' ]
import matplotlib.colors as col
import matplotlib.cm as cm
cm.register_cmap(cmap=col.ListedColormap(colors, 'classes'))
