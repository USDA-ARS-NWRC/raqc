import numpy as np
import math

#plottables

class Plotables:
    "this does some stuff"



    def cb_readable(self, list_or_array, flag, num_ticks):
        ''' this just gives tick marks plotting '''
        # A = array
        # L = List (with mn and max)
        # get vmin and max, and cbar ticks relative to those values
        # tkmn, tkmx = colormap min, max scalers
        # cbmn, cbmx = colorbar tick min, max scalers
        # NOTE: tkmx HARDCODED. Just change this to desired clipping value

        if flag.upper() == 'A':
            mn = np.nanmin(list_or_array)
            mx = np.nanmax(list_or_array)
        elif flag.upper() == 'L':
            mn = list_or_array[0]
            mx = list_or_array[1]


        val_range = mx - mn

        shift_multiplier = 0.1
        if mn < 0:
            mn -= mn * shift_multiplier
        else:
            mn += mn * shift_multiplier
        if mx < 0:
            mx += mx * shift_multiplier
        else:
            mx -= mx * shift_multiplier

        # Go through val range scenarios
        if val_range < 0.1:
            self.cb_range = np.array([mn, mx])
        else:
            cb_range = np.linspace(mn, mx, num_ticks)
            if val_range < 100*.5:
                self.cb_range = self.range_cust(cb_range, 1)
            elif val_range < 100*1:
                self.cb_range = self.range_cust(cb_range, 0)
            elif val_range < 100*10:
                self.cb_range = self.range_cust(cb_range, -1)
            elif val_range < 1000*10:
                self.cb_range = self.range_cust(cb_range, -2)
        # rd = input('min = {:.2} max = {:2}  \n enter rounding precision as integer  \n ex) -1 = nearest 10, 2 = two decimal places: \n '.format(mn,mx))

    def dist_subP(self, num_subP):
        '''sets distribution of subplot rows and columns based on number of
        subplots'''
        self.panel_flag = not ((num_subP % 9) % 2 == 0)  #Let's user know that there is one empty subplot
        # which should not be plotted later on as slicing will throw errors
        num_subP_t = num_subP
        imgs = math.ceil(num_subP / 9)
        rw = [0] * imgs
        cl = [0] * imgs
        for i in range(imgs):
            if imgs > 1:
                rw[i] = 3
                cl[i] = 3
                imgs = imgs - 1  #countergcdf
            else:
                num_subP = num_subP_t - 9*i
                if num_subP == 1:
                    rw[i] = 1
                    cl[i] = 1
                elif num_subP < 9:
                    rw[i] = math.ceil(num_subP/2)
                    cl[i] = 2
                elif num_subP == 9:
                    rw[i] = 3
                    cl[i] = 3
            self.row = rw
            self.col = cl

    def range_cust(self, np_obj, rd):
        # ZRU  5/20/19  This needs work.  Just ensure min rounds up and max rounds down
        np_objT = np_obj   #temp object
        for i, j in enumerate(np_obj):
            if rd > 0:
                np_objT[i] = round(j, rd)
            elif rd <= 0:
                np_objT[i] = int(round(j, rd))  # reduce numbers to display in colorbar
        return np_objT

    def marks_colors(self):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import cmocean
        import copy

        #plt.Set1_r is a colormap
        # np.linspace will pull just one value in this case
        # colorsbad = plt.cm.Set1_r(np.linspace(0., 1, 1))
        colorsbad = np.array([0.9, 0.9, 0.9, 1]).reshape((1, 4))
        colors1 = cmocean.cm.matter_r(np.linspace(0., 1, 126))
        colors2 = plt.cm.Blues(np.linspace(0, 1, 126))
        colors = np.vstack((colorsbad, colors1, colorsbad, colorsbad, colors2, colorsbad))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        mymap.set_bad('white', 1)
        cmap = copy.copy(mymap)
        self.cmap_marks = cmap

    def set_zero_colors(self, zeros):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import cmocean
        import copy

        if zeros == 0:  # this sets zero to white, BUT stretches with large data value range
            colorsbad = np.array([[0,0,0,0]])
            colors1 = cmocean.cm.matter_r(np.linspace(0., 1, 255))
            # colors1 = plt.cm.gist_stern(np.linspace(0., 1, 255))
            colors = np.vstack((colorsbad, colors1))
            mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            mymap.set_bad('gray', 1)
        elif zeros == 1:  # does not alter plt.cm colormap, just sets zero to white when vmin specified
            colors = plt.cm.gist_stern(np.linspace(0., 1, 256))
            mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            mymap.set_bad('gray', 1)
            mymap.set_under('white')
        cmap = copy.copy(mymap)
        self.cmap_choose = cmap
