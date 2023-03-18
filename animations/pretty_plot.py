import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

plt.style.use('seaborn-dark-palette')
# plt.style.available;

def plot(xy,
         major_ticks,
         minor_ticks,
         _figsize=(4, 4),
         title='',
         xlabel='',
         ylabel='',
         xlims=None,
         ylims=None,
         xticks=None,
         yticks=None,
         **plot_kwargs):
    '''
        pretty 1x1 plot (but not yet beautiful)
    '''
    fig, ax = plt.subplots(1, 1)
    # plt.figure(figsize=_figsize, dpi=80)
    fig.set_size_inches(_figsize)
    # fig.set_dpi(200)

    if len(xy) >= 2:
        ax.plot(*xy, **plot_kwargs)
    else:
        ax.plot(xy, **plot_kwargs)
    
    # _____________________ limits _____________________
    # Set axis ranges; by default this will put major ticks every 25.
    if xlims:
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        
    # _____________________ ticks _____________________
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)

    # _____________________ Grid _____________________
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(major_ticks[0]))
    ax.yaxis.set_major_locator(MultipleLocator(major_ticks[1]))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks[0]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks[1]))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
