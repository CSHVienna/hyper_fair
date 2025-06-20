import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objs as go


def plot_ranking(ranking, alpha=None, ci=None, omega=1, ax=None):
    """
    Plots the cumulative proportion of protected candidates in a ranking, along with confidence intervals and theoretical curves.
    
    Parameters:
        ranking (np.ndarray or list): A binary array indicating the ranking of items (1 for protected group, 0 for unprotected).
        alpha (float): Significance level for the confidence interval (e.g., 0.05 for 95% CI). Default is None, which means no CI is plotted.
        ci (str): Type of confidence interval to plot:
            - 'lower': Lower confidence bound.
            - 'upper': Upper confidence bound.
            - 'two-sided': Both lower and upper bounds.
            - None: No confidence interval is plotted. This is the default.
        omega (float, optional): Odds ratio for the noncentral hypergeometric distribution. If 1, uses the standard hypergeometric distribution.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created.

    Notes:
        - The function visualizes the empirical cumulative proportion of protected candidates in the ranking.
        - Theoretical mean and confidence intervals are computed using (noncentral) hypergeometric distributions.
        - Shaded regions indicate the confidence intervals and theoretical bounds.
    """

    assert isinstance(ranking, list) or (isinstance(ranking, np.ndarray) and ranking.ndim == 1), "ranking must be a 1D numpy array"
    assert isinstance(alpha, (float, type(None))) or (isinstance(alpha, float) and 0 < alpha < 1), "alpha must be a float between 0 and 1 or None"
    assert ci in ['lower', 'upper', 'two-sided', None], "ci must be one of ['lower', 'upper', 'two-sided', None]"
    assert isinstance(omega, (int, float)) and omega > 0, "omega must be a positive number"
    assert ax is None or isinstance(ax, plt.Axes), "ax must be a matplotlib Axes object or None"

    n = len(ranking)
    n_protected = np.sum(ranking)
    proportions = np.cumsum(ranking) / np.arange(1, n+1)
    p = proportions[-1]
    x_up = np.arange(p,1+1/n,1/n)
    y_up_curve = p/x_up
    x_down = np.arange(1-p,1+1/n,1/n)
    y_down_curve = 1-(1-p)/(x_down)
    if ci == 'lower' and alpha:
        ci_lower_hyp = [stats.nchypergeom_wallenius.ppf(alpha, n, n_protected, j, omega)/j for j in range(1,n+1)]
        ci_upper_hyp = 1
    elif ci == 'upper' and alpha:
        ci_lower_hyp = 0
        ci_upper_hyp = [stats.nchypergeom_wallenius.ppf(1-alpha, n, n_protected, j, omega)/j for j in range(1,n+1)]
    elif ci == 'two-sided' and alpha:
        ci_lower_hyp = [stats.nchypergeom_wallenius.ppf(alpha/2, n, n_protected, j, omega)/j for j in range(1,n+1)]
        ci_upper_hyp = [stats.nchypergeom_wallenius.ppf(1-alpha/2, n, n_protected, j, omega)/j for j in range(1,n+1)]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.set_xticks(np.arange(0,1.1,1/5))
    ax.set_xticklabels([str(int(i*100))+'%' for i in np.arange(0,1.1,1/5)])
    ax.set_xlim(1/n,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Top % of candidates')
    ax.set_title(f'Ranking of candidates ($n={n}$, $n_p={int(n_protected)}$)')
    ax.set_ylabel(f'Proportion of protected candidates')
    ax.plot(np.arange(1/n,1+1/n,1/n), proportions, '#018571', label = 'Empirical ranking')
    if omega != 1:
        mean_curve = [stats.nchypergeom_wallenius.mean(n, n_protected, j, omega)/j for j in range(1,n+1)]
        ax.plot(np.arange(1/n,1+1/n,1/n), mean_curve, "#0774CD", linestyle = '--', label = 'Mean')
    if ci is not None and alpha is not None: 
        ax.plot([0,1], [p,p], color='k', linestyle = '--', linewidth = .8)
        alpha_2_digits = float('%.2g' % alpha)
        ax.fill_between(np.arange(1/n,1+1/n,1/n), ci_lower_hyp, ci_upper_hyp, color = '#de8f05', alpha = .5, label = f'{100-alpha_2_digits*100}% CI', edgecolor='black', linewidth=0.5, linestyle='--')
    else:
        ax.plot([0,1], [p,p], color='k', linestyle = '--', linewidth = .8, label = f'Total proportion of protected candidates: {int(p*100)}%')
    ax.set_axisbelow(True)
    ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.fill_between(x_up,y_up_curve, 1, color = 'gray', linestyle = '-',edgecolor='black')
    ax.fill_between(x_down,y_down_curve, 0, color = 'gray', linestyle = '-', edgecolor='black')
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True)
    plt.rcParams.update({'font.size': 12})


def plot_only_ci(n, n_protected, alpha, ci, omega=1, ax=None):
    """
    Plots the confidence intervals for the proportion of protected candidates in a ranking.

    Parameters:
        n (int): Total number of candidates.
        n_protected (int): Number of protected candidates.
        alpha (float): Significance level for the confidence interval (e.g., 0.05 for 95% CI).
        ci (str): Type of confidence interval to plot:
            - 'lower': Lower confidence bound.
            - 'upper': Upper confidence bound.
            - 'two-sided': Both lower and upper bounds.
        omega (float, optional): Odds ratio for the noncentral hypergeometric distribution. If 1, uses the standard hypergeometric distribution.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created.
    """

    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    assert isinstance(n_protected, int) and 0 < n_protected < n, "n_protected must be an integer between 0 and n (excluded)"
    assert isinstance(alpha, float) and 0 < alpha < 1, "alpha must be a float between 0 and 1"
    assert ci in ['lower', 'upper', 'two-sided'], "ci must be one of ['lower', 'upper', 'two-sided']"
    assert isinstance(omega, (int, float)) and omega > 0, "omega must be a positive number"
    assert ax is None or isinstance(ax, plt.Axes), "ax must be a matplotlib Axes object or None"

    p = n_protected/n
    x_up = np.arange(p,1+1/n,1/n)
    y_up_curve = p/x_up
    x_down = np.arange(1-p,1+1/n,1/n)
    y_down_curve = 1-(1-p)/(x_down)
    if ci == 'lower':
        ci_lower_hyp = [stats.nchypergeom_wallenius.ppf(alpha, n, n_protected, j, omega)/j for j in range(1,n+1)]
        ci_upper_hyp = 1
    elif ci == 'upper':
        ci_lower_hyp = 0
        ci_upper_hyp = [stats.nchypergeom_wallenius.ppf(1-alpha, n, n_protected, j, omega)/j for j in range(1,n+1)]
    elif ci == 'two-sided':
        ci_lower_hyp = [stats.nchypergeom_wallenius.ppf(alpha/2, n, n_protected, j, omega)/j for j in range(1,n+1)]
        ci_upper_hyp = [stats.nchypergeom_wallenius.ppf(1-alpha/2, n, n_protected, j, omega)/j for j in range(1,n+1)]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.plot([0,1], [p,p], color='k', linestyle = '--', linewidth = .8)
    ax.set_xticks(np.arange(0,1.1,1/5))
    ax.set_xticklabels([str(int(i*100))+'%' for i in np.arange(0,1.1,1/5)])
    ax.set_xlim(1/n,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Top % of candidates')
    ax.set_title(f'Ranking of candidates ($n={n}$, $n_p={int(n_protected)}$)')
    ax.set_ylabel(f'Proportion of protected candidates')
    if omega != 1:
        mean_curve = [stats.nchypergeom_wallenius.mean(n, n_protected, j, omega)/j for j in range(1,n+1)]
        ax.plot(np.arange(1/n,1+1/n,1/n), mean_curve, "#0774CD", linestyle = '--', label = 'Mean')
    alpha_2_digits = float('%.2g' % alpha)
    ax.fill_between(np.arange(1/n,1+1/n,1/n), ci_lower_hyp, ci_upper_hyp, color = '#de8f05', alpha = .5, label = f'{100-alpha_2_digits*100}% CI', edgecolor='black', linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.fill_between(x_up,y_up_curve, 1, color = 'gray', linestyle = '-',edgecolor='black')
    ax.fill_between(x_down,y_down_curve, 0, color = 'gray', linestyle = '-', edgecolor='black')
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True)
    plt.rcParams.update({'font.size': 12})




def plot_ranking_plotly(ranking, alpha=None, ci=None, omega=1):
    """
    Plots the cumulative proportion of protected candidates in a ranking, along with confidence intervals and theoretical curves using Plotly.
    Returns a plotly.graph_objs.Figure object.
    """
    assert isinstance(ranking, list) or (isinstance(ranking, np.ndarray) and ranking.ndim == 1), "ranking must be a 1D numpy array"
    assert isinstance(alpha, (float, type(None))) or (isinstance(alpha, float) and 0 < alpha < 1), "alpha must be a float between 0 and 1 or None"
    assert ci in ['lower', 'upper', 'two-sided', None], "ci must be one of ['lower', 'upper', 'two-sided', None]"
    assert isinstance(omega, (int, float)) and omega > 0, "omega must be a positive number"

    n = len(ranking)
    n_protected = np.sum(ranking)
    proportions = np.cumsum(ranking) / np.arange(1, n+1)
    p = proportions[-1]
    x_vals = np.arange(1/n, 1+1/n, 1/n)

    traces = []

    # Empirical ranking
    traces.append(go.Scatter(
        x=x_vals,
        y=proportions,
        mode='lines+markers',
        name='Empirical ranking',
        line=dict(color='#018571')
    ))

    # Mean curve for omega != 1
    if omega != 1:
        mean_curve = [stats.nchypergeom_wallenius.mean(n, n_protected, j, omega)/j for j in range(1, n+1)]
        traces.append(go.Scatter(
            x=x_vals,
            y=mean_curve,
            mode='lines',
            name='Mean',
            line=dict(color='#0774CD', dash='dash')
        ))

    # Confidence intervals
    ci_lower_hyp, ci_upper_hyp = None, None
    if ci == 'lower' and alpha:
        ci_lower_hyp = [stats.nchypergeom_wallenius.ppf(alpha, n, n_protected, j, omega)/j for j in range(1, n+1)]
        ci_upper_hyp = [1]*n
    elif ci == 'upper' and alpha:
        ci_lower_hyp = [0]*n
        ci_upper_hyp = [stats.nchypergeom_wallenius.ppf(1-alpha, n, n_protected, j, omega)/j for j in range(1, n+1)]
    elif ci == 'two-sided' and alpha:
        ci_lower_hyp = [stats.nchypergeom_wallenius.ppf(alpha/2, n, n_protected, j, omega)/j for j in range(1, n+1)]
        ci_upper_hyp = [stats.nchypergeom_wallenius.ppf(1-alpha/2, n, n_protected, j, omega)/j for j in range(1, n+1)]

    if ci_lower_hyp is not None and ci_upper_hyp is not None:
        traces.append(go.Scatter(
            x=x_vals,
            y=ci_upper_hyp,
            fill=None,
            mode='lines',
            line=dict(color='rgba(222,143,5,0.5)'),
            showlegend=False
        ))
        traces.append(go.Scatter(
            x=x_vals,
            y=ci_lower_hyp,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(222,143,5,0.5)'),
            name=f'{100-(alpha if ci!="two-sided" else alpha/2)*100:.1f}% CI'
        ))

    # Total proportion line
    traces.append(go.Scatter(
        x=[0, 1],
        y=[p, p],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name=f'Total proportion: {int(p*100)}%'
    ))

    # Upper impossible region: y > p/x for x in [p, 1]
    x_up = np.arange(p, 1+1/n, 1/n)
    y_up_curve = p / x_up
    traces.append(go.Scatter(
        x=np.concatenate([x_up, x_up[::-1]]),
        y=np.concatenate([y_up_curve, np.ones_like(y_up_curve)]),
        fill='toself',
        fillcolor='rgba(128,128,128,1)',
        line=dict(color='rgba(128,128,128,0)'),
        showlegend=True,
        name='Impossible region'
    ))

    # Lower impossible region: y < 1-(1-p)/x for x in [1-p, 1]
    x_down = np.arange(1-p, 1+1/n, 1/n)
    y_down_curve = 1 - (1-p) / x_down
    traces.append(go.Scatter(
        x=np.concatenate([x_down, x_down[::-1]]),
        y=np.concatenate([np.zeros_like(y_down_curve), y_down_curve[::-1]]),
        fill='toself',
        fillcolor='rgba(128,128,128,1)',
        line=dict(color='rgba(128,128,128,0)'),
        showlegend=False,
        name=''
    ))

    # Layout
    layout = go.Layout(
        title=f'Ranking of candidates (n={n}, n_p={int(n_protected)})',
        xaxis=dict(
            title='Top % of candidates',
            tickvals=np.arange(0, 1.1, 0.2),
            ticktext=[f'{int(i*100)}%' for i in np.arange(0, 1.1, 0.2)],
            range=[1/n, 1]
        ),
        yaxis=dict(
            title='Proportion of protected candidates',
            range=[0, 1]
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.7)'
        ),
        template='plotly_white'
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig