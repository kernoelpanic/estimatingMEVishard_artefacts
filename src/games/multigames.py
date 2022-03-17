#extract to .py file later
def payoff_first_n_steps(p=0.1,delta=0.9999,n=1):
    # return total payoff for the first n steps/rounds
    return ( p * (1-delta**n) )/(1 - delta)
#extract to .py file later
def plot_bitcoin_delta_approx(deltas=(0.9999,0.99999,0.999999),
                       power=0.1,
                       step=2016,
                       steps=2*13,
                       ticks=[ 1,8,2*6,2*9,2*12 ],
                       title_string="",
                       save_path=None,
                       ylim=None,
                       xlim=None):
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # https://matplotlib.org/api/markers_api.html
    marker = itertools.cycle((',','v', 'o', '+','.', 's', '*','1','D','x','^'))
    
    for d in deltas:
        X = [ n for n in np.arange(0,step*steps,step) ]
        reward = [ payoff_first_n_steps(p=power,delta=d,n=x) for x in X ]
        plt.plot(X, 
                 reward, 
                 marker=next(marker), 
                 linewidth=3, 
                 label="$\delta$ = " + str(d))
    
    plt.vlines(2016,ymin=0,ymax=2016*power,color="black",linestyle="dashed")
    plt.hlines(2016*power,xmin=0,xmax=2016,color="black",linestyle="dashed")
    
    plt.vlines(2016*2,ymin=0,ymax=2016*2*power,color="black",linestyle="dashed")
    plt.hlines(2016*2*power,xmin=0,xmax=2016*2,color="black",linestyle="dashed")
    
    plt.vlines(2016*2*6,ymin=0,ymax=2016*2*6*power,color="black",linestyle="dashed")
    plt.hlines(2016*2*6*power,xmin=0,xmax=2016*2*6,color="black",linestyle="dashed")
    
    ax.annotate('average rewards after 6 months', 
                xy=(2016*2*6, 2016*2*6*power), 
                xytext=(2016*2, 2900),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.vlines(2016*2*12,ymin=0,ymax=2016*2*12*power,color="black",linestyle="dashed")
    plt.hlines(2016*2*12*power,xmin=0,xmax=2016*2*12,color="black",linestyle="dashed")
    
    ax.annotate('average rewards after one year', 
                xy=(2016*2*12, 2016*2*12*power), 
                xytext=(2016*2*3, 5500),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.vlines(2016*2*24,ymin=0,ymax=2016*2*24*power,color="black",linestyle="dashed")
    plt.hlines(2016*2*24*power,xmin=0,xmax=2016*2*24,color="black",linestyle="dashed")
    
    ax.annotate('average rewards after two years', 
                xy=(2016*2*24, 2016*2*24*power), 
                xytext=(2016*2*9, 10100),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.vlines(2016*2*36,ymin=0,ymax=2016*2*36*power,color="black",linestyle="dashed")
    plt.hlines(2016*2*36*power,xmin=0,xmax=2016*2*36,color="black",linestyle="dashed")
    
    ax.annotate('average rewards after three years', 
                xy=(2016*2*36, 2016*2*36*power), 
                xytext=(2016*2*20, 15000),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # tidy up the figure
    ax.grid(True)
    #ax.legend(loc='center right', bbox_to_anchor=(0.8, 0.57))
    #ax.legend(loc='center right', 
    ax.legend(loc='upper left',
              bbox_to_anchor=(0.71, .55), # location of the legend
              framealpha=1.0)             # turn off transparency of legend
    #ax.set_title("")
    ax.set_xlabel("relative block height (in steps of 2016 blocks)")
    ax.set_ylabel("normalized block rewards for mined blocks")

    if ylim is not None:
        ax.set_ylim([0,ylim])
    if xlim is not None:
        ax.set_xlim([0,xlim]) 
    #plt.yticks(np.arange(0.0, 1.5, step=0.1))
    plt.xticks(np.arange(0, step*steps, step=step))
    
    #for label in ax.xaxis.get_ticklabels()[::2]:
    for tick,label in enumerate(ax.xaxis.get_ticklabels()[::]):
        if tick in ticks:        
            label.set_visible(True)
        else:
            label.set_visible(False)
    
    #plt.yscale('log')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    if save_path is not None:
        plt.savefig(save_path, dpi=200) 
    plt.show()
#extract to .py file later
def bribe_solve(p,p_new,delta,e=None):
    # Return the bribe required such that the total payoff without 
    # defect/attack actiton is equal to the total payoff with defect/attack action
    if e is not None:
        p_new=p*e 
    return ( ( delta * p_new ) - p ) / ( delta - 1)
#extract to .py file later
def bribe_solve_r(p,p_new,delta,e=None,r=0):
    # Return the bribe required such that the total payoff without 
    # defect/attack actiton is equal to the total payoff with defect/attack action
    # Added funds in the respective resource the attacker already has (r)
    if e is not None:
        p_new=p*e 
    return ( ( delta * p_new ) - p ) / ( delta - 1) - r*(e-1)
#extract to .py file later
def bribedrop_solve_r(p,p_new,delta,e=None,r=0):
    # Return the bribe required such that the total payoff without 
    # defect/attack actiton is equal to the total payoff with defect/attack action
    # Added funds in the respective resource the attacker already has (r)
    # Also the bribe is subject to the exchange rate reduction
    if e is not None:
        p_new=p*e
    return ( ( ( delta * p_new ) - p ) / ( delta - 1) - r*(e-1) )/e
#extract to .py file later
def EEV_honest(p0=0.1,p1=0.1,r0=0,r1=0,d0=0.9999,d1=0.9999):
    # Calculate EEV for infinite honest strategy, 
    # for two resources
    return (p0/(1-d0))+r0+(p1/(1-d1))+r1
#extract to .py file later
def EEV_attack(p0=0.1,p1=0,r0=0,r1=0,d0=0.9999,d1=0,e0=0,e1=0,E0=0,E1=0):
    # Calculate EEV for attack with side effects i.e., defect with bribes and then infinite honest strategy, 
    # for two resources
    return (d0*(p0*e0))/(1-d0) + (E0+r0)*e0 + (p1*e1)/(1-d1) + (E1+r1)*e1
#extract to .py file later
def bribe_solve_two_sympy(p0=0.1,p1=0.1,
                                  d0=0.9999,d1=0.9999,
                                  r0=0,r1=0,
                                  e0=1,e1=0,
                                  E0=0,E1=0,
                                  bribeIn="R0"):
    _E0 = Symbol('E0') # varepsilon/bribe in R_0
    _E1 = Symbol('E1') # varepsilon/bribe in R_1
    _d0 = Symbol('d0')
    _d1 = Symbol('d1')
    _p0 = Symbol('p0')
    _p1 = Symbol('p1')
    _r0 = Symbol('r0')
    _r1 = Symbol('r1')
    _e0 = Symbol('e0')
    _e1 = Symbol('e1')
    expr_zero = (  
                  (_d0 * (_p0 * _e0))/(1-_d0) +
                  (_E0 + _r0) * _e0 + 
                  (_p1 * _e1)/(1-_d1) +
                  (_E1 + _r1) * _e1 - 
                  (_p0/(1-_d0)) - 
                  _r0 - 
                  (_p1/(1-_d1)) -
                  _r1 ) 
    expr_repl = expr_zero.subs(_p0,p0).subs(_p1,p1)
    expr_repl = expr_repl.subs(_d0,d0).subs(_d1,d1)
    expr_repl = expr_repl.subs(_r0,r0).subs(_r1,r1)
    expr_repl = expr_repl.subs(_e0,e0).subs(_e1,e1)
    if bribeIn == "R0":
        expr_repl = expr_repl.subs(_E1,E1)
        rslt = float(solve(expr_repl, _E0)[0])
    elif bribeIn == "R1":
        expr_repl = expr_repl.subs(_E0,E0)
        rslt = float(solve(expr_repl, _E1)[0])
    else:
        assert False,'Solve for bribeIn="R0" or "R1"'
    return rslt
#extract to .py file later
def EEV_gains(p=0.1,e=1,d=0.9999):
    return (p * e)/(1 - d)
#extract to .py file later
def EEV_funds(r=0,e=1):
    return r * e
#extract to .py file later
def d_solve(p=0.1,R=1):
    # Given a EEV result R and power p, return delta
    return (R-p)/R
#extract to .py file later
def EEV_gains_after(p=0.1,e=1,d=0.9999):
    return (d * (p * e))/(1 - d)
#extract to .py file later
def EEV_funds_after(r=0,e=1,E=0):
    return (E + r) * e
#extract to .py file later
def d_solve_after(p=0.1,R=1):
    # Given a EEV result R and power p, return delta
    return R/(p+R)
#extract source to .py later
def plot_bar_payoff_after(b_p0=0,b_p1=0,
                          a_p0=0,a_p1=0,
                          b_d0=0,b_d1=0,
                          a_d0=0,a_d1=0,
                          b_r0=0,b_r1=0,
                          a_r0=0,a_r1=0,
                          b_e0=1,b_e1=1,
                          a_e0=1,a_e1=1,
                          E0=0,E1=0,
                          ymax_ax1=None,
                          ymax_ax2=None,
                          yticklist_ax1=None,
                          yticklist_ax2=None,
                          save_path=None,
                          skip_round=False,
                          ylabel='normalized block rewards',
                          xticklabels=["before","after"],
                          show_diff=True,
                          double_spend=0):
     
    x_names = ('$ C_0 $,$ C_1 $,sum (before)','$C_0$,$C_1$,sum (after)','total difference',)
    x_values = np.arange(len(x_names))
    
    
    b_payoff0 = EEV_gains(p=b_p0,e=b_e0,d=b_d0)
    b_payoff1 = EEV_gains(p=b_p1,e=b_e1,d=b_d1)
    b_funds0 = EEV_funds(r=b_r0,e=b_e0)
    b_funds1 = EEV_funds(r=b_r1,e=b_e1)
    b_sum = b_payoff0 + b_payoff1 + b_funds0 + b_funds1
        
    if skip_round:
        a_payoff0 = EEV_gains_after(p=a_p0,e=a_e0,d=a_d0)
        a_payoff1 = EEV_gains_after(p=a_p1,e=a_e1,d=a_d1)
    else:
        a_payoff0 = EEV_gains(p=a_p0,e=a_e0,d=a_d0)
        a_payoff1 = EEV_gains(p=a_p1,e=a_e1,d=a_d1)
    a_funds0 = EEV_funds(r=a_r0,e=a_e0)
    a_funds1 = EEV_funds(r=a_r1,e=a_e1)
    a_sum = a_payoff0 + a_payoff1 + a_funds0 + a_funds1 + E0*a_e0 + E1*a_e1

    gain = (a_sum+double_spend) - b_sum 
    
    print("sum (before) = ",b_sum)
    print("sum (after)  = ",a_sum)
    print("double spend = ",double_spend)
    print("total after  = ",a_sum + double_spend)
    print("gain         = ",gain)
    
    loss_funds0 = a_funds0 - b_funds0
    loss_funds1 = a_funds1 - b_funds1
    loss_payoff0 = a_payoff0 - b_payoff0
    loss_payoff1 = a_payoff1 - b_payoff1
    
    total_loss = 0
    if loss_funds0 < 0:
        total_loss += loss_funds0
    if loss_funds1 < 0:
        total_loss += loss_funds1
    if loss_payoff0 < 0:
        total_loss += loss_payoff0
    if loss_payoff1 < 0:
        total_loss += loss_payoff1
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    s = 0.35
    fig, ax1 = plt.subplots(figsize=(16*s,9*s))
                            #gridspec_kw = {'height_ratios':[10, 8]})
    
    #plt.subplots_adjust(wspace=0, 
    #                    hspace=0.025)
  
    width=0.5
    # currency 0 before 
    ax1.bar(0 - width/2,
            b_funds0,
            width=width,
            label='Funds $ r_0 $',
            color='lightgray', 
            align='edge', 
            hatch='oo', 
            edgecolor='black')    
    ax1.bar(0 - width/2, 
            b_payoff0,
            width=width,
            bottom=b_funds0,
            label='Expected Funds $ p_0 $',
            color='lightgray', 
            align='edge', 
            hatch='\\\\\\',
            edgecolor='black')
    
    # currency 1 before
    ax1.bar(0 - width/2,
            b_funds1,
            width=width,
            bottom=b_funds0 + b_payoff0,
            label='Funds $ r_1 $',
            color='lightblue', 
            align='edge', 
            hatch='oo', 
            edgecolor='black')    
    ax1.bar(0 - width/2, 
            b_payoff1,
            width=width,
            bottom=b_funds0 + b_payoff0 + b_funds1,
            label='Expected Funds $ p_1 $',
            color='lightblue', 
            align='edge', 
            hatch='\\\\\\',
            edgecolor='black')
    
    # currency 0 after
    if E0 != 0:
        ax1.bar(1 - width/2,
                E0*a_e0,
                width=width,
                label='Bribe in $ r_0 $',
                color='gold', 
                align='edge', 
                hatch='oo', 
                edgecolor='black')
    ax1.bar(1 - width/2,
            a_funds0,
            width=width,
            bottom=E0*a_e0,
            #label='Funds',
            color='lightgray', 
            align='edge', 
            hatch='oo', 
            edgecolor='black')    
    ax1.bar(1 - width/2, 
            a_payoff0,
            width=width,
            bottom=a_funds0 + E0*a_e0,
            #label='Payoffs',
            color='lightgray', 
            align='edge', 
            hatch='\\\\\\',
            edgecolor='black')
    
    # currrency 1 after
    if E1 != 0:
        ax1.bar(1 - width/2,
                E1*a_e1,
                width=width,
                bottom=a_funds0 + E0*a_e0 + a_payoff0,
                label='Bribe in $ r_1 $',
                color='gold', 
                align='edge', 
                hatch='oo', 
                edgecolor='black')
    ax1.bar(1- width/2,
            a_funds1,
            bottom=a_funds0 + E0*a_e0 + a_payoff0 + E1*a_e1,
            width=width,
            #label='Funds',
            color='lightblue', 
            align='edge', 
            hatch='oo', 
            edgecolor='black')    
    ax1.bar(1- width/2, 
            a_payoff1,
            bottom=a_funds0 + E0*a_e0 + a_payoff0 + a_funds1 + E1*a_e1,
            width=width,
            #label='Payoffs',
            color='lightblue', 
            align='edge', 
            hatch='\\\\\\',
            edgecolor='black')
    
    if show_diff and gain > 0:
        ax1.bar(1- width/2, 
                gain,
                width=width,
                bottom=b_funds0 + b_payoff0 + b_funds1 + b_payoff1,
                label='Gain',
                color='mediumseagreen', 
                align='edge', 
                hatch='++', 
                edgecolor='black')
        
    if show_diff and gain < 0:
        ax1.bar(1- width/2, 
                b_sum - a_sum,
                width=width,
                bottom=a_funds0 + E0*a_e0 + a_payoff0 + a_funds1 + E1*a_e1 + a_payoff1,
                label='Loss',
                color='lightcoral', 
                align='edge', 
                hatch='XXX', 
                edgecolor='black')

    if double_spend > 0:
        ax1.bar(1- width/2, 
                double_spend,
                bottom=a_funds0 + E0*a_e0 + a_payoff0 + a_funds1 + E1*a_e1 + a_payoff1,
                width=width,
                label='Double-spend $r_3$',
                color='mediumseagreen', 
                align='edge', 
                hatch='++',
                edgecolor='black')
    
    if ymax_ax1 is not None:
        ax1.set_ylim(0, ymax_ax1)
        #ax2.set_ylim(-(ymax_ax2), 0)
        
    if ymax_ax1 is not None and yticklist_ax1 is None:
        yticks = [ 0,ymax_ax1//5,(ymax_ax1//5)*2,(ymax_ax1//5)*3,(ymax_ax1//5)*4, (ymax_ax1//5)*5 ]
        ax1.yaxis.set_ticks(yticks)
        yticks = [ -ymax_ax2//5,(-ymax_ax2//5)*2,(-ymax_ax2//5)*3,(-ymax_ax2//5)*4, (-ymax_ax2//5)*5,0 ]
        #ax2.yaxis.set_ticks(yticks)
   
    if ymax_ax1 is not None and yticklist_ax1 is not None:
        ax1.yaxis.set_ticks(yticklist_ax1)
        #ax2.yaxis.set_ticks(yticklist_ax2)
   
    #ax2.yaxis.get_major_ticks()[-1].label1.set_visible(False)
        
    # ax.bar(, color='r')
    ax1.set_xlim(0 - 0.5,3)
    #ax2.set_xlim(0 - 0.5,3)
    
    ax1.xaxis.set_ticks([0,1])
    ax1.set_xticklabels(xticklabels)
    
    #ax2.set_xticks(x_values)
    #ax2.set_xticklabels(x_names)
    #ax2.get_xticklabels()[0].set_ha("left")
    #ax2.get_xticklabels()[-1].set_ha("right")
    
    #ax2.xaxis.set_major_formatter(FuncFormatter(lambda x,_: str(int())))
    #ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_: str(int(abs(x)))))
    #ax2.yaxis.get_major_ticks()[-1].label1.set_visible(False) # undisplay 0 on y axis of ax2
    
    ax1.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    #ax2.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    
    #ax2.set_xlabel('')    
    #ax2.set_ylabel(' '*45 + 'expected normalized block rewards')
    ax1.set_ylabel(ylabel)
    
    plt.figlegend(loc='upper right',
                  bbox_to_anchor=(0.9, 0.89),
                  framealpha=1,
                  ncol=1)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
