'''
Testing Early Reward Scenarios

Plotting 3 Figures (#rd Figure in Dopamine Analysis)

Testing VS Lesion - 2 Figures
'''

from dana import *
import numpy as np
# import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import seaborn as  sns
import sys

'''
Experiment parameters
'''
time = 500.0  # Total time of trial
dt = 1  # Time step
trial_num = 16  # Trial number

'''
Population sizes
'''
VTA_dop_size = 10
VTA_GABA_size = 5
PPN_RD_size = 4
PPN_FT_size = 4
BLA_size = 1
CE_size = 1
LH_size = 1
IT_size = 4
NAc_size = 1
OFC_size = 1
VP_size = 1

'''
Time Constantsll
'''
tau_dop = 5
tau_GABA = 20
tau_PPN_FT = 20
tau = 10
tau_amyg = 10
tau_adaptation = 5
tau_adaptation_amyg = 10
tau_ppn = 5
tau_NAc = 1

'''
Adaptation constants
'''
K_PPN = 1
K_Amyg = 1

'''Other constants
'''
mag_learning_rate = .002  # Magntiude learning rate of stimuli
B = 0.2  # Backgroundfiring rate
K = 1
time_learning_rate = .4  # Time Learning rate of stimuli

'''Helper functions
'''


# Noise level
def noise():
    return np.random.uniform(-0.01, 0.01)


# Weights of timelearning updation

def time_weight_update(W, bound, value, dop, check):
    if check == 0:
        return 0
    elif bound == 1:
        print ("here")
        W = (bound) * (W * (value)) / (1 - value)
        # print "weight update", round(float(W.toarray()[0]), 6),"current value",value
        return round(float(W.toarray()[0]), 6)
        # return W
    elif bound == 0:
        print (dop)
        W = -(time_learning_rate * W)
        # print "Bound not touched"
        # print "weight update", round(float(W.toarray()[0]), 6), "current value", value
        return round(float(W.toarray()[0]), 6)
    else:
        return 0

def threshold(x, value):
    if x < value:
        return 0
    else:
        return 1


'''
Defining population membrane potentials
'''
LH = zeros((LH_size,), '''V=I;I''')

IT = zeros((IT_size), '''V=I;I''')

BLA = zeros((BLA_size,),
            '''dadapt_mod/dt=(-adapt_mod+np.maximum(I_IT,0))/tau_adaptation_amyg;dV/dt=(-V + np.maximum(I_IT - K_Amyg*adapt_mod,0) + noise())/tau_amyg;U=np.maximum(V,0);I_IT;I_Dop;US;I_Dop_Max;I_LH;U_max;''')

CE = zeros((CE_size,),
           '''dadapt_exc/dt=(-adapt_exc+np.maximum(I_BLA,0))/tau_adaptation;dV/dt=(-V + np.maximum(I_BLA - K*adapt_exc,0) + noise())/(tau_amyg*2);U=np.maximum(V,0);I_BLA''')

VTA_dop = zeros((VTA_dop_size,),
                '''dadapt_exc/dt =  (-adapt_exc+np.maximum(I_PPN_RD,0))/tau_adaptation;dV/dt = (-V +np.maximum(I_PPN_RD- K*adapt_exc,0) - I_VTA_GABA  + noise())/tau_dop;U=np.maximum(V,0)+B; I_PPN_RD;I_VTA_GABA''')

PPN_RD = zeros((PPN_RD_size,),
               '''dadapt_exc/dt =  (-adapt_exc+np.maximum(I_LH+I_CE,0))/tau_adaptation;dV/dt= (-V + np.maximum(I_LH+I_CE- K_PPN*adapt_exc,0)+ noise())/tau_ppn;U=np.maximum(V,0);I_LH;I_CE''')

VTA_GABA = zeros((VTA_GABA_size,),
                 '''dV/dt= (-V + np.maximum(I_PPN_FT-I_NAc,0) + noise())/(tau_ppn);U=np.maximum(V,0);I_NAc;I_PPN_FT''')

PPN_FT = zeros((PPN_FT_size,), '''dV/dt=(-V+np.maximum(I_CE+U-I_PPN_RD,0))/(tau_ppn);U=np.maximum(V,0);I_CE;I_PPN_RD''')


NAc = zeros((NAc_size),'''dadapt_exc/dt =  (-adapt_exc+I_LH)/15;dV/dt=(I_OFC- V*adapt_exc)/tau_NAc;U=np.maximum(np.ceil(I_OFC-adapt_exc)-np.clip(V,0,1),0);I_LH;I_OFC;NAc_max;Bound;Value;check;I_Dop;US''')

# Working
# NAc=zeros((NAc_size),'''dadapt_exc/dt =  (-adapt_exc+I_LH)/15;dV/dt=(I_OFC- V*LH)/tau_NAc;U=np.maximum(np.ceil(I_OFC-LH)-np.clip(V,0,1),0);I_LH;I_OFC;NAc_max;Bound;Value;check;I_Dop;US''')

OFC = zeros((OFC_size), '''dV/dt=(-V + I_IT)/tau_NAc;U=np.maximum(V,0);I_IT''')

'''
Connections
'''

SparseConnection(BLA('U'), CE('V'), np.full((CE_size, BLA_size), .1))

SparseConnection(LH('V'), PPN_RD('I_LH'), np.full((PPN_RD_size, LH_size), 1.2))

SparseConnection(VTA_dop('U'), BLA('I_Dop'), np.ones((BLA_size, VTA_dop_size)))

SparseConnection(VTA_dop('U'), NAc('I_Dop'), np.ones((NAc_size, VTA_dop_size)))

SparseConnection(LH('V'), BLA('I_LH'), np.full((BLA_size, LH_size), 1))

Y = SparseConnection(IT('V'), BLA('I_IT'), np.full((BLA_size, IT_size), .01),
                     'dW/dt=(post.US*mag_learning_rate*pre.V*np.maximum(post.I_LH-post.U_max,0))')

SparseConnection(CE('U'), PPN_RD('I_CE'), np.full((PPN_RD_size, CE_size), 3))

SparseConnection(CE('U'), PPN_FT('I_CE'), np.full((PPN_RD_size, CE_size), 0.3))

SparseConnection(PPN_RD('U'), PPN_FT('I_PPN_RD'), np.full((PPN_FT_size, PPN_FT_size), .8))

SparseConnection(PPN_RD('U'), VTA_dop('I_PPN_RD'), np.full((VTA_dop_size, PPN_RD_size), 1))

SparseConnection(VTA_GABA('U'), VTA_dop('I_VTA_GABA'), np.full((VTA_dop_size, VTA_GABA_size), .2))

SparseConnection(LH('V'), NAc('I_LH'), np.ones((OFC_size, LH_size)))

SparseConnection(IT('V'), OFC('I_IT'), np.full((OFC_size, IT_size), 0.25))


X = SparseConnection(OFC('U'), NAc('I_OFC'), np.full((OFC_size, NAc_size), .006),
                     'dW/dt=time_weight_update(W,post.Bound,post.Value,threshold(post.I_Dop-2,0.1),post.check)')



SparseConnection(PPN_FT('U'), VTA_GABA('I_PPN_FT'), np.full((VTA_GABA_size, PPN_FT_size), .28))

SparseConnection(NAc('U'), VTA_GABA('I_NAc'), np.full((VTA_GABA_size, NAc_size), 1))

'''
Setting up monitors
'''

PPN_RD_record = np.zeros((trial_num, (int(time / dt) + 1)))
LH_record = np.zeros((trial_num, (int(time / dt) + 1)))
VTA_dop_record = np.zeros((trial_num, (int(time / dt) + 1)))
VTA_GABA_record = np.zeros((trial_num, (int(time / dt) + 1)))
BLA_record = np.zeros((trial_num, (int(time / dt) + 1)))
IT_record = np.zeros((trial_num, (int(time / dt) + 1)))
CE_record = np.zeros((trial_num, (int(time / dt) + 1)))
PPN_FT_record = np.zeros((trial_num, (int(time / dt) + 1)))
OFC_record = np.zeros((trial_num, (int(time / dt) + 1)))
NAc_record = np.zeros((trial_num, (int(time / dt) + 1)))
VP_record = np.zeros((trial_num, (int(time / dt) + 1)))
temp = np.zeros((trial_num, (int(time / dt) + 1)))
Max_Dop_record = np.zeros(11, )
Max_GABA_record = np.zeros(trial_num, )
Max_US_Dop_record = np.zeros(trial_num, )
Max_CS_Dop_record = np.zeros(trial_num, )
Max_BLA_Mag_record = np.zeros(trial_num, )
Mag_wts_record = np.zeros(trial_num, )
Time_wts_record = np.zeros(trial_num, )

temp_record = np.zeros((trial_num, (int(time / dt) + 1)))

'''
Constants
'''
index = 0
i = 0
BLA['US'] = 0
NAc['US'] = 0
NAc['Bound'] = 0
learning = 0
BLA['I_Dop_Max'] = 0
BLA['U_max'] = 0
early_onset = time - 100
early_scenario = False
reward_time = time - 100
stimulus_onset = 10
'''
Compute
'''


@before(clock.tick)
def func(t):
    global index
    if t > stimulus_onset and t <= reward_time + 30:
        IT['I'][:] = 1
    else:
        IT['I'][:] = 0

    # Final trial used for reward delivery

    if i == trial_num - 1:
        # print "early reward"
        # Change this to set early reward
        if early_scenario:
            if t >= early_onset and t <= early_onset + 30:
                LH['I'][:] = 1
            else:
                LH['I'][:] = 0
        else:
            if t >= reward_time and t <= reward_time + 30:
                LH['I'][:] = 2
            else:
                LH['I'][:] = 0

    elif t >= reward_time and t <= reward_time + 30:

        # Update BLA weights at US reward delivery (active throughout reward delivery)
        BLA['US'] = 1

        NAc['US'] = 1
        # normal reward delivery
        LH['I'][:] = 1

        if t == reward_time:
            # gets initialized even without reward for now
            # Update NAc weights at reward delivery (just at this time step)
            NAc['check'] = 1

            # checking for NAc min, if NAc firing slope is greater than 0, switch on bound,if not zero at reward delivery
            if round(NAc['U'], 2) > 0:
                print (NAc['U'])
                NAc['Bound'] = 1
                NAc['Value'] = NAc['U']
                print ("Bound reached")

                # print "new value", NAc['U']
            '''
            else:
                #print "new value", NAc['U']
                NAc['Bound'] = 0
            '''
        else:
            NAc['check'] = 0
            if NAc['Bound'] == 1:
                NAc['Bound'] = 2
    else:
        LH['I'][:] = 0
        BLA['US'] = 0
        NAc['US'] = 0

    PPN_RD_record[i][index] = sum(PPN_RD['U'].ravel()) / PPN_RD_size
    LH_record[i][index] = sum(LH['I'].ravel()) / LH_size
    VTA_dop_record[i][index] = sum(VTA_dop['U'].ravel()) / VTA_dop_size
    VTA_GABA_record[i][index] = sum(VTA_GABA['U'].ravel()) / VTA_GABA_size
    BLA_record[i][index] = sum(BLA['U'].ravel()) / BLA_size
    IT_record[i][index] = sum(IT['I'].ravel()) / IT_size
    CE_record[i][index] = sum(CE['U'].ravel()) / CE_size
    PPN_FT_record[i][index] = sum(PPN_FT['U'].ravel()) / PPN_FT_size
    OFC_record[i][index] = sum(OFC['U'].ravel()) / OFC_size
    NAc_record[i][index] = sum(NAc['U'].ravel()) / NAc_size
    temp[i][index] = sum(NAc['V'].ravel()) / NAc_size

    temp_record[i][index] = sum(NAc['adapt_exc'].ravel()) / NAc_size
    # print NAc['Bound']
    index += 1


# NAc['NAc_max']=0
# NAc['Bound']=0

def main(args):
    print (args)
    global i
    global index
    global early_onset

    early_onset = int(args[0][1])

    scenarios = ["Early","Double_Magnitude_Lesion","Double_Magnitude_Control"]
    scenario = 2
    var = []
    while i < trial_num:

        run(time, dt)

        index = 0
        # print learning
        # print i,"learn",NAc['L']

        # Checking for BLA max
        temp2 = np.amax(BLA_record[i])

        if temp2 > BLA['U_max'][0] and BLA['U_max'][0] != max(LH_record[i]):
            BLA['U_max'] = temp2



        Max_GABA_record[i] = np.amax(VTA_GABA_record[i])
        Max_US_Dop_record[i] = np.amax(VTA_dop_record[i][int(reward_time):])
        Max_CS_Dop_record[i] = np.amax(VTA_dop_record[i][stimulus_onset:50])
        Max_BLA_Mag_record[i] = np.amax(BLA_record[i])
        Mag_wts_record[i] = max(Y.weights.toarray()[0])
        Time_wts_record[i] = max(X.weights.toarray()[0])

        i += 1

    Max_Dop_record = np.amax(VTA_dop_record[trial_num - 1][early_onset:]) - .2
    print (Max_Dop_record)
    with open("test.txt", "a") as earlyRewardFile:
        earlyRewardFile.write(str(early_onset) + " " + str(Max_Dop_record) + "\n")

    '''
    PLot graphs
    '''






    sns.set_style("darkgrid")
    sns.set()
    sns.set_context("paper")


    #Figures for Early Reward and VS Lesion Scenarios
    plt.figure(5)
    # sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    ax = plt.subplot(3, 2, 1)
    plt.plot(IT_record[trial_num - 1],color="grey")
    ax.set_ylabel("IT")
    ax.set_ylim(-0.3, 2)
    ax.set_xticks(np.arange(0, 600, step=200))
    #ax.get_xaxis().set_ticks([0, 200, 400])
    #ax.set(xticklabels=[0, '', 200, '', 400])
    #ax.set(yticklabels=['', 0, '', 1, '', 2])
    # ax.set_axis_bgcolor('grey')

    ax = plt.subplot(3, 2, 2)
    plt.plot(LH_record[trial_num - 1],color="grey")
    ax.set_ylim(-0.3, 2)
    ax.set_ylabel("LH")
    #print ax.get_xaxis().get_ticks([0, 200, 400])
    # ax.set_axis_bgcolor('grey')
    #ax.set(yticklabels=['', 0, '', 1, '', 2])
    ax.set(xticklabels=['',0, '', 200, '', 400])

    with open("LH_"+scenarios[scenario]+"_VS_Lesion_record.csv", "ab") as f:
        np.savetxt(f, LH_record[trial_num-1], delimiter=',', fmt='%f', newline=" ")
        f.write("\n")

    ax = plt.subplot(3, 2, 3)
    plt.plot(VTA_dop_record[trial_num - 1],color="red")
    ax.set_ylabel("VTA DA")
    ax.set_ylim(-0.3, 2)
    #ax.get_xaxis().set_ticks([0, 200, 400])
    # ax.set_axis_bgcolor('red')
    #ax.set(yticklabels=['', 0, '', 1, '', 2])
    ax.set(xticklabels=['',0, '', 200, '', 400])

    with open("VTA_DA_"+scenarios[scenario]+"_VS_Lesion_record.csv", "ab") as f:
        np.savetxt(f, VTA_dop_record[trial_num-1], delimiter=',', fmt='%f', newline=" ")
        f.write("\n")

    ax = plt.subplot(3, 2, 4)
    plt.plot(VTA_GABA_record[trial_num - 1],color="red")
    ax.set_ylabel("VTA GABA")
    ax.set_ylim(-0.3, 2)
    #ax.get_xaxis().set_ticks([0, 200, 400])
    ax.set(xticklabels=['',0, '', 200, '', 400])
    #ax.set(yticklabels=['', 0, '', 1, '', 2])
    # ax.set_axis_bgcolor('red')

    with open("VTA_GABA_" + scenarios[scenario] + "_VS_Lesion_record.csv", "ab") as f:
        np.savetxt(f, VTA_GABA_record[trial_num - 1], delimiter=',', fmt='%f', newline=" ")
        f.write("\n")



    ax = plt.subplot(3, 2, 5)
    plt.plot(PPN_RD_record[trial_num - 1],color="orange")
    ax.set_ylim(-0.3, 2)
    ax.set_ylabel("PPN RD")
    #ax.get_xaxis().set_ticks([0, 200, 400])
    #ax.set(yticklabels=['', 0, '', 1, '', 2])
    ax.set(xticklabels=['',0, '', 200, '', 400])
    # ax.set_axis_bgcolor('orange')

    with open("PPN_RD_" + scenarios[scenario] + "_VS_Lesion_record.csv", "ab") as f:
        np.savetxt(f, PPN_RD_record[trial_num - 1], delimiter=',', fmt='%f', newline=" ")
        f.write("\n")

    # for axis in ['top', 'bottom', 'left', 'right']:
    # ax.spines[axis].set_linewidth(2)
    # ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_linewidth(.9)
    # ax.set_frame_on(True)
    ax = plt.subplot(3, 2, 6)
    plt.plot(PPN_FT_record[trial_num - 1],color="orange")
    ax.set_ylabel("PPN FT")
    ax.set_ylim(-0.3, 2)
    ax.set(xticklabels=['', 0, '', 200, '', 400])
    #ax.set(yticklabels=['',0, '', 1, '', 2])
    #ax.set(xticklabels=[0,'', 200,'', 400])

    with open("PPN_FT_" + scenarios[scenario] + "_VS_Lesion_record.csv", "ab") as f:
        np.savetxt(f, PPN_FT_record[trial_num - 1], delimiter=',', fmt='%f', newline=" ")
        f.write("\n")

    #plt.savefig("Figure12.svg") #latest change


    plt.tight_layout()
    plt.show()
    # plt.close("all")

    '''
    End
    '''


if __name__ == '__main__':
    main([sys.argv])


