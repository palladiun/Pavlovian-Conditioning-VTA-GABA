import numpy as np

import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
import seaborn as  sns
import sys
time=500
trial=15


IT_record = np.mean(np.loadtxt("IT_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
LH_record = np.mean(np.loadtxt("LH_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
BLA_record = np.mean(np.loadtxt("BLA_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_dop_record = np.mean(np.loadtxt("VTA_dop_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_record =  np.mean(np.loadtxt("VTA_GABA_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
NAc_record =  np.mean(np.loadtxt("NAc_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_record =  np.mean(np.loadtxt("PPN_RD_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_record =  np.mean(np.loadtxt("PPN_FT_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)

PPN_RD_0_record = np.mean(np.loadtxt("PPN_RD_0_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_6_record = np.mean(np.loadtxt("PPN_RD_6_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_16_record = np.mean(np.loadtxt("PPN_RD_16_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_0_record = np.mean(np.loadtxt("PPN_FT_0_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_6_record = np.mean(np.loadtxt("PPN_FT_6_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_16_record = np.mean(np.loadtxt("PPN_FT_16_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_dop_0_record = np.mean(np.loadtxt("VTA_dop_0_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_dop_6_record = np.mean(np.loadtxt("VTA_dop_6_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_dop_16_record = np.mean(np.loadtxt("VTA_dop_16_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_0_record = np.mean(np.loadtxt("VTA_GABA_0_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_6_record = np.mean(np.loadtxt("VTA_GABA_6_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_16_record = np.mean(np.loadtxt("VTA_GABA_16_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)

Max_US_Dop_record = np.mean(np.loadtxt("Max_US_Dop_record.csv", usecols=range(0, trial), dtype=np.float32),axis = 0)
Max_CS_Dop_record = np.mean(np.loadtxt("Max_CS_Dop_record.csv", usecols=range(0, trial), dtype=np.float32),axis = 0)
Max_GABA_record = np.mean(np.loadtxt("Max_GABA_Dop_record.csv", usecols=range(0, trial), dtype=np.float32),axis = 0)

LH_Early100_record = np.mean(np.loadtxt("LH_Early100_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_Early100_record = np.mean(np.loadtxt("PPN_RD_Early100_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_Early100_record = np.mean(np.loadtxt("PPN_FT_Early100_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_DA_Early100_record = np.mean(np.loadtxt("VTA_DA_Early100_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_Early100_record = np.mean(np.loadtxt("VTA_GABA_Early100_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)

LH_Early300_record = np.mean(np.loadtxt("LH_Early300_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_Early300_record = np.mean(np.loadtxt("PPN_RD_Early300_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_Early300_record = np.mean(np.loadtxt("PPN_FT_Early300_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_DA_Early300_record = np.mean(np.loadtxt("VTA_DA_Early300_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_Early300_record = np.mean(np.loadtxt("VTA_GABA_Early300_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)

LH_Early_VS_Lesion_record = np.mean(np.loadtxt("LH_Early_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_EarlyVS_Lesion_record = np.mean(np.loadtxt("PPN_RD_EarlyVS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_EarlyVS_Lesion_record = np.mean(np.loadtxt("PPN_FT_EarlyVS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_DA_EarlyVS_Lesion_record = np.mean(np.loadtxt("VTA_DA_EarlyVS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_EarlyVS_Lesion_record = np.mean(np.loadtxt("VTA_GABA_EarlyVS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)

LH_Double_Magnitude_Lesion_VS_Lesion_record = np.mean(np.loadtxt("LH_Double_Magnitude_Lesion_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_Double_Magnitude_Lesion_VS_Lesion_record = np.mean(np.loadtxt("PPN_RD_Double_Magnitude_Lesion_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_Double_Magnitude_Lesion_VS_Lesion_record = np.mean(np.loadtxt("PPN_FT_Double_Magnitude_Lesion_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_DA_Double_Magnitude_Lesion_VS_Lesion_record = np.mean(np.loadtxt("VTA_DA_Double_Magnitude_Lesion_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_Double_Magnitude_Lesion_VS_Lesion_record = np.mean(np.loadtxt("VTA_GABA_Double_Magnitude_Lesion_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)

LH_Double_Magnitude_Control_VS_Lesion_record = np.mean(np.loadtxt("LH_Double_Magnitude_Control_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_RD_Double_Magnitude_Control_VS_Lesion_record = np.mean(np.loadtxt("PPN_RD_Double_Magnitude_Control_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
PPN_FT_Double_Magnitude_Control_VS_Lesion_record = np.mean(np.loadtxt("PPN_FT_Double_Magnitude_Control_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_DA_Double_Magnitude_Control_VS_Lesion_record = np.mean(np.loadtxt("VTA_DA_Double_Magnitude_Control_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)
VTA_GABA_Double_Magnitude_Control_VS_Lesion_record = np.mean(np.loadtxt("VTA_GABA_Double_Magnitude_Control_VS_Lesion_record.csv", usecols=range(0, time), dtype=np.float32),axis = 0)

sns.set_style("darkgrid")
sns.set()
sns.set_context("paper")

plt.figure(1,figsize=(12,5))
#sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ax = plt.subplot(3, 4, 2)
plt.plot(IT_record,color="grey")
ax.set_ylabel("IT")
ax.set_ylim(-0.3, 2)
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])

ax = plt.subplot(3, 4, 3)
plt.plot(LH_record, color="grey")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("LH")
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])



ax = plt.subplot(3, 4, 5)
plt.plot(BLA_record,color="blue")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("BLA")
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])



ax = plt.subplot(3, 4, 6)
plt.plot(VTA_dop_record,color="red")
ax.set_ylabel("VTA DA")
ax.set_ylim(-0.3, 2)
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])



ax = plt.subplot(3, 4, 7)
plt.plot(VTA_GABA_record,color="red")
ax.set_ylabel("VTA GABA")
ax.set_ylim(-0.3, 2)
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])


ax = plt.subplot(3, 4, 8)
plt.plot(NAc_record,color="green")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("VS")
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])



ax = plt.subplot(3, 4, 10)
plt.plot(PPN_RD_record,color="orange")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("PPN RD")
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])





ax = plt.subplot(3, 4, 11)
plt.plot(PPN_FT_record,color="orange")
ax.set_ylabel("PPN FT")
ax.set_ylim(-0.3, 2)
ax.set_xlim(0, time)
ax.set(xticklabels=[0, '', 200, '', 400])



plt.tight_layout()
plt.savefig("Figure1.svg")


#NExt Figure

# PPN Evloution Across Trials
# plt.figure(10)
x_size = 8
y_size = 3

fig1 = plt.figure(2, figsize=(x_size, y_size))
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=2.5)
# fig1.text(0,1,"B",fontweight='bold',horizontalalignment='left', verticalalignment='top')
ax0 = plt.subplot2grid((x_size, y_size), (0, 0), rowspan=4)
ax0.plot(PPN_RD_0_record, color="orange")
ax0.set_xlim(0, time)
ax0.set_ylim(-0.3, 2)
ax0.set_ylabel("PPN RD")
# ax0.get_xaxis().set_ticks([0,1,2])
# ax0.get_yaxis().set_ticks([0, 0.5, 1])
ax0.title.set_text('Trial 1')
# ax0.text(-1, 2, 'A', fontsize=15, weight = 'bold',color = "blue")



ax1 = plt.subplot2grid((x_size, y_size), (0, 1), rowspan=4)
ax1.plot(PPN_RD_6_record, color="orange")
ax1.set_xlim(0, time)
ax1.set_ylim(-0.3, 2)
# ax1.get_xaxis().set_ticks([])
# ax1.get_yaxis().set_ticks([0, 0.5, 1])
ax1.title.set_text('Trial 7')



ax2 = plt.subplot2grid((x_size, y_size), (0, 2), rowspan=4)
ax2.plot(PPN_RD_16_record, color="orange")
ax2.set_xlim(0, time)
ax2.set_ylim(-0.3, 2)
# ax2.get_xaxis().set_ticks([])
# ax2.get_yaxis().set_ticks([0, 0.5, 1])
ax2.title.set_text('Trial 16')



ax3 = plt.subplot2grid((x_size, y_size), (4, 0), rowspan=4)
ax3.plot(PPN_FT_0_record, color="orange")
ax3.set_xlim(0, time)
ax3.set_ylim(-0.3, 2)
ax3.set_ylabel("PPN FT")
# ax3.get_xaxis().set_ticks([])
# ax3.get_yaxis().set_ticks([0, 0.5, 1])
# ax.title.set_text('Trial 1')



ax4 = plt.subplot2grid((x_size, y_size), (4, 1), rowspan=4)
ax4.plot(PPN_FT_6_record, color="orange")
ax4.set_xlim(0, time)
ax4.set_ylim(-0.3, 2)
# ax4.get_xaxis().set_ticks([])
# ax4.get_yaxis().set_ticks([0, 0.5, 1])
# ax.title.set_text('Trial 7')



ax5 = plt.subplot2grid((x_size, y_size), (4, 2), rowspan=4)
ax5.plot(PPN_FT_16_record, color="orange")
ax5.set_xlim(0, time)
ax5.set_ylim(-0.3, 2)
# ax5.get_xaxis().set_ticks([])
# ax5.get_yaxis().set_ticks([0, 0.5, 1])
plt.savefig("Figure2.svg")
# ax.title.set_text('Trial 16')



plt.figure(3, figsize=(x_size, y_size))
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=2.5)
# fig1.text(0, .75, "C", fontweight='bold', horizontalalignment='left', verticalalignment='top')
ax6 = plt.subplot2grid((x_size, y_size), (0, 0), rowspan=4)
ax6.plot(VTA_dop_0_record, color="red")
ax6.set_ylim(-0.3, 2)
ax6.set_xlim(0, time)
ax6.set_ylabel("VTA DA")
# ax6.get_xaxis().set_ticks([])
# ax6.get_yaxis().set_ticks([0, 0.5, 1])
ax6.title.set_text('Trial 1')
# ax6.text(-1, 2, 'B', fontsize=15, weight='bold', color="blue")



ax7 = plt.subplot2grid((x_size, y_size), (0, 1), rowspan=4)
ax7.plot(VTA_dop_6_record, color="red")
ax7.set_xlim(0, time)
ax7.set_ylim(-0.3, 2)
# ax7.get_xaxis().set_ticks([])
# ax7.get_yaxis().set_ticks([0, 0.5, 1])
ax7.title.set_text('Trial 7')



ax8 = plt.subplot2grid((x_size, y_size), (0, 2), rowspan=4)
ax8.plot(VTA_dop_16_record, color="red")
ax8.set_xlim(0, time)
ax8.set_ylim(-0.3, 2)
# ax8.get_xaxis().set_ticks([])
# ax8.get_yaxis().set_ticks([0, 0.5, 1])
ax8.title.set_text('Trial 16')



ax9 = plt.subplot2grid((x_size, y_size), (4, 0), rowspan=4)
ax9.plot(VTA_GABA_0_record, color="red")
ax9.set_xlim(0, time)
ax9.set_ylim(-0.3, 2)
ax9.set_ylabel("VTA GABA")
# ax9.get_xaxis().set_ticks([])
# ax9.get_yaxis().set_ticks([0, 0.5, 1])
# ax.title.set_text('Trial 1')



ax10 = plt.subplot2grid((x_size, y_size), (4, 1), rowspan=4)
ax10.plot(VTA_GABA_6_record, color="red")
ax10.set_xlim(0, time)
ax10.set_ylim(-0.3, 2)
# ax10.get_xaxis().set_ticks([])
# ax10.get_yaxis().set_ticks([0, 0.5, 1])
# ax.title.set_text('Trial 7')



ax11 = plt.subplot2grid((x_size, y_size), (4, 2), rowspan=4)
ax11.plot(VTA_GABA_16_record, color="red")
ax11.set_xlim(0, time)
ax11.set_ylim(-0.3, 2)
# ax11.get_xaxis().set_ticks([])
# ax11.get_yaxis().set_ticks([0, 0.5, 1])
plt.savefig("Figure3.svg")



# fig1.text(0, .5, "D", fontweight='bold', horizontalalignment='left', verticalalignment='top')
plt.figure(4, figsize=(8, 4))
plt.plot(Max_US_Dop_record, label="Max US Dopamine")
plt.plot(Max_GABA_record, label="Max VTA GABA")
plt.plot(Max_CS_Dop_record, label="Max CS Dopamine")
plt.legend(loc="upper right")
plt.ylabel("Firing rate")
plt.xlabel("Trial Number")
plt.ylim(-0.3, 1.5)
plt.yticks(np.arange(0, 1.5, step=0.5))
plt.savefig("Figure4.svg")



'''
ax12 = plt.subplot2grid((6, 2),(0,0), colspan=6, rowspan=2)


ax12.plot(Max_US_Dop_record[0:trial_num - 1], label="Max US Dopamine")
ax12.plot(Max_GABA_record[0:trial_num - 1], label="Max VTA GABA")
ax12.plot(Max_CS_Dop_record[0:trial_num - 1], label="Max CS Dopamine")
ax12.legend(loc="upper left")
ax12.set_ylabel("Firing rate")
ax12.set_xlabel("Trial Number")
ax12.set_ylim(-0.3, 1.5)
#ax12.text(0, 1.8, 'C', fontsize=15, weight='bold', color="blue")
'''
'''
ax13 = plt.subplot2grid((8, 3), (6, 0), colspan=2, rowspan=2)
ax13.set_ylim(0, 1)


ax13.plot(Max_BLA_Mag_record[0:trial_num - 1], label="BLA Firing")
ax13.legend(loc="upper left")
ax13.set_ylabel("Firing Rate")
ax13.set_xlabel("Trial Number")
ax13.text(0, 1.1, 'D', fontsize=15, weight='bold', color="blue")
# ax.set_axis_bgcolor('orange')
'''
plt.tight_layout()
plt.show()
# plt.close("all")


#Figures for Early Reward and VS Lesion Scenarios

#Figure Early Reward 100
plt.figure(5)
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ax = plt.subplot(3, 2, 1)
plt.plot(IT_record,color="grey")
ax.set_ylabel("IT")
ax.set_ylim(-0.3, 2)
ax.set_xticks(np.arange(0, 600, step=200))
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(xticklabels=[0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('grey')

ax = plt.subplot(3, 2, 2)
plt.plot(LH_Early100_record,color="grey")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("LH")
#print ax.get_xaxis().get_ticks([0, 200, 400])
# ax.set_axis_bgcolor('grey')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])



ax = plt.subplot(3, 2, 3)
plt.plot(VTA_DA_Early100_record,color="red")
ax.set_ylabel("VTA DA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
# ax.set_axis_bgcolor('red')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])


ax = plt.subplot(3, 2, 4)
plt.plot(VTA_GABA_Early100_record,color="red")
ax.set_ylabel("VTA GABA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
ax.set(xticklabels=['',0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('red')


ax = plt.subplot(3, 2, 5)
plt.plot(PPN_RD_Early100_record,color="orange")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("PPN RD")
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])
# ax.set_axis_bgcolor('orange')


# for axis in ['top', 'bottom', 'left', 'right']:
# ax.spines[axis].set_linewidth(2)
# ax.spines['left'].set_visible(True)
# ax.spines['left'].set_linewidth(.9)
# ax.set_frame_on(True)
ax = plt.subplot(3, 2, 6)
plt.plot(PPN_FT_Early100_record,color="orange")
ax.set_ylabel("PPN FT")
ax.set_ylim(-0.3, 2)
ax.set(xticklabels=['', 0, '', 200, '', 400])
#ax.set(yticklabels=['',0, '', 1, '', 2])
#ax.set(xticklabels=[0,'', 200,'', 400])
plt.tight_layout()
plt.savefig("Figure5.svg")





#Figure Early Reward 300
plt.figure(6)
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ax = plt.subplot(3, 2, 1)
plt.plot(IT_record,color="grey")
ax.set_ylabel("IT")
ax.set_ylim(-0.3, 2)
ax.set_xticks(np.arange(0, 600, step=200))
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(xticklabels=[0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('grey')

ax = plt.subplot(3, 2, 2)
plt.plot(LH_Early300_record,color="grey")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("LH")
#print ax.get_xaxis().get_ticks([0, 200, 400])
# ax.set_axis_bgcolor('grey')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])



ax = plt.subplot(3, 2, 3)
plt.plot(VTA_DA_Early300_record,color="red")
ax.set_ylabel("VTA DA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
# ax.set_axis_bgcolor('red')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])


ax = plt.subplot(3, 2, 4)
plt.plot(VTA_GABA_Early300_record,color="red")
ax.set_ylabel("VTA GABA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
ax.set(xticklabels=['',0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('red')


ax = plt.subplot(3, 2, 5)
plt.plot(PPN_RD_Early300_record,color="orange")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("PPN RD")
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])
# ax.set_axis_bgcolor('orange')


# for axis in ['top', 'bottom', 'left', 'right']:
# ax.spines[axis].set_linewidth(2)
# ax.spines['left'].set_visible(True)
# ax.spines['left'].set_linewidth(.9)
# ax.set_frame_on(True)
ax = plt.subplot(3, 2, 6)
plt.plot(PPN_FT_Early300_record,color="orange")
ax.set_ylabel("PPN FT")
ax.set_ylim(-0.3, 2)
ax.set(xticklabels=['', 0, '', 200, '', 400])
#ax.set(yticklabels=['',0, '', 1, '', 2])
#ax.set(xticklabels=[0,'', 200,'', 400])
plt.tight_layout()
plt.savefig("Figure6.svg")



plt.show()


#Figure Early Reward 100 and VS Lesioned
plt.figure(7)
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ax = plt.subplot(3, 2, 1)
plt.plot(IT_record,color="grey")
ax.set_ylabel("IT")
ax.set_ylim(-0.3, 2)
ax.set_xticks(np.arange(0, 600, step=200))
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(xticklabels=[0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('grey')

ax = plt.subplot(3, 2, 2)
plt.plot(LH_Early_VS_Lesion_record,color="grey")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("LH")
#print ax.get_xaxis().get_ticks([0, 200, 400])
# ax.set_axis_bgcolor('grey')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])



ax = plt.subplot(3, 2, 3)
plt.plot(VTA_DA_EarlyVS_Lesion_record,color="red")
ax.set_ylabel("VTA DA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
# ax.set_axis_bgcolor('red')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])


ax = plt.subplot(3, 2, 4)
plt.plot(VTA_GABA_EarlyVS_Lesion_record,color="red")
ax.set_ylabel("VTA GABA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
ax.set(xticklabels=['',0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('red')


ax = plt.subplot(3, 2, 5)
plt.plot(PPN_RD_EarlyVS_Lesion_record,color="orange")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("PPN RD")
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])
# ax.set_axis_bgcolor('orange')


# for axis in ['top', 'bottom', 'left', 'right']:
# ax.spines[axis].set_linewidth(2)
# ax.spines['left'].set_visible(True)
# ax.spines['left'].set_linewidth(.9)
# ax.set_frame_on(True)
ax = plt.subplot(3, 2, 6)
plt.plot(PPN_FT_EarlyVS_Lesion_record,color="orange")
ax.set_ylabel("PPN FT")
ax.set_ylim(-0.3, 2)
ax.set(xticklabels=['', 0, '', 200, '', 400])
#ax.set(yticklabels=['',0, '', 1, '', 2])
#ax.set(xticklabels=[0,'', 200,'', 400])
plt.tight_layout()
plt.savefig("Figure7.svg")



plt.show()


#Figure Double Reward and VS Lesioned
plt.figure(8)
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ax = plt.subplot(3, 2, 1)
plt.plot(IT_record,color="grey")
ax.set_ylabel("IT")
ax.set_ylim(-0.3, 2)
ax.set_xticks(np.arange(0, 600, step=200))
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(xticklabels=[0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('grey')

ax = plt.subplot(3, 2, 2)
plt.plot(LH_Double_Magnitude_Lesion_VS_Lesion_record,color="grey")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("LH")
#print ax.get_xaxis().get_ticks([0, 200, 400])
# ax.set_axis_bgcolor('grey')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])



ax = plt.subplot(3, 2, 3)
plt.plot(VTA_DA_Double_Magnitude_Lesion_VS_Lesion_record,color="red")
ax.set_ylabel("VTA DA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
# ax.set_axis_bgcolor('red')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])


ax = plt.subplot(3, 2, 4)
plt.plot(VTA_GABA_Double_Magnitude_Lesion_VS_Lesion_record,color="red")
ax.set_ylabel("VTA GABA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
ax.set(xticklabels=['',0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('red')


ax = plt.subplot(3, 2, 5)
plt.plot(PPN_RD_Double_Magnitude_Lesion_VS_Lesion_record,color="orange")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("PPN RD")
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])
# ax.set_axis_bgcolor('orange')


# for axis in ['top', 'bottom', 'left', 'right']:
# ax.spines[axis].set_linewidth(2)
# ax.spines['left'].set_visible(True)
# ax.spines['left'].set_linewidth(.9)
# ax.set_frame_on(True)
ax = plt.subplot(3, 2, 6)
plt.plot(PPN_FT_Double_Magnitude_Lesion_VS_Lesion_record,color="orange")
ax.set_ylabel("PPN FT")
ax.set_ylim(-0.3, 2)
ax.set(xticklabels=['', 0, '', 200, '', 400])
#ax.set(yticklabels=['',0, '', 1, '', 2])
#ax.set(xticklabels=[0,'', 200,'', 400])
plt.tight_layout()
plt.savefig("Figure8.svg")


plt.show()


#Figure Double Reward and Control
plt.figure(9)
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ax = plt.subplot(3, 2, 1)
plt.plot(IT_record,color="grey")
ax.set_ylabel("IT")
ax.set_ylim(-0.3, 2)
ax.set_xticks(np.arange(0, 600, step=200))
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(xticklabels=[0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('grey')

ax = plt.subplot(3, 2, 2)
plt.plot(LH_Double_Magnitude_Control_VS_Lesion_record,color="grey")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("LH")
#print ax.get_xaxis().get_ticks([0, 200, 400])
# ax.set_axis_bgcolor('grey')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])



ax = plt.subplot(3, 2, 3)
plt.plot(VTA_DA_Double_Magnitude_Control_VS_Lesion_record,color="red")
ax.set_ylabel("VTA DA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
# ax.set_axis_bgcolor('red')
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])


ax = plt.subplot(3, 2, 4)
plt.plot(VTA_GABA_Double_Magnitude_Control_VS_Lesion_record,color="red")
ax.set_ylabel("VTA GABA")
ax.set_ylim(-0.3, 2)
#ax.get_xaxis().set_ticks([0, 200, 400])
ax.set(xticklabels=['',0, '', 200, '', 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
# ax.set_axis_bgcolor('red')


ax = plt.subplot(3, 2, 5)
plt.plot(PPN_RD_Double_Magnitude_Control_VS_Lesion_record,color="orange")
ax.set_ylim(-0.3, 2)
ax.set_ylabel("PPN RD")
#ax.get_xaxis().set_ticks([0, 200, 400])
#ax.set(yticklabels=['', 0, '', 1, '', 2])
ax.set(xticklabels=['',0, '', 200, '', 400])
# ax.set_axis_bgcolor('orange')


# for axis in ['top', 'bottom', 'left', 'right']:
# ax.spines[axis].set_linewidth(2)
# ax.spines['left'].set_visible(True)
# ax.spines['left'].set_linewidth(.9)
# ax.set_frame_on(True)
ax = plt.subplot(3, 2, 6)
plt.plot(PPN_FT_Double_Magnitude_Control_VS_Lesion_record,color="orange")
ax.set_ylabel("PPN FT")
ax.set_ylim(-0.3, 2)
ax.set(xticklabels=['', 0, '', 200, '', 400])
#ax.set(yticklabels=['',0, '', 1, '', 2])
#ax.set(xticklabels=[0,'', 200,'', 400])
plt.tight_layout()
plt.savefig("Figure9.svg")





plt.show()



'''
End
'''


