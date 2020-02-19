import svgutils.transform as sg
from svgutils.compose import *
import sys

#create new SVG figure
#fig = sg.SVGFigure("1080", "360")

# load matpotlib-generated figures

fig1 = sg.fromfile('Figure1.svg')
fig2 = sg.fromfile('Figure2.svg')
fig3 = sg.fromfile('Figure3.svg')
fig4 = sg.fromfile('Figure4.svg')

print fig1.get_size()
print fig2.get_size()
print fig3.get_size()
print fig4.get_size()
# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot3 = fig3.getroot()
plot4 = fig4.getroot()


'''
Figure("1224", "432",
        Panel(
              SVG("Figure1.svg").scale(.75),
              Text("A", 25, 20, size=12, weight='bold')
             ),
        Panel(
              SVG("Figure3.svg"),
              Text("B", 25, 20, size=12, weight='bold')
             ).move(648, 0),
        Panel(
              SVG("Figure2.svg"),
              Text("C", 25, 20, size=12, weight='bold')
             ).move(648, 216),
        Panel(
              SVG("Figure4.svg").scale(.56),
              Text("D", 25, 20, size=12, weight='bold')
             ).move(161, 270)
        ).save("Panel1.svg")

'''


fig1 = sg.fromfile('Figure5.svg')
fig2 = sg.fromfile('Figure6.svg')
fig3 = sg.fromfile('Figure10.svg')

print fig1.get_size()
print fig2.get_size()
print fig3.get_size()

fig1 = sg.fromfile('Figure7.svg')
fig2 = sg.fromfile('Figure8.svg')
fig3 = sg.fromfile('Figure9.svg')



Figure("1382.4", "345.6",
        Panel(
              SVG("Figure7.svg"),
              Text("A", 5, 20, size=12, weight='bold')
             ),
        Panel(
              SVG("Figure8.svg"),
              Text("B", 5, 20, size=12, weight='bold')
             ).move(460.8,0),
        Panel(
              SVG("Figure9.svg").scale(1),
              Text("C", 5, 20, size=12, weight='bold')
             ).move(921.6,0)
        ).save("Panel3.svg")




fig1 = sg.fromfile('Figure7.svg')
fig2 = sg.fromfile('Figure8.svg')
fig3 = sg.fromfile('Figure9.svg')

print fig1.get_size()
print fig2.get_size()
print fig3.get_size()
'''

Figure("921.6", "691.2",
        Panel(
              SVG("Figure7.svg"),
              Text("A", 5, 20, size=12, weight='bold')
             ).move(230.4,0),
        Panel(
              SVG("Figure8.svg"),
              Text("B", 5, 20, size=12, weight='bold')
             ).move(0, 345.6),
        Panel(
              SVG("Figure9.svg"),
              Text("C", 5, 20, size=12, weight='bold')
             ).move(460.8,345.6)
        ).save("Panel3.svg")

'''





