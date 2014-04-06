using DataFrames
using GLM
using Winston

# Enter data
A=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
Y=[200,150,220,110,50,180,90,170,170,30,70,110,80,50,10,20]

# Plot Y versus A
p = FramedPlot()
a = Points(A,Y)
add(p,a)

# Describe Y for different levels of A
describe(Y[A .== 0])
describe(Y[A .== 1])

A2 = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
Y2 = [110,80,50,40,170,30,70,50,110,50,180,130,200,150,220,210]

# Plot Y2 versus A2
p2 = FramedPlot()
a2 = Points(A2,Y2)
add(p2,a2)

# Describe Y2 for levels of A2
describe(Y2[A2.==1])
describe(Y2[A2.==2])
describe(Y2[A2.==3])
describe(Y2[A2.==4])

#PROGRAM 11.2
#2-parameter linear model
#Data from Figures 11.3 and 11.1

A3 = [3,11,17,23,29,37,41,53,67,79,83,97,60,71,15,45]
Y3 = [21,54,33,101,85,65,157,120,111,200,140,220,230,217,11,190]

# Need to put data in dataframes for lm
D = DataFrame(A=A,Y=Y)
D3 = DataFrame(A3=A3,Y3=Y3)

lm(Y~A,D)
lm(Y3~A3,D3)

#PROGRAM 11.3
#3-parameter linear model
#Data from Figure 11.3

D3[:Asq] = A3.*A3
lm(Y3~A3+Asq,D3)
