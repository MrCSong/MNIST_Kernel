import numpy as np
i = 0.0
list=[]
m=0.0
while m>0:
    list.append(m)
    m-=0.00001
    m=round(m,5)
print(list)
