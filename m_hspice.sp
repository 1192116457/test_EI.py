test M in subcricuit definitions
.option post acct=2
.subckt   vdiv net1 net2 rload=100 M=1  
  r1   net1 net2  'rload*M'
  r2   net2 0  'rload'
.ends

X1   n1   n2   vdiv rload=1000  M=2
V1 n1 0 5

.op
.end

