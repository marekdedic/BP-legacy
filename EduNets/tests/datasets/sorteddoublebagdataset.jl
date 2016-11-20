using StatsBase
include("../../src/datatypes/sorteddoublebagdataset.jl")

l=100
subbagids=sample(1:10,100)
bagids=sample(1:5,100)
labels=sample(1:2,100)
X=randn(5,l)
#sort the datasets
I1=sortperm(subbagids);
I2=sortperm(bagids[I1]);
I=I1[I2]
bagids=bagids[I];
subbagids=subbagids[I];
X=X[:,I];
labels=labels[I];

ds=SortedDoubleBagDataset(bagids,subbagids,X,labels)
println(ds.subbags)
println(ds.bags)
println(ds.bigbags)