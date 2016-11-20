using EduNets
include("../../../lib/src/layers/clusterloss.jl")


function testclusterloss1()
  X=randn(2,15)/10
  Y=ones(Int,15)
  X=randn(2,15)/10
  Y=ones(Int,15)
  X[1,1:5]+=2
  Y[1:5]=1
  Y[6:end-1]=2

  X[2,end]-=2
  Y[end]=3
  k=10

  loss=ClusterLoss(k);
  f=forward(loss,X,Y)
  println("error in the aligned case $f")
  Y=sample(1:3,15)
  f=forward(loss,X,Y)
  println("error in the non-aligned case $f")
end

function testclusterloss2()
  X=randn(5,10)
  Y=sample(1:3,10)
  k=3

  loss=ClusterLoss(k);
  function optFun(x::Matrix)
    (f,g)=gradient!(loss,x,Y)
    return(f,g)
  end
  EduNets.testgradient(optFun,X;verbose=1)
end

testclusterloss1();
testclusterloss2();