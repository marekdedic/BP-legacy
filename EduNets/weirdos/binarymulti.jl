
using EduNets
include("../examples/doublebag.jl");
include("../examples/multidoublebagmodels.jl");
function createdataset()
  d=10;
  ni=100;
  nb=10;
  x=map(Float64,randn(d,ni));

  ids=sample(1:nb,ni);
  subbags=map(i->find(ids.==i),unique(ids));
  ids=sample(1:nb,length(subbags));
  bags=map(i->find(ids.==i),unique(ids));
  y=sample(1:4,length(bags));

  ds=DoubleBagDataset(x,y,bags,subbags,DataFrames.DataFrame([]));
end

function optFun!(ds,x::Vector,gg,binmodel,gbin,multimodel,gmulti,lambda,alpha,nbin,nmulti,ggmulti)
  offset=update!(binmodel,x)
  update!(multimodel.third,x,offset=offset)
  
  #calculate the gradient with respect to the binary part classifier
  # classsizes=fill(nbin,maximum(ds.y));
  # classsizes[2:end]=div(nbin,length(classsizes)-1)+1;
  # dss=EduNets.sample(ds,classsizes)
  dss=deepcopy(ds);
  dss.y[dss.y.>1]=2;
  fbin=gradient!(binmodel,dss,gbin)
  model2vector!(gbin,gg)

  #calculate the gradient with respect to the multi-class part of the optimization problem
  # classsizes=fill(nmulti,maximum(ds.y))
  # classsizes[1]=0;
  # dss=EduNets.sample(ds,classsizes)
  # dss.y-=1
  dss=deepcopy(ds);
  fmulti=gradient!(multimodel,dss,gmulti)
  model2vector!(gmulti,ggmulti)
  # println("fbin = $fbin fmulti = $fmulti")

  fbin+=l1regularize!(binmodel,gbin,lambda);
  fmulti+=l1regularize!(multimodel,gmulti,lambda);

  #fold gradients from two models together
  l=length(gmulti.first)+length(gmulti.second)
  gg*=alpha;
  gg[1:l]+=(1-alpha)*ggmulti[1:l];
  gg[offset:end]=(1-alpha)*ggmulti[l+1:end];

  return(alpha*fbin+(1-alpha)*fmulti,gg)
end



modeltype="relumaxrelumax"

kb=(10,3,3,2)
km=(10,3,3,4)
w=0.01
(bintrnmodel,bintstmodel,modelname)=getmodel(modeltype,kb,T=Float64)
bintrnmodel.loss.w=[1-w,w];
(multitrnmodel,multitstmodel,modelname)=getmodel(modeltype,km,T=Float64)

#now force both models to share the first and second layers and assume that pooling function is non-trainable
bintrnmodel.first=multitrnmodel.first;
bintrnmodel.second=multitrnmodel.second;

lambda=Float64(1e-6);
gbin=deepcopy(bintrnmodel);
gmulti=deepcopy(multitrnmodel);

theta=vcat(model2vector(bintrnmodel),model2vector(multitrnmodel.third));

lambda=Float64(1e-6/length(theta));
nbin=1000;
nmulti=350;
oprefix="results/asyncbm_1/"*modelname

gg=deepcopy(theta)
ggmulti=model2vector(multitrnmodel)
ds=createdataset()

alpha=0.5f0

EduNets.testgradient((x)->optFun!(ds,x,gg,bintrnmodel,gbin,multitrnmodel,gmulti,lambda,alpha,nbin,nmulti,ggmulti),theta;verbose=2);