

"""
function aadam{T}(fOpt!,theta::Vector{T};maxiter::Int=Int(1e5),oprefix::String="")

Dirty version of Adam with asynchronous loading of the data. The function assumes that loaddata() that 
loads the dataset and return it  is defined in the process number 2. 
 
"""
function aadam{T}(fOpt!,theta::Vector{T};maxiter::Int=Int(1e5),oprefix::String="")
  #if the old state is saved, reload it
  m=zeros(T,length(theta));
  v=zeros(T,length(theta));
  is=1:maxiter;
  errors=zeros(T,div(length(is),10),3);
  if !isempty(oprefix) 
    ofile=@sprintf("%s.jld",oprefix);
    if isfile(ofile)
      iter=load(ofile,"iter");
      is=iter+1:maxiter;
      theta=load(ofile,"theta");
      m=load(ofile,"m");
      v=load(ofile,"v");
      errors=load(ofile,"errors");
    end
  end
  
  beta1=T(0.99);
  beta2=T(0.999);
  epsilon=T(1e-8);
  alpha=T(0.001);
  gf=deepcopy(theta);


  tic();
  ds=remotecall_fetch(()->loaddata(),2)
  loadtime=toq();
  ch = Channel(1)
  @async for i in 1:maxiter/20
    dss=remotecall_fetch(()->loaddata(),2);
    put!(ch,dss)
  end
  
  sumF=zero(T)
  tic();
  batchtime=0.0;
  batchiter=0;
  eidx=1;
  for i in is
    #evaluate the function
    f=fOpt!(ds,theta,gf);


    #make a step in Adam
    m=beta1*m + (1-beta1)*gf;
    v=beta2*v + (1-beta2)*gf.^2;
    mHat=m/(1-beta1^i);
    vHat=v/(1-beta2^i);
    theta=theta-alpha*mHat./(sqrt(v)+epsilon);

    #update state variables
    sumF+=f;
    batchtime+=toq()
    batchiter+=1
    tic();
    if batchtime>loadtime
      lastF=sumF/batchiter
      errors[eidx,:]=[i,batchtime,lastF];
      sumF=zero(T);
      #check if the time of loading the dataset is longer and fetch the dataset
      ds=take!(ch)
      transfertime=toq()
      println(@sprintf("%d error = %g computetime = %.2f data transfer time = %.2f",i,lastF,batchtime,transfertime));
      tic()
      batchtime=0.0;
    end

    if mod(i,100)==0 && !isempty(oprefix)
      save(@sprintf("%s.jld",oprefix),"iter",i,"theta",theta,"m",m,"v",v,"errors",errors;compress=true);
    end
  end
end