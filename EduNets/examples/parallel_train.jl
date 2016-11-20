
#copied from http://stackoverflow.com/questions/27677399/julia-how-to-copy-data-to-another-processor-in-julia
function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, eval(Main, Expr(:(=), nm, val)))
    end
end


function sendto(ps::Vector{Int}; args...)
    for p in ps
        sendto(p; args...)
    end
end

function paralleltrainshared(filenames,model,preprocess,oprefix,lambda;N=100::Int)
  #send all the stuff to the remote server
  gvec=model2vector(model);
  gvecs=SharedArray(eltype(gvec),nworkers()*length(gvec));
  xs=SharedArray(eltype(gvec),length(gvec));

  sendto(workers(),filenames=filenames,model=model,preprocess=preprocess,lambda=lambda,N=N);
  @everywhere begin
    negfiles=filter(x->contains(x,"neg.jld"),filenames)
    posfiles=filter(x->contains(x,"pos.jld"),filenames)

    #Create deep copy of the model to store the gradient
    g=deepcopy(model)

    function loaddata()
      ds=ReadData.loadscattered(ReadData.samplefiles((negfiles,posfiles),[1,1]);T=Float32);
      ds.y[ds.y.>1]=2;
      dss=EduNets.forward!(preprocess,ds)
      return(dss)
    end

    function optFun(x::SharedArray,gvecs::SharedArray)
      # println("optfun invoked")
      ReadData.update!(model,sdata(x))
      # println("model updated")
      f=ReadData.gradient!(model,EduNets.sample(ds,[N,N];subbagsize=100),g)
      # println("gradient calculated")
      ReadData.model2vector!(g,sdata(gvecs),offset=(myid()-2)*length(x)+1)
      # println("gradient stored")
      return(f)
    end
  end

  function poptFun(x,xs)
    # println("solution copied to the shared array")
    copy!(xs,x)
    
    # println("mapreduce started")
    f=reduce(+,map(fetch,[@spawnat id optFun(xs,gvecs) for id in workers()]))
    # println("mapreduce finished")
    fill!(gvec,0);
    l=length(gvec)
    gx=sdata(gvecs)
    for j in 1:nworkers()
       for i in 1:l
         gvec[i]+=gx[(j-1)*l+i]
      end
    end
    f/=nworkers()
    scale!(gvec,1.0/nworkers())
    return(f,gvec)
  end

  # @everywhere ds=loaddata()
  # EduNets.testgradient(x->poptFun(x,x),model2vector(model),verbose=1)
  # poptFun(model2vector(model),x)
  EduNets.paralleladam(loaddata,x->poptFun(x,xs),model2vector(model);options=EduNets.AdamOptions(;maxIter=1000,progressFile=oprefix),numberofdataloads=300)
end

function parallel1bittrain(filenames,model,preprocess,oprefix,lambda;N=100::Int)
  #send all the stuff to the remote server
  sendto(workers(),filenames=filenames,model=model,preprocess=preprocess,lambda=lambda,N=N);
  @everywhere begin
    negfiles=filter(x->contains(x,"neg.jld"),filenames)
    posfiles=filter(x->contains(x,"pos.jld"),filenames)

    #prepare structures for calculating and storing the gradient 
    g=deepcopy(model)
    gg=ReadData.model2vector(model);
    ReadData.update!(g,zeros(eltype(gg),length(gg)))
    lambda/=length(gg)
    gg=vcat(gg,zero(eltype(gg)));
    gg1=gg.>0;

    function loaddata()
      ds=ReadData.loadscattered(ReadData.samplefiles((negfiles,posfiles),[1,1]);T=Float32);
      ds.y[ds.y.>1]=2;
      dss=EduNets.forward!(preprocess,ds)
      return(dss)
    end

    function optFun(x::Vector)
      ReadData.update!(model,x)
      f=ReadData.gradient!(model,EduNets.sample(ds,[N,N];subbagsize=100),g,update=true)
      f+=ReadData.l1regularize!(model,g,lambda=lambda);
      ReadData.model2vector!(g,gg)
      gg[end]=f;
      map!(v->v.>0,gg1,gg);
      for (i,v) in enumerate(gg)
        if gg1[i]
           gg[i]-=1;
        else
           gg[i]+=1;
        end
      end
      ReadData.update!(g,gg)
      return(gg)
    end
  end

  function poptFun(x::Vector)
    fg=reduce(+,map(xx->(fetch(xx)-eltype(x)(0.5)),[@spawnat id optFun(x) for id in workers()]))
    fg/=eltype(fg)(nworkers()/2)
    return(fg[end],view(fg,1:length(fg)-1))
  end

  theta=ReadData.model2vector(model);
  EduNets.paralleladam(loaddata,poptFun,theta;options=EduNets.AdamOptions(;maxIter=1000,progressFile=oprefix),numberofdataloads=300)
end


function paralleltrain(filenames,model,preprocess,oprefix,lambda;N=100::Int)
  #send all the stuff to the remote server
  sendto(workers(),filenames=filenames,model=model,preprocess=preprocess,lambda=lambda,N=N);
  @everywhere begin
    negfiles=filter(x->contains(x,"neg.jld"),filenames)
    posfiles=filter(x->contains(x,"pos.jld"),filenames)

    #prepare structures for calculating and storing the gradient 
    g=deepcopy(model)
    gg=ReadData.model2vector(model);
    lambda/=length(gg)
    gg=vcat(gg,zero(eltype(gg)));

    function loaddata()
      ds=ReadData.loadscattered(ReadData.samplefiles((negfiles,posfiles),[3,3]);T=Float32);
      ds.y[ds.y.>1]=2;
      dss=EduNets.forward!(preprocess,ds)
      return(dss)
    end

    function optFun(x::Vector)
      ReadData.update!(model,x)
      f=ReadData.gradient!(model,EduNets.sample(ds,[N,N];subbagsize=100),g)
      f+=ReadData.l1regularize!(model,g,lambda=lambda);
      ReadData.model2vector!(g,gg)
      gg[end]=f;
      return(gg)
    end
  end

  function poptFun(x::Vector)
    fg=reduce(+,map(fetch,[@spawnat id optFun(x) for id in workers()]))
    fg/=nworkers()
    return(fg[end],view(fg,1:length(fg)-1))
  end

  theta=ReadData.model2vector(model);
  EduNets.paralleladam(loaddata,poptFun,theta;options=EduNets.AdamOptions(;maxIter=1000,progressFile=oprefix),numberofdataloads=300)
end


function prcurve(model::AbstractModel,preprocessing,filenames,ofname;N=10000,yhatfile::String="",T::DataType=Float32,maxsamples::Int=Int(2e7))
  allfiles=filter(x->endswith(x,".pos.jld"),filenames);
  append!(allfiles,filter(x->endswith(x,".neg.jld"),filenames));

  yHat=Array{T,1}(maxsamples)
  y=Array{Int,1}(maxsamples)
  idx=1;
  for f in allfiles
    # try
      dss=loadscattered([f],withinfo=true,T=T);
      dss.y[dss.y.>1]=2;
      ds=forward!(preprocessing,dss)
      l=length(ds.y)
      if idx+l >maxsamples
        break;
      end

      y[idx:idx+l-1]=2*(ds.y-1)-1;  #true label
      yHat[idx:idx+l-1]=squeeze(forward!(model,ds),1);;  #estimated label
      idx+=l;
      println(idx)
    # catch
    #   continue
    # end
  end
  yHat=yHat[1:idx-1]
  y=y[1:idx-1]

  thresholds=quantile(yHat[y.==1],0:0.01:1)
  sort!(thresholds)
  rocnums=MLBase.roc(y,yHat,thresholds);
  precision=map(p->(p.tp)/(p.tp+p.fp),rocnums);
  recall=map(p->(p.tp)/(p.tp+p.fn),rocnums);
  writedlm(ofname,hcat(recall,precision,thresholds));
end