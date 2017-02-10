push!(LOAD_PATH, "EduNets/src");

using EduNets

function trainModelUrl!(model::UrlModelCompound, loss::AbstractLoss, dataset::UrlDatasetCompound; T::DataType=Float32, lambda::Float32=1f-6, iter::Int=1000)::Void
  #sc=ScalingLayer(dataset.x);
  gg=model2vector(model);

  function optFun(x::Vector)
    update!(model,x);
	dss = dataset;
    #dss = sample(dataset,[1000,1000]);

	dsd = dss.domains;
	dsp = dss.paths;
	dsq = dss.queries;

	od = forward!(model.domainModel, dsd.x, (dsd.bags,));
	op =forward!(model.pathModel, dsp.x, (dsp.bags,));
	oq = forward!(model.queryModel, dsq.x, (dsq.bags,));

	o = vcat(od[end], op[end], oq[end]);   #this is expensive!!!!

	#o[1:size(od,1),:]=od;
	#o[size(od,1)+1:,:]=od;

	(something, oo)=forward!(model.model,o);

	(f,goo) = gradient!(loss,oo,dss.labels); #calculate the gradient of the loss function 

	g2=deepcopy(model.model);

	go=backprop!(model.model,(o,),goo,g2);

	god=view(go,1:size(od,1),:);
	gop=view(go,1:size(od,1),:);
	goq=view(go,1:size(od,1),:);

	gd = deepcopy(model.domainModel);
	gp = deepcopy(model.pathModel);
	gq = deepcopy(model.queryModel);

	gradient!(model.domainModel,od,(dsd.bags,),god,gd);
	gradient!(model.pathModel,op,(dsp.bags,),gop,gp);
	gradient!(model.queryModel,oq,(dsq.bags,),goq,gq);

	ggd = model2vector(gd);
	ggp = model2vector(gp);
	ggq = model2vector(gq);
	gg2 = model2vector(g2);
	gg = vcat(ggd, ggp, ggq, gg2);

	return(f,gg) #return function value and gradient of the error

    #f = gradient!(model,dss,g);
    #f += l1regularize!(model,g, T(lambda));
    #model2vector!(g,gg);
    #return(f,gg);
  end

  theta=model2vector(model);
  adam(optFun, theta, AdamOptions(;maxIter=iter));
  return nothing;
end
