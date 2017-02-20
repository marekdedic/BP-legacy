push!(LOAD_PATH, "EduNets/src");

using EduNets

function trainModelUrl!(model::UrlModelCompound, loss::AbstractLoss, dataset::UrlDatasetCompound; T::DataType=Float32, lambda::Float32=1f-6, iter::Int=1000)::Void
  #sc=ScalingLayer(dataset.x);
  gg=model2vector(model);
  g=deepcopy(model)

  function optFun(x::Vector)
    update!(model,x);
	dss = dataset;
    #dss = sample(dataset,[1000,1000]);

	dsd = dss.domains;
	dsp = dss.paths;
	dsq = dss.queries;

	oo = forward!(model, dss);

	#=
	od = forward!(model.domainModel, dsd.x, (dsd.bags,));
	op =forward!(model.pathModel, dsp.x, (dsp.bags,));
	oq = forward!(model.queryModel, dsq.x, (dsq.bags,));

	o = vcat(od[end], op[end], oq[end]);   #this is expensive!!!!

	#o[1:size(od,1),:]=od;
	#o[size(od,1)+1:,:]=od;

	(something, oo)=forward!(model.model,o);
	=#

	(f,goo) = gradient!(loss,oo,dss.labels); #calculate the gradient of the loss function 


	go=backprop!(model.model,(model.state.o,),goo,g.model);

	dsize = size(model.domainModel[end], 1);
	psize = size(model.pathModel[end], 1);
	qsize = size(model.queryModel[end], 1);

	god=view(go,1:dsize,:);
	gop=view(go,dsize + 1:dsize + psize,:);
	goq=view(go,dsize + psize + 1:dsize + psize + qsize,:);

	gradient!(model.domainModel, (dataset.domains.x, model.state.od), (dsd.bags,), god, g.domainModel);
	gradient!(model.pathModel, (dataset.paths.x, model.state.op), (dsp.bags,), gop, g.pathModel);
	gradient!(model.queryModel, (dataset.queries.x, model.state.oq), (dsq.bags,), goq, g.queryModel);

	model2vector!(g, gg);

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
