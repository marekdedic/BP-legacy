using EduNets;
import EduNets: update!, model2vector!, model2vector, project!, forward!, fgradient!;

type UrlModel{A<:Tuple, B<:Tuple, C<:Tuple, D<:Tuple}<:AbstractModel
	domainModel::A;
	pathModel::B;
	queryModel::C;
	model::D;
end

function UrlModel(domainModel::Tuple, pathModel::Tuple, queryModel::Tuple, model::Tuple)
	UrlModel(domainModel, pathModel, queryModel, model);
end

# update = vector2model
function update!(model::UrlModel, theta::Vector; offset::Int = 1)
	offset = update!(model.domainModel, theta; offset = offset);
	offset = update!(model.pathModel, theta; offset = offset);
	offset = update!(model.queryModel, theta; offset = offset);
	offset = update!(model.model, theta; offset = offset);
end

function model2vector!(model::UrlModel, theta::Vector; offset::Int = 1)
	offset = model2vector!(model.domainModel, theta; offset = offset);
	offset = model2vector!(model.pathModel, theta; offset = offset);
	offset = model2vector!(model.queryModel, theta; offset = offset);
	offset = model2vector!(model.model, theta; offset = offset);
end

function model2vector(model::UrlModel)
	vcat(model2vector(model.domainModel), model2vector(model.pathModel), model2vector(model.queryModel), model2vector(model.model))
end

function project!(model::UrlModel, dataset::UrlDataset)
	od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,));
	op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,));
	oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,));

	o::StridedMatrix = Matrix{Float32}(size(od[end], 1) + size(op[end], 1) + size(oq[end], 1), size(od[end], 2))
	dsize = size(od[end], 1);
	psize = size(op[end], 1);
	o[1:dsize, :] = od[end];
	o[dsize + 1:dsize + psize, :] = op[end];
	o[dsize + psize + 1:end, :] = oq[end];

	oo = forward!(model.model, o)[end];
	return oo;
end

function forward!(model::UrlModel, dataset::UrlDataset)
	od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,));
	op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,));
	oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,));

	o::StridedMatrix = Matrix{Float32}(size(od[end], 1) + size(op[end], 1) + size(oq[end], 1), size(od[end], 2))
	dsize = size(od[end], 1);
	psize = size(op[end], 1);
	o[1:dsize, :] = od[end];
	o[dsize + 1:dsize + psize, :] = op[end];
	o[dsize + psize + 1:end, :] = oq[end];

	oo = forward!(model.model, o);
	return oo;
end

function fgradient!(model::UrlModel,loss::EduNets.AbstractLoss, dataset::UrlDataset, g::UrlModel)
	od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,));
	op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,));
	oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,));

	o::StridedMatrix = Matrix{Float32}(size(od[end], 1) + size(op[end], 1) + size(oq[end], 1), size(od[end], 2))
	dsize = size(od[end], 1);
	psize = size(op[end], 1);
	o[1:dsize, :] = od[end];
	o[dsize + 1:dsize + psize, :] = op[end];
	o[dsize + psize + 1:end, :] = oq[end];

	oo = forward!(model.model, o);
	(f, goo) = gradient!(loss, oo[end], dataset.y); #calculate the gradient of the loss function 

	(f1,go)=fbackprop!(model.model,oo, goo, g.model);

	dsize = size(model.domainModel[end], 1);
	psize = size(model.pathModel[end], 1);
	qsize = size(model.queryModel[end], 1);

	god = view(go, 1:dsize,:);
	gop = view(go, dsize + 1:dsize + psize,:);
	goq = view(go, dsize + psize + 1:dsize + psize + qsize,:);

	f2=fgradient!(model.domainModel, od, (dataset.domains.bags,), god, g.domainModel);
	f3=fgradient!(model.pathModel, op, (dataset.paths.bags,), gop, g.pathModel);
	f4=fgradient!(model.queryModel, oq, (dataset.queries.bags,), goq, g.queryModel);
	return f+f1+f2+f3+f4;
end

function addsoftmax(model::UrlModel,T)
	UrlModel(model.domainModel, model.pathModel, model.queryModel, (model.model...,SoftmaxLayer(size(model.model[end], 2), T = T)))
end

