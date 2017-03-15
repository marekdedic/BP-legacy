using EduNets;
import EduNets: update!, model2vector, model2vector!, forward!, gradient!, fgradient!;

type UrlModelState{T<:AbstractFloat}
	od::StridedMatrix{T};
	op::StridedMatrix{T};
	oq::StridedMatrix{T};
	o::StridedMatrix{T};
end

function UrlModelState(;T::DataType = Float32)
	z = zeros(T, 0, 0);
	UrlModelState(z, z, z, z);
end

type UrlModel{A<:Tuple, B<:Tuple, C<:Tuple, D<:Tuple}<:AbstractModel
	domainModel::A;
	pathModel::B;
	queryModel::C;
	model::D;

	state::UrlModelState;
end

function UrlModel(domainModel::Tuple, pathModel::Tuple, queryModel::Tuple, model::Tuple)
	UrlModel(domainModel, pathModel, queryModel, model,UrlModelState());
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

function forward!(model::UrlModel, dataset::UrlDataset)
	model.state.od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,))[end];
	model.state.op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,))[end];
	model.state.oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,))[end];

	model.state.o::StridedMatrix = Matrix{Float32}(size(model.state.od, 1) + size(model.state.op, 1) + size(model.state.oq, 1), size(model.state.od, 2))
	dsize = size(model.state.od, 1);
	psize = size(model.state.op, 1);
	model.state.o[1:dsize, :] = model.state.od;
	model.state.o[dsize + 1:dsize + psize, :] = model.state.op;
	model.state.o[dsize + psize + 1:end, :] = model.state.oq;

	o2 = forward!(model.model, model.state.o)[end];
	return o2;
end

function fgradient!(model::UrlModel,loss::EduNets.AbstractLoss, dataset::UrlDataset, g::UrlModel)
	oo=forward!(model,dataset)
	(f, goo) = gradient!(loss, oo, dataset.y); #calculate the gradient of the loss function 

	(f1,go)=fbackprop!(model.model, (model.state.o,), goo, g.model);

	dsize = size(model.domainModel[end], 1);
	psize = size(model.pathModel[end], 1);
	qsize = size(model.queryModel[end], 1);

	god = view(go, 1:dsize,:);
	gop = view(go, dsize + 1:dsize + psize,:);
	goq = view(go, dsize + psize + 1:dsize + psize + qsize,:);

	f2=fgradient!(model.domainModel, (dataset.domains.x, model.state.od), (dataset.domains.bags,), god, g.domainModel);
	f3=fgradient!(model.pathModel, (dataset.paths.x, model.state.op), (dataset.paths.bags,), gop, g.pathModel);
	f4=fgradient!(model.queryModel, (dataset.queries.x, model.state.oq), (dataset.queries.bags,), goq, g.queryModel);
	return f+f1+f2+f3+f4;
end

function addsoftmax(model::UrlModel,T)
	UrlModel(model.domainModel,model.pathModel,model.queryModel, (model.model...,SoftMaxLayer(size(model[end],2),T=T)),UrlModelState())
end

