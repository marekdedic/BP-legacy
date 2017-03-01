push!(LOAD_PATH, "EduNets/src");

using EduNets;
import EduNets: update!, model2vector, model2vector!, forward!, gradient!;

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

type UrlModel{A<:Tuple, B<:Tuple, C<:Tuple, D<:Tuple, E<:AbstractLoss}<:AbstractModel
	domainModel::A;
	pathModel::B;
	queryModel::C;
	model::D;
	loss::E;

	state::UrlModelState;
end

function UrlModel(domainModel::Tuple, pathModel::Tuple, queryModel::Tuple, model::Tuple, loss::AbstractLoss)
	UrlModel(domainModel, pathModel, queryModel, model, loss, UrlModelState());
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

function gradient!(model::UrlModel, dataset::UrlDataset, g::UrlModel, oo)
	(f, goo) = gradient!(model.loss, oo, dataset.labels); #calculate the gradient of the loss function 

	go=backprop!(model.model, (model.state.o,), goo, g.model);

	dsize = size(model.domainModel[end], 1);
	psize = size(model.pathModel[end], 1);
	qsize = size(model.queryModel[end], 1);

	god = view(go, 1:dsize,:);
	gop = view(go, dsize + 1:dsize + psize,:);
	goq = view(go, dsize + psize + 1:dsize + psize + qsize,:);

	gradient!(model.domainModel, (dataset.domains.x, model.state.od), (dataset.domains.bags,), god, g.domainModel);
	gradient!(model.pathModel, (dataset.paths.x, model.state.op), (dataset.paths.bags,), gop, g.pathModel);
	gradient!(model.queryModel, (dataset.queries.x, model.state.oq), (dataset.queries.bags,), goq, g.queryModel);
	return f;
end

function fgradient!(model::UrlModel, dataset::UrlDataset, g::UrlModel)
	oo = forward!(model, dataset);
	gradient!(model, dataset, g, oo)
end

