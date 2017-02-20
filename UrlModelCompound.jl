push!(LOAD_PATH, "EduNets/src");

using EduNets;
import EduNets: update!, model2vector, model2vector!, forward!;

type UrlModelCompoundState{T<:AbstractFloat}
	od::StridedMatrix{T};
	op::StridedMatrix{T};
	oq::StridedMatrix{T};
	o::StridedMatrix{T};
end

function UrlModelCompoundState(;T::DataType = Float32)
	z = zeros(T, 0, 0);
	UrlModelCompoundState(z, z, z, z);
end

type UrlModelCompound{A<:Tuple, B<:Tuple, C<:Tuple, D<:Tuple}<:AbstractModel
	domainModel::A;
	pathModel::B;
	queryModel::C;
	model::D;

	state::UrlModelCompoundState;
end

function UrlModelCompound(domainModel::Tuple, pathModel::Tuple, queryModel::Tuple, model::Tuple)
	UrlModelCompound(domainModel, pathModel, queryModel, model, UrlModelCompoundState());
end

# update = vector2model
function update!(model::UrlModelCompound, theta::Vector; offset::Int = 1)
	offset = update!(model.domainModel, theta; offset = offset);
	offset = update!(model.pathModel, theta; offset = offset);
	offset = update!(model.queryModel, theta; offset = offset);
	offset = update!(model.model, theta; offset = offset);
end

function model2vector!(model::UrlModelCompound, theta::Vector; offset::Int = 1)
	offset = model2vector!(model.domainModel, theta; offset = offset);
	offset = model2vector!(model.pathModel, theta; offset = offset);
	offset = model2vector!(model.queryModel, theta; offset = offset);
	offset = model2vector!(model.model, theta; offset = offset);
end

function model2vector(model::UrlModelCompound)
	vcat(model2vector(model.domainModel), model2vector(model.pathModel), model2vector(model.queryModel), model2vector(model.model))
end

function forward!(model::UrlModelCompound, dataset::UrlDatasetCompound)
	model.state.od = forward!(model.domainModel, dataset.domains.x, (dataset.domains.bags,))[end];
	model.state.op = forward!(model.pathModel, dataset.paths.x, (dataset.paths.bags,))[end];
	model.state.oq = forward!(model.queryModel, dataset.queries.x, (dataset.queries.bags,))[end];

	model.state.o = vcat(model.state.od, model.state.op, model.state.oq);   #this is expensive!!!!

	o2 = forward!(model.model, model.state.o);
	return o2[end];
end

