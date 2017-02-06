push!(LOAD_PATH, "EduNets/src")

using EduNets

type UrlModelState{T<:AbstractFloat}
	O1::StridedMatrix{T};
	O2::StridedMatrix{T};
	O3::StridedMatrix{T};
	O4::StridedMatrix{T};
	O5::StridedMatrix{T};
end;

function UrlModelState(;T::DataType = Float64)
	zm = zeros(T, 0, 0);
	return UrlModelState(zm, zm, zm, zm, zm);
end

type UrlModel{A<:AbstractModel, B<:AbstractModel, C<:AbstractModel, D<:AbstractModel, E<:AbstractModel, F<:AbstractLoss, T<:AbstractFloat}<:AbstractModel
	perUrlPart::A;
	perUrlPartPooling::B;
	perUrl::C;
	perUrlPooling::D;
	all::E;
	loss::F;

	state::UrlModelState{T};
end;

function UrlModel(perUrlPart::AbstractModel, perUrlPartPooling::AbstractModel, perUrl::AbstractModel, perUrlPooling::AbstractModel, all::AbstractModel, loss::AbstractLoss; T::DataType = Float64)::UrlModel
	return UrlModel(perUrlPart, perUrlPartPooling, perUrl, perUrlPooling, all, loss, UrlModelState(T = T));
end

@inline function model2vector(model::UrlModel)::Vector
	vcat(model2vector(model.perUrlPart), model2vector(model.perUrlPartPooling), model2vector(perUrl), model2vector(perUrlPooling), model2vector(all))
end

@inline function update!(model::UrlModel, theta::AbstractArray; offset::Int = 1)::Int
	offset=update!(model.perUrlPart, theta; offset = offset)
	offset=update!(model.perUrlPartPooling, theta; offset = offset)
	offset=update!(model.perUrl, theta; offset = offset)
	offset=update!(model.perUrlPooling, theta; offset = offset)
	offset=update!(model.all, theta; offset = offset)
	return(offset)
end

