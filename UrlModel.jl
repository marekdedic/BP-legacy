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
	return UrlModelState(zm, zm, zm);
end

type UrlModel{A<:AbstractModel, B<:AbstractModel, C<:AbstractModel, D<:AbstractModel, E<:AbstractModel, F<:AbstractLoss, T<:AbstractFloat}<:AbstractModel
	perUrlPart::A; # Why one model, why not copies of a smaller model?
	perUrlPartPooling::B;
	perUrl::C;
	perUrlPooling::D;
	all::E;
	loss::F;

	state::UrlModelState{T};
end;

function UrlModel(perUrlPart::AbstractModel, perUrlPartPooling::AbstractModel, perUrl::AbstractModel, perUrlPooling::AbstractModel, all::AbstractModel, loss::AbstractLoss; T::DataType = Float64)
	return UrlModel(eprUrlPart, perUrlPartPooling, perUrl, perUrlPooling, all, loss, UrlModelState(T = T));
end

