push!(LOAD_PATH, "EduNets/src");

using EduNets

type UrlModelCompound{A<:AbstractModel, B<:AbstractModel, C<:AbstractModel, D<:AbstractModel, T<:AbstractFloat}<:AbstractModel
	domainModel::A;
	pathModel::B;
	queryModel::C;
	model::D;
end

function UrlModelCompound(A::AbstractModel, B::AbstractModel, C::AbstractModel, D::AbstractModel; T::DataType = Float32)
	UrlModel
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
	vcat(model2vector(model.domainModel), model2vector(model.pathModel), model2vector(model.queryModel)model2vector(model.model))
end

