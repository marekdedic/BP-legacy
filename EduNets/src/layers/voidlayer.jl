export VoidLayer,l1regularize!
type VoidLayer<:AbstractLayer
    k::Int
    T::DataType
end

function VoidLayer(k::Int;T::DataType=Float64)
  return(VoidLayer(k,T));
end

function length(model::VoidLayer)
    return(0)
end

function size(model::VoidLayer)
    return((model.k,model.k));
end

function init!(layer::VoidLayer,X::StridedMatrix)
end

function size(model::VoidLayer,k::Int)
    return(model.k);
end

@inline function model2vector(model::VoidLayer)
  return(zeros(model.T,0));
end

@inline function model2vector!(model::VoidLayer,theta::AbstractArray;offset::Int=1)
    return(offset);
end

@inline function add!(model::VoidLayer,theta::AbstractArray;offset::Int=1)
    return(offset);
end

@inline function update!(model::VoidLayer,theta::AbstractArray;offset::Int=1)
    return(offset)
end

@inline function l1regularize!(model::VoidLayer,g::VoidLayer,lambda::AbstractFloat)
  return(model.T(0))
end

""" Regularizes by a log of L1 norm of rows / columns specified by the dimension dimension( 2 --- rows, 1--- columns) """
@inline function logl1regularize!(model::VoidLayer,g::VoidLayer,lambda::AbstractFloat,epsilon::AbstractFloat;dimension=2)
  return(model.T(0))
end

@inline function forward!(model::VoidLayer,X::StridedMatrix)
  return(X)
end

@inline function backprop(model::VoidLayer,X::StridedMatrix,gO::StridedMatrix)
  return(deepcopy(model),gO)
end

@inline function backprop!(model::VoidLayer,X::StridedMatrix,gO::StridedMatrix,g::VoidLayer;update=false)
  return(gO)
end


@inline function gradient(model::VoidLayer,X::StridedMatrix,gO::StridedMatrix)
  return(deepcopy(model))
end

@inline function gradient!(model::VoidLayer,X::StridedMatrix,gO::StridedMatrix,g::VoidLayer;update=false)
end