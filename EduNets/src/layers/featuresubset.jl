export FeatureSubsetLayer;

type FeatureSubsetLayer{T}<:AbstractLayer
    idxs::Vector{Int} #this will be subtracted
    O::Matrix{T}
end

function FeatureSubsetLayer(idxs::UnitRange{Int64};T::DataType=Float64)
    FeatureSubsetLayer(collect(idxs),T=T)
end

function FeatureSubsetLayer(idxs::Vector{Int};T::DataType=Float64)
    return FeatureSubsetLayer{T}(deepcopy(idxs),zeros(T,0,0));
end

function FeatureSubsetLayer(filename::AbstractString;T::DataType=Float64)
    return FeatureSubsetLayer{T}(squeeze(readdlm(filename,Int),2),zeros(T,0,0));
end

function forward!{T}(model::FeatureSubsetLayer{T},x::StridedMatrix{T})
    if maximum(model.idxs)>size(x,1)
        error("FeatureSubsetLayer: number of features in x is smaller then the number of to-be selected indexes")
    end
    return(x[model.idxs,:])
end

    