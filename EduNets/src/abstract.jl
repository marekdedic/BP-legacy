export AbstractModel,AbstractLayer,AbstractMatMulLayer,AbstractDataset,AbstractLoss,Bags;

abstract Model;
abstract AbstractModel;
abstract AbstractLayer<:AbstractModel;
abstract AbstractMatMulLayer<:AbstractLayer;
abstract AbstractDataset;
abstract AbstractLoss;
abstract AbstractPooling<:AbstractLayer;

Bags = Union{Array{UnitRange{Int64},1},Array{Array{Int64,1},1}};