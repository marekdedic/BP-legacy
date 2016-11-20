
export TwoJoinedBlocks, update!,model2vector,model2vector!,forward!,gradient!,backprop!;
type TwoJoinedBlocks{T}<:AbstractModel
    first::AbstractModel;    #convert tokens to some dense representation. This layer is the same for all parts of the url.
    second::AbstractModel;    #convert tokens to some dense representation. This layer is the same for all parts of the url.
    O::Matrix{T};
    k::Tuple{UnitRange{Int64},UnitRange{Int64}};    #ranges where the output of first and second model will be folded
end

function TwoJoinedBlocks(first::AbstractModel,second::AbstractModel)
    kk=cumsum([1,size(first,2),size(second,2)])
    k=(kk[1]:kk[2]-1,kk[2]:kk[3]-1);
    return(TwoJoinedBlocks(first,second,zeros(0,0),k));
end

function length(model::TwoJoinedBlocks)
    return(length(model.first)+length(model.second));
end

@inline function update!{T}(model::TwoJoinedBlocks{T},theta::AbstractArray{T};offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
    return offset;
end

@inline function model2vector!{T}(model::TwoJoinedBlocks{T},theta::AbstractArray{T};offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
    return offset;
end

function model2vector(model::TwoJoinedBlocks)
    return(vcat(model2vector(model.first),model2vector(model.second)))
end

function size(model::TwoJoinedBlocks,k::Int)
    if k==2
        return(size(model.first,2)+size(model.second,2));
    end
    if k==1
        return((size(model.first,1),size(model.second,1)));
    end
end

@inline function forward!(model::TwoJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix)
    l=size(x1,2);
    if size(model.O,2)<size(model,2) || size(model.O,2)<l
        model.O=zeros(eltype(x1),size(model,2),l)
    end
    return(forward!(model,x1,x2,model.O))
end

@inline function forward!(model::TwoJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,O::StridedMatrix)
    if size(x1,2)!=size(x2,2)
        error("TwoJoinedBlocks::forward! sizes of input matrices are not the same")
    end
    l=size(x1,2);
    if size(O,2)<size(model,2) || size(O,2)<l
        error("TwoJoinedBlocks::forward! O is not big enough")
    end

    #first, we forward the quary part into its storage
    forward!(model.first,x1,view(O,model.k[1],1:l));
    forward!(model.second,x2,view(O,model.k[2],1:l));
    return(view(O,1:size(model,2),1:l))
end

@inline function forward!(model::TwoJoinedBlocks,x1::AbstractMatrix,bags1,x2::AbstractMatrix,bags2)
    l=length(bags1);
    if size(model.O,2)<size(model,2) || size(model.O,2)<l
        model.O=zeros(eltype(x1),size(model,2),l)
    end
    return(forward!(model,x1,bags1,x2,bags2,model.O))
end

@inline function forward!(model::TwoJoinedBlocks,x1::AbstractMatrix,bags1,x2::AbstractMatrix,bags2,O::StridedMatrix)
    if length(bags1)!=length(bags2)
        error("TwoJoinedBlocks::forward! bags should have the same length")
    end
    l=length(bags1);
    if size(O,2)<size(model,2) || size(O,2)<l
        error("TwoJoinedBlocks::forward! O is not big enough")
    end
    #first, we forward the quary part into its storage
    forward!(model.first,x1,bags1,view(O,model.k[1],1:l));
    forward!(model.second,x2,bags2,view(O,model.k[2],1:l));
    return(view(model.O,1:size(model,2),1:l))
end

@inline function gradient!(model::TwoJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,go,g::TwoJoinedBlocks)
    gradient!(model.first,x1,view(go,model.k[1],:),g.first)
    gradient!(model.second,x2,view(go,model.k[2],:),g.second)
end

@inline function backprop!(model::TwoJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,go,g::TwoJoinedBlocks)
    gx1=backprop!(model.first,x1,view(go,model.k[1],:),g.first)
    gx2=backprop!(model.second,x2,view(go,model.k[2],:),g.second)
    return(gx1,gx2)
end