
export ThreeJoinedBlocks, update!,model2vector,model2vector!,forward!,gradient!,backprop!;
type ThreeJoinedBlocks{T}<:AbstractModel
    first::AbstractModel;    #convert tokens to some dense representation. This layer is the same for all parts of the url.
    second::AbstractModel;    #convert tokens to some dense representation. This layer is the same for all parts of the url.
    third::AbstractModel;    #convert tokens to some dense representation. This layer is the same for all parts of the url.
    O::Matrix{T};
    k::Tuple{UnitRange{Int64},UnitRange{Int64},UnitRange{Int64}};    #ranges where the output of first and second model will be folded
end

function ThreeJoinedBlocks(first::AbstractModel,second::AbstractModel,third::AbstractModel)
    kk=cumsum([1,size(first,2),size(second,2),size(third,2)])
    k=(kk[1]:kk[2]-1,kk[2]:kk[3]-1,kk[3]:kk[4]-1);
    return(ThreeJoinedBlocks(first,second,third,zeros(0,0),k));
end

function length(model::ThreeJoinedBlocks)
    return(length(model.first)+length(model.second)+length(model.third));
end

function update!{T}(model::ThreeJoinedBlocks{T},theta::AbstractArray{T};offset::Int=1)
    offset=update!(model.first,theta;offset=offset)
    offset=update!(model.second,theta;offset=offset)
    offset=update!(model.third,theta;offset=offset)
    return offset;
end

function model2vector!{T}(model::ThreeJoinedBlocks{T},theta::AbstractArray{T};offset::Int=1)
    offset=model2vector!(model.first,theta;offset=offset);
    offset=model2vector!(model.second,theta;offset=offset);
    offset=model2vector!(model.third,theta;offset=offset);
    return offset;
end

function model2vector(model::ThreeJoinedBlocks)
    return(vcat(model2vector(model.first),model2vector(model.second),model2vector(model.third)))
end

function size(model::ThreeJoinedBlocks,k::Int)
    if k==2
        return(size(model.first,2)+size(model.second,2)+size(model.third,2));
    end
    if k==1
        return((size(model.first,1),size(model.second,1),size(model.third,1)));
    end
end

function forward!(model::ThreeJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,x3::AbstractMatrix)
    l=size(x1,2);
    if size(model.O,2)<size(model,2) || size(model.O,2)<l
        model.O=zeros(eltype(x1),size(model,2),l)
    end
    return(forward!(model,x1,x2,x3,model.O))
end

function forward!(model::ThreeJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,x3::AbstractMatrix,O::StridedMatrix)
    if size(x1,2)!=size(x2,2)
        error("ThreeJoinedBlocks::forward! sizes of input matrices are not the same")
    end
    l=size(x1,2);
    if size(O,2)<size(model,2) || size(O,2)<l
        error("ThreeJoinedBlocks::forward! O is not big enough")
    end

    #first, we forward the quary part into its storage
    forward!(model.first,x1,view(O,model.k[1],1:l));
    forward!(model.second,x2,view(O,model.k[2],1:l));
    forward!(model.third,x3,view(O,model.k[3],1:l));
    return(view(O,1:size(model,2),1:l))
end

function forward!(model::ThreeJoinedBlocks,x1::AbstractMatrix,bags1,x2::AbstractMatrix,bags2,x3::AbstractMatrix,bags3)
    l=length(bags1);
    if size(model.O,2)<size(model,2) || size(model.O,2)<l
        model.O=zeros(eltype(x1),size(model,2),l)
    end
    return(forward!(model,x1,bags1,x2,bags2,x3,bags3,model.O))
end

function forward!(model::ThreeJoinedBlocks,x1::AbstractMatrix,bags1,x2::AbstractMatrix,bags2,x3::AbstractMatrix,bags3,O::StridedMatrix)
    if length(bags1)!=length(bags2)
        error("ThreeJoinedBlocks::forward! bags should have the same length")
    end
    l=length(bags1);
    if size(O,2)<size(model,2) || size(O,2)<l
        error("ThreeJoinedBlocks::forward! O is not big enough")
    end
    #first, we forward the quary part into its storage
    forward!(model.first,x1,bags1,view(O,model.k[1],1:l));
    forward!(model.second,x2,bags2,view(O,model.k[2],1:l));
    forward!(model.third,x3,bags3,view(O,model.k[3],1:l));
    return(view(model.O,1:size(model,2),1:l))
end

function gradient!(model::ThreeJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,x3::AbstractMatrix,go,g::ThreeJoinedBlocks)
    gradient!(model.first,x1,view(go,model.k[1],:),g.first)
    gradient!(model.second,x2,view(go,model.k[2],:),g.second)
    gradient!(model.third,x3,view(go,model.k[3],:),g.third)
end

function backprop!(model::ThreeJoinedBlocks,x1::AbstractMatrix,x2::AbstractMatrix,x3::AbstractMatrix,go,g::ThreeJoinedBlocks)
    gx1=backprop!(model.first,x1,view(go,model.k[1],:),g.first)
    gx2=backprop!(model.second,x2,view(go,model.k[2],:),g.second)
    gx3=backprop!(model.third,x3,view(go,model.k[3],:),g.third)
    return(gx1,gx2,gx3)
end