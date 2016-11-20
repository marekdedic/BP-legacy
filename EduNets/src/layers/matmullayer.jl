export length,size,model2vector,update!,l1regularize!,logl1regularize!,add!
function length(model::AbstractMatMulLayer)
    return(length(model.W)+length(model.B))
end

function size(model::AbstractMatMulLayer)
    return(size(model.W));
end

function size(model::AbstractMatMulLayer,k::Int)
    return(size(model.W,k::Int));
end

function model2vector(model::AbstractMatMulLayer)
  theta=zeros(eltype(model.W),length(model.W)+length(model.B));
  theta[1:length(model.W)]=model.W;
  theta[length(model.W)+1:end]=model.B;
  return(theta);
end

function model2vector!(model::AbstractMatMulLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(theta,offset,model.W,1,length(model.W));
    Base.LinAlg.copy!(theta,offset+length(model.W),model.B,1,length(model.B));
    return(offset+length(model));
end

function init!(layer::AbstractMatMulLayer,X::StridedMatrix)
    idxs=sample(1:size(X,2),size(layer.W,2))
    layer.W=X[:,idxs];
    fill!(layer.B,0)
end

function add!(model::AbstractMatMulLayer,theta::AbstractArray;offset::Int=1)
  @simd for i in 1:length(model.W)
    @inbounds theta[offset+i-1]+=model.W[i]
  end
  offset+=length(model.W)

  @simd for i in 1:length(model.B)
    @inbounds theta[offset+i-1]+=model.B[i]
  end    
  offset+=length(model.B)
  return(offset);
end

function update!(model::AbstractMatMulLayer,theta::AbstractArray;offset::Int=1)
    Base.LinAlg.copy!(model.W,1,theta,offset,length(model.W))
    Base.LinAlg.copy!(model.B,1,theta,length(model.W)+offset,length(model.B))
    return(offset+length(model))
end

function l1regularize!(model::AbstractMatMulLayer,g::AbstractMatMulLayer,lambda::AbstractFloat)
  f=eltype(model.W)(0.0);
  f+=lambda*sumabs(model.W)
  g.W+=lambda*sign(model.W)
  return(f)
end

""" Regularizes by a log of L1 norm of rows / columns specified by the dimension dimension( 2 --- rows, 1--- columns) """
function logl1regularize!(model::AbstractMatMulLayer,g::AbstractMatMulLayer,lambda::AbstractFloat,epsilon::AbstractFloat;dimension=2)
  arglog=mean(abs(model.W),dimension)+epsilon;
  f=lambda*(mean(log(arglog))-log(epsilon))
  g.W+=lambda*broadcast(*,sign(model.W),1./(size(model.W,dimension)*arglog));
  return(f)
end