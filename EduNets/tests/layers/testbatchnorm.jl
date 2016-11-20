using EduNets

function f(x::Vector,w::Matrix,layer)
  go=w
  x=reshape(x,size(w))
  O=forward!(layer,x)
  gx=backprop!(layer,x,w,layer)
  gx[2:end,:]=0
  fo=sum(O.*w,2)
  return(fo[1],reshape(gx,length(gx)))
end

x=randn(5,5)
w=randn(5,5)
layer=BatchNormLayer(5)
testgradient(s->f(s,w,layer),reshape(x,length(x)),verbose=2);
