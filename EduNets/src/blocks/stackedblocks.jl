export StackedBlocks,gradient!,add!

function stacked(n::Int)
  name="StackedBlocks$(n)"
  inputnames=vcat("x",map(i->@sprintf("model.o%d",i),1:n))
  ginputnames=vcat("gx",map(i->@sprintf("go%d",i),1:n))

  expressions=Array{Expr,1}(0)

  typelist=""
   for i in 1:n
    typelist*="A$(i)<:AbstractModel"
    typelist*=(i<n)?",":"";
  end

  push!(expressions,parse("export $(name)"))
  #This is the definition of the model
  deftype="type $(name){"*typelist*",T<:AbstractFloat}<:AbstractModel\n";
  deftype*=mapreduce(i->"l$(i)::A$(i);\n",*,1:n)
  deftype*=mapreduce(i->"o$(i)::StridedMatrix{T};\n",*,1:n)
  deftype*="end"
  push!(expressions,parse(deftype))

  constructorfun="function $(name){$(typelist)}("
  constructorfun*=mapreduce(i->"a$(i)::A$(i),",*,1:n-1)
  constructorfun*="a$(n)::A$(n);T::DataType=Float64)\n"
  constructorfun*="aa=zeros(T,0,0)\n"
  constructorfun*="return($(name)("
  constructorfun*=mapreduce(i->"a$(i),",*,1:n)
  constructorfun*=repeat("aa,",n-1)
  constructorfun*="aa))\n"
  constructorfun*="end"
  push!(expressions,parse(constructorfun))

  constructorfun="function StackedBlocks{$(typelist)}("
  constructorfun*=mapreduce(i->"a$(i)::A$(i),",*,1:n-1)
  constructorfun*="a$(n)::A$(n);T::DataType=Float64)\n"
  constructorfun*="aa=zeros(T,0,0)\n"
  constructorfun*="return($(name)("
  constructorfun*=mapreduce(i->"a$(i),",*,1:n)
  constructorfun*=repeat("aa,",n-1)
  constructorfun*="aa))\n"
  constructorfun*="end"
  push!(expressions,parse(constructorfun))

  updatefun="@inline function update!(model::$name,theta::Vector;offset::Int=1)\n"
  updatefun*=mapreduce(i->@sprintf("offset=update!(model.l%d,theta;offset=offset)\n",i),*,1:n)
  updatefun*="return(offset);\n"
  updatefun*="end"
  push!(expressions,parse(updatefun))

  addfun="@inline function add!(model::$name,theta::Vector;offset::Int=1)\n"
  addfun*=mapreduce(i->@sprintf("offset=add!(model.l%d,theta;offset=offset)\n",i),*,1:n)
  addfun*="return offset;\n"
  addfun*="end"
  push!(expressions,parse(addfun))

  model2vectorfun1="@inline function model2vector!(model::$name,theta::Vector;offset::Int=1)\n"
  model2vectorfun1*=mapreduce(i->@sprintf("offset=model2vector!(model.l%d,theta;offset=offset)\n",i),*,1:n)
  model2vectorfun1*="return offset;\n"
  model2vectorfun1*="end"
  push!(expressions,parse(model2vectorfun1))

  model2vectorfun2="@inline function model2vector(model::$name)\n"
  model2vectorfun2*=" vcat("
  model2vectorfun2*=mapreduce(i->@sprintf("model2vector(model.l%d),",i),*,1:n-1)
  model2vectorfun2*=@sprintf("model2vector(model.l%d))\n",n)
  model2vectorfun2*="end"
  push!(expressions,parse(model2vectorfun2))


  forwardfun2="@inline function forward!(model::$name,x::StridedMatrix,)\n"
  forwardfun2*=@sprintf("model.o1=forward!(model.l1,x)\n")
  forwardfun2*=mapreduce(i->"model.o$(i)=forward!(model.l$(i),model.o$(i-1))\n",*,2:n)
  forwardfun2*="end"
  push!(expressions,parse(forwardfun2))

  backwardfun=@sprintf("@inline function backprop!(model::%s,x::StridedMatrix,go%d::StridedMatrix,gmodel::%s;update::Bool=false)\n",name,n,name)
  backwardfun*=mapreduce(i->@sprintf("%s=backprop!(model.l%d,%s,%s,gmodel.l%d,update=update)\n",ginputnames[i],i,inputnames[i],ginputnames[i+1],i),*,n:-1:1)
  backwardfun*="end"
  push!(expressions,parse(backwardfun))

  gradientfun=@sprintf("@inline function gradient!(model::%s,x::StridedMatrix,go%d::StridedMatrix,gmodel::%s;update::Bool=false)\n",name,n,name)
  gradientfun*=mapreduce(i->@sprintf("%s=backprop!(model.l%d,%s,%s,gmodel.l%d,update=update)\n",ginputnames[i],i,inputnames[i],ginputnames[i+1],i),*,n:-1:2)
  gradientfun*=@sprintf("gradient!(model.l1,%s,%s,gmodel.l1,update=update)\n",inputnames[1],ginputnames[2])
  gradientfun*="end"
  push!(expressions,parse(gradientfun))

  l1regularizefun1="@inline function l1regularize!(model::$name,g::$name,lambda::AbstractFloat)\n"
  l1regularizefun1*="f=zero(typeof(lambda))\n"
  l1regularizefun1*=mapreduce(i->@sprintf("f+=l1regularize!(model.l%d,g.l%d,lambda)\n",i,i,),*,1:n)
  l1regularizefun1*="return(f)\n"
  l1regularizefun1*="end"
  push!(expressions,parse(l1regularizefun1))

  initfun1="@inline function init!(model::$name,x::StridedMatrix)\n"
  initfun1*="init!(model.l1,x)\n"
  initfun1*="model.o1=forward!(model.l1,x)\n"
  for i in 2:n
    initfun1*="init!(model.l$(i),model.o$(i-1))\n"
    initfun1*="model.o$(i)=forward!(model.l$(i),model.o$(i-1))\n"
  end
  initfun1*="end"
  push!(expressions,parse(initfun1))
  return(expressions)
end

for n in 2:10
  expressions=stacked(n)
  for e in expressions
    eval(e)
  end
end