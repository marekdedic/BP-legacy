module EduNets
  import EduNets.gradient
  export update!,model2vector,model2vector!,forward!,gradient!,backprop!,add!;
  include("abstract.jl")
  include("datatypes/columnsparsematrix.jl")
  include("datatypes/dataset.jl")
  include("datatypes/doublebagdataset.jl")
  include("datatypes/singlebagdataset.jl")


  include("layers/scaling.jl")
  include("layers/featuresubset.jl")
  include("layers/voidlayer.jl")
  include("layers/matmullayer.jl")
  include("layers/hingeloss.jl")
  include("layers/clusterloss.jl")
  include("layers/equalhingeloss.jl")
  include("layers/multihingeloss.jl")
  include("layers/logisticloss.jl")
  include("layers/fp50loss.jl")
  include("layers/l2loss.jl")
  include("layers/fp50losshinge.jl")
  include("layers/softmaxloss.jl")
  include("layers/linear.jl")
  include("layers/relu.jl")
  include("layers/leakyrelu.jl")
  include("layers/batchnorm.jl")
  include("layers/exponential.jl")
  include("layers/precisionat.jl")
  include("layers/reluMax.jl")
  include("layers/reluMean.jl")
  include("layers/softmax.jl")
  include("layers/lqpoolinglayer.jl")
  include("layers/gausspoolinglayer.jl")
  include("layers/maxpoolinglayer.jl")
  include("layers/meanpoolinglayer.jl")
  include("layers/whitening.jl")
  include("layers/dropout.jl")

  include("layers/randomgauss.jl")
  include("layers/randomunitgauss.jl")
  include("layers/meanl2bagloss.jl")
  include("layers/pairwisel2bagloss.jl")
  include("layers/hausdorffbagloss.jl")
  include("layers/gaussmmdbagloss.jl")
  include("layers/kldivxybagloss.jl")
  include("layers/kldivyxbagloss.jl")

  include("blocks/relurelumax.jl")
  include("blocks/stackedblocks.jl")
  include("blocks/relurelumean.jl")
  include("blocks/twojoinedblocks.jl")
  include("blocks/threejoinedblocks.jl")

  include("../examples/singlebag.jl")

  include("optimization/adam.jl")


  #tests of gradients
  include("gradient_estimate.jl")
end
