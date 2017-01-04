require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'


function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

print(trainData:size())

saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
print(classes[trainLabels[100]]) -- display the 100-th image class


--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:

local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



--  ****************************************************************
--  Define our neural network
--  ****************************************************************
--[[
Flips image src horizontally (left<->right). 
If dst is provided, it is used to store the output image. 
Otherwise, returns a new res Tenso
]]
local function hflip(x)
   return torch.random(0,1) == 1 and x or image.hflip(x)
end

local function randomcrop(im , pad, randomcrop_type)
   if randomcrop_type == 'reflection' then
      -- Each feature map of a given input is padded with the replication of the input boundary
      module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
   elseif randomcrop_type == 'zero' then
      -- Each feature map of a given input is padded with specified number of zeros.
	  -- If padding values are negative, then input is cropped.
      module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
   end
	
   local padded = module:forward(im:float())
   local x = torch.random(1,pad*2 + 1)
   local y = torch.random(1,pad*2 + 1)
   image.save('img2ZeroPadded.jpg', padded)

   return padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
end


do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local flip_mask = torch.randperm(input:size(1))
      for i=1,input:size(1) do
		-- if flip_mask[i] % 3 == 1 then image.crop(input[i],'c', 32, 32) end
		-- if flip_mask[i] % 3 == 0 then image.scale(input[i], 32, 32) end
      end
    end
    self.output:set(input:cuda())
    return self.output
  end
end



local model = nn.Sequential()
model:add(nn.BatchFlip():float())

local function Block(...)

  local arg = {...}

  model:add(nn.SpatialConvolution(...))

  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))

  model:add(nn.ReLU(true))

  return model

end



--best try:

Block(3,64,5,5,1,1,2,2)

Block(64,32,1,1)

Block(32,32,1,1)

model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())

model:add(nn.Dropout())

Block(32,32,5,5,1,1,2,2)

Block(32,64,1,1)

Block(64,32,1,1)

model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())

model:add(nn.Dropout())

Block(32,32,3,3,1,1,1,1)

Block(32,32,1,1)

Block(32,10,1,1)

model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())

model:add(nn.View(10))



for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do

  v.weight:normal(0,0.05)

  v.bias:zero()

end

model = model:cuda()
-- criterion = nn.ClassNLLCriterion():cuda()
criterion = nn.CrossEntropyCriterion():cuda()

w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 64
local optimState = { 
    learningRate = 1,
    momentum = 0.9,
    weightDecay = 0.0005
}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)--:cuda()
        local yt = labels:narrow(1, i, batchSize)--:cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
                return err, dE_dw
            end

            optim.sgd(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

function plotLoss(trainLoss, testLoss, title)
	require 'gnuplot'
	local range = torch.range(1, trainLoss:size(1))
	gnuplot.pngfigure('testVsTrainLoss.png')
	gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()
end

---------------------------------------------------------------------

epochs = 250
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

for e = 1, epochs do
     
    if e%25 == 0 then optimState.learningRate = optimState.learningRate/2 end

    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e==1 or e==5 or e==10 or e==15 or e==20 or e==25 or e==30 or e==35 or e==100 or e==150 or e==200 or e>=230 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')

plotLoss(trainLoss, testLoss, 'Classification Loss')

torch.save('model.t7',model)

--  ****************************************************************
--  Network predictions
--  ****************************************************************


model:evaluate()   --turn off dropout

print(classes[testLabels[10]])
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
for i=1,predicted:size(2) do
    print(classes[i],predicted[1][i])
end


