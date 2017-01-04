require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'


function TestFunction (model)

    local trainset = torch.load('cifar.torch/cifar10-train.t7')
    local testset = torch.load('cifar.torch/cifar10-test.t7')

    local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

    local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
    local trainLabels = trainset.label:float():add(1)
    local testData = testset.data:float()
    local testLabels = testset.label:float():add(1)

    --We'll start by normalizing our data
    local mean = {}  -- store the mean, to normalize the test set in the future
    local stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
        trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

--  Normalize test set using same values

    for i=1,3 do -- over each image channel
        testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
        testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end;

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
          end
        end
        self.output:set(input:cuda())
        return self.output
      end
    end


criterion = nn.CrossEntropyCriterion():cuda()

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
        m:training()
    else
        m:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)--:cuda()
        local yt = labels:narrow(1, i, batchSize)--:cuda()
        local y = m:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                m:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                m:backward(x, dE_dy) -- backpropagation
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
    

    m = torch.load(model)


--our original
    --local conf = optim.ConfusionMatrix(torch.range(0,9):totable())
    --local x = testData:narrow(1,1,testData:size(1))--:cuda()
    local testLoss
	local testError
	local confusion
    testData = testData:cuda()
    testLabels = testLabels:cuda()
    --local y = m:forward(testData)
    -- up: it was testData
	testLoss, testError, confusion = forwardNet(testData, testLabels, false)
    --conf:batchAdd(y, testLabels)

    --conf:updateValids()
    --    local err = 1 - conf.totalValid
     return testError



--our new
--[[
	local conf = optim.ConfusionMatrix(10)
	local bs = 64
    --local x = testData:narrow(1,1,testData:size(1))--:cuda()
    
    testData = testData:cuda()
    testLabels = testLabels:cuda()

	for i=1,testData:size(1),bs do
		local outputs = m:forward(testData:narrow(1,i,bs):cuda())
		conf:batchAdd(outputs, testLabels:narrow(1,i,bs):cuda())
	end

    --local y = m:forward(x)
    -- up: it was testData
    --conf:batchAdd(y, testLabels)

    conf:updateValids()
        local err = 1 - conf.totalValid
        return err
		]]


--her's	
--[[confusion = optim.ConfusionMatrix(10)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 64
  for i=1,testData.data:size(1),bs do
    local outputs = model:forward(testData.data:narrow(1,i,bs):cuda())
    confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs):cuda())
  end

  confusion:updateValids()
  --print('Test accuracy:', confusion.totalValid * 100..'%')
  --print('Test error:',(1- confusion.totalValid) * 100..'%')
]]

end


r = TestFunction('model.t7')
print (r)






  
