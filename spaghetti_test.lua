require 'spaghetti'
require 'sys'

local function TH2table(t)
   local out = {}
   assert(t:nDimension() == 1)
   for i = 1,t:size(1) do
      out[i] = t[i]
   end
   return out
end

local SpaghettiControl, parent = torch.class("nn.SpaghettiControl", "nn.Module")

function SpaghettiControl:__init(conSrc, conDst, dimDst, do_not_reset)
   parent.__init(self)
   assert(conSrc:size(1) == conDst:size(1))
   self.nCon = conSrc:size(1)
   self.conSrc = conSrc:long()
   self.conDst = conDst:long()
   self.weight = torch.Tensor(self.nCon)
   self.gradWeight = torch.Tensor():resizeAs(self.weight)
   self.output = torch.Tensor(dimDst)
   self.gradInput = torch.Tensor()
   if not do_not_reset then
      self:reset()
   end
end

function SpaghettiControl:reset(stdv)
   stdv = stdv or 1
   self.weight:apply(function() return torch.uniform(-stdv, stdv) end)
end

function SpaghettiControl:updateOutput(input)
   self.output:zero()
   for i = 1,self.nCon do
      self.output[TH2table(self.conDst[i])] = self.output[TH2table(self.conDst[i])] + self.weight[i] * input[TH2table(self.conSrc[i])]
   end
   return self.output
end

function SpaghettiControl:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   for i = 1,self.nCon do
      self.gradInput[TH2table(self.conSrc[i])] = self.gradInput[TH2table(self.conSrc[i])] + self.weight[i] * gradOutput[TH2table(self.conDst[i])]
   end
   return self.gradInput
end

function SpaghettiControl:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i = 1,self.nCon do
      self.gradWeight[i] = self.gradWeight[i] + scale*input[TH2table(self.conSrc[i])]*gradOutput[TH2table(self.conDst[i])]
   end
end

function SpaghettiControl:decayParameters(decay)
   self.weight:add(-decay, self.weight)
end

function Spaghetti_testme()
   local dimSrc = torch.LongStorage{100}
   local dimDst = torch.LongStorage{100}
   local nCon = torch.random(400)
   local conSrc = torch.Tensor(nCon, 1)
   local conDst = torch.Tensor(nCon, 1)
   for i = 1,nCon do
      conSrc[i][1] = torch.random(dimSrc[1])
      conDst[i][1] = torch.random(dimDst[1])
   end
   --test contiguous
   local spa = nn.Spaghetti(conSrc, conDst, dimDst, true)
   local cspa = nn.SpaghettiControl(conSrc, conDst, dimDst, true)
   local w = torch.randn(nCon)
   spa:copyWeights(w)
   cspa.weight:copy(w)
   for t = 1,100 do
      local input = torch.randn(dimSrc)
      local out1 = spa:forward(input)
      local out2 = cspa:forward(input)
      local delta = (out1-out2):abs():max()
      assert(delta < 1e-3)
      spa.gradWeight:zero()
      cspa.gradWeight:zero()
      local go = torch.randn(out1:size())
      local bp1 = spa:backward(input, go)
      local bp2 = cspa:backward(input, go)
      delta = (bp1-bp2):abs():max()
      assert(delta < 1e-3)
      local gw = torch.Tensor(cspa.gradWeight:size(1))
      for i = 1,gw:size(1) do
	 gw[i] = cspa.gradWeight[spa.order[i] ]
      end
      delta = (spa.gradWeight-gw):abs():max()
      assert(delta < 1e-3)      
   end
   --test non contiguous
   for t = 1,100 do
      local input = torch.randn(dimSrc[1],2)
      input = input:narrow(2,1,1):squeeze()
      local out1 = spa:forward(input)
      local out2 = cspa:forward(input)
      local delta = (out1-out2):abs():max()
      assert(delta < 1e-3)
      spa.gradWeight:zero()
      cspa.gradWeight:zero()
      local go = torch.randn(out1:size())
      local bp1 = spa:backward(input, go)
      local bp2 = cspa:backward(input, go)
      delta = (bp1-bp2):abs():max()
      assert(delta < 1e-3)
      local gw = torch.Tensor(cspa.gradWeight:size(1))
      for i = 1,gw:size(1) do
	 gw[i] = cspa.gradWeight[spa.order[i] ]
      end
      delta = (spa.gradWeight-gw):abs():max()
      assert(delta < 1e-3)      
   end
   --test multiple dimensions
   local dimSrc = torch.LongStorage{42,23,2}
   local dimDst = torch.LongStorage{12,31,4,3}
   local nCon = torch.random(100)
   local conSrc = torch.Tensor(nCon, dimSrc:size(1))
   local conDst = torch.Tensor(nCon, dimDst:size(2))
   for i = 1,nCon do
      for j = 1,dimSrc:size(1) do
	 conSrc[i][j] = torch.random(dimSrc[j])
      end
      for j = 1,dimDst:size(1) do
	 conDst[i][j] = torch.random(dimDst[j])
      end
   end
   local spa = nn.Spaghetti(conSrc, conDst, dimDst, true)
   local cspa = nn.SpaghettiControl(conSrc, conDst, dimDst, true)
   local w = torch.randn(nCon)
   spa:copyWeights(w)
   cspa.weight:copy(w)
   for t = 1,100 do
      local input = torch.randn(dimSrc)
      local out1 = spa:forward(input)
      local out2 = cspa:forward(input)
      local delta = (out1-out2):abs():max()
      assert(delta < 1e-3)
      spa.gradWeight:zero()
      cspa.gradWeight:zero()
      local go = torch.randn(out1:size())
      local bp1 = spa:backward(input, go)
      local bp2 = cspa:backward(input, go)
      delta = (bp1-bp2):abs():max()
      assert(delta < 1e-3)
      local gw = torch.Tensor(cspa.gradWeight:size(1))
      for i = 1,gw:size(1) do
	 gw[i] = cspa.gradWeight[spa.order[i] ]
      end
      delta = (spa.gradWeight-gw):abs():max()
      assert(delta < 1e-3)
   end
end

function Spaghetti_timeme(N)
   print(N)
   local dimSrc = torch.LongStorage{N/4}
   local dimDst = torch.LongStorage{N/4}
   local nCon = torch.random(N)
   local conSrc = torch.Tensor(nCon, 1)
   local conDst = torch.Tensor(nCon, 1)
   for i = 1,nCon do
      conSrc[i][1] = torch.random(dimSrc[1])
      conDst[i][1] = torch.random(dimDst[1])
   end
   
   local spa = nn.Spaghetti(conSrc, conDst, dimDst, true)
   local w = torch.randn(nCon)
   local input = torch.randn(dimSrc)
   spa.gradWeight:zero()
   local go = torch.randn(dimDst)
   local tim = torch.Timer()
   local t1 = 0
   local t2 = 0
   local T = 25
   for t = 1,T do
      tim:reset()
      local out1 = spa:forward(input)
      t1 = t1 + tim:time()['real']
      tim:reset()
      local bp1 = spa:backward(input, go)
      t2 = t2 + tim:time()['real']
   end
   print("forward  " .. t1/(T*N))
   print("backward " .. t2/(T*N))
end

torch.setnumthreads(4)

local N = 10
for i = 1,6 do
   Spaghetti_timeme(N)
   N = N*10
end