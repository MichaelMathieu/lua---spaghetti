require 'nn'
require 'math'
require 'libspaghetti'

local Spaghetti, parent = torch.class('nn.Spaghetti', 'nn.Module')

-- Note: NEVER change conSrc or conDst once the module is created,
-- due to the optimizations
function Spaghetti:__init(conSrc, conDst, dimDst, do_not_reset)
   parent.__init(self)
   assert(dimDst:size(1) == conDst:size(2))
   self.nCon = conSrc:size(1)
   self.conSrc = conSrc:long()
   self.conDst = conDst:long()
   self.weight = torch.Tensor(self.nCon)
   self.gradWeight = torch.Tensor():resizeAs(self.weight)
   self.output = torch.Tensor(dimDst)
   self.gradInput = torch.Tensor()

   if libspaghetti.spaghetti_blas() == 0 then
      self.conSrcC = torch.LongTensor(self.nCon) --these 4 arrays are 0 based
      self.conDstC = torch.LongTensor(self.nCon)
      self.conGOC = torch.LongTensor(self.nCon)
      self.conGIC = torch.LongTensor(self.nCon)
      self.currentSrcStride = torch.LongStorage{}
      self.currentDstStride = torch.LongStorage{}
      self.currentGOStride = torch.LongStorage{}
      self.currentGIStride = torch.LongStorage{}
      
      local dummyStrideSrc = torch.Tensor(self.conSrc:max(1):storage()):stride()
      local dummyStrideDst = torch.Tensor(self.conDst:max(1):storage()):stride()
      self:recomputeContiguous(self.conSrc, self.conSrcC,
			       dummyStrideSrc, self.currentSrcStride)
      self:recomputeContiguous(self.conDst, self.conDstC,
			       dummyStrideDst, self.currentDstStride)
      local sortedSrc, orderSrc = self.conSrcC:sort()
      local n = sortedSrc:size(1)
      self.nChunks = 4
      self.nChunks2 = 4
      local cuts = {0}
      for i = 1,(self.nChunks-1) do
	 local cut0 = math.min(math.floor(n/self.nChunks)*i,n-1)
	 local cut1 = cut0 + 1
	 while (cut1 < n) and (sortedSrc[cut1] == sortedSrc[cut0]) do
	    cut1 = cut1 + 1
	 end
	 cut1 = cut1 - 1
	 if (cut1 ~= cuts[#cuts]) and (cut1 < n-1) then
	    table.insert(cuts, cut1)
	 end
      end
      table.insert(cuts, n)
      self.chunks = cuts
      self.chunksB = {}
      self.order = torch.LongTensor(n)
      for iChunk = 1,(#cuts-1) do
	 local newchunk = {}
	 local i0 = cuts[iChunk]
	 local i1 = cuts[iChunk+1]
	 local n2 = i1-i0
	 local tosort = torch.Tensor(n2)
	 for i = 0,(n2-1) do
	    tosort[i+1] = self.conDstC[orderSrc[i0+i+1]]
	 end
	 local sortedDst, orderDst = tosort:sort()
	 for i = 0,(n2-1) do
	    self.order[i0+i+1] = orderSrc[i0+orderDst[i+1]]
	 end
	 table.insert(newchunk, i0)
	 for i = 1,(self.nChunks2-1) do
	    local cut0 = math.min(math.floor(n2/self.nChunks2)*i,n2-1)
	    if cut0 > 0 then
	       local cut1 = cut0 + 1
	       while (cut1 < n2) and (sortedDst[cut1] == sortedDst[cut0]) do
		  cut1 = cut1 + 1
	       end
	       cut1 = cut1 - 1
	       if (cut1 ~= cuts[#cuts]) and (cut1 < n2-1) then
		  table.insert(newchunk, i0+cut1)
	       end
	    end
	 end
	 while #newchunk <= self.nChunks2 do
	    table.insert(newchunk, i1)
	 end
	 table.insert(self.chunksB, newchunk)
      end
      table.insert(self.chunks, n)
      local conSrcTmp = torch.LongTensor(self.conSrc:size(1), self.conSrc:size(2))
      local conDstTmp = torch.LongTensor(self.conDst:size(1), self.conDst:size(2))
      for i = 1,n do
	 conSrcTmp[i] = self.conSrc[self.order[i] ]
	 conDstTmp[i] = self.conDst[self.order[i] ]
	 --conSrcTmp[self.order[i] ] = self.conSrc[i]
	 --conDstTmp[self.order[i] ] = self.conDst[i]
      end
      self.conSrc = conSrcTmp
      self.conDst = conDstTmp
      self.currentSrcStride = torch.LongStorage{}
      self.currentDstStride = torch.LongStorage{}
      self.chunks = torch.LongTensor(self.chunks)
      self.chunks:resize(1,self.chunks:size(1))
      self.chunksB = torch.LongTensor(self.chunksB)
   else
      self.conSrc:add(-1) -- Careful : 0-based, only if blas
      self.conDst:add(-1)
   end
   
   if not do_not_reset then
      self:reset()
   end
end

function Spaghetti:copyWeights(w)
   if libspaghetti.spaghetti_blas() == 0 then
      for i = 1,w:size(1) do
	 self.weight[i] = w[self.order[i] ]
	 --self.weight[self.order[i] ] = w[i]
      end
   else
      self.weight:copy(w)
   end
end

function Spaghetti:recomputeContiguous(nonContig, contig, stride, curStride)
   local function strideEq(a,b)
      if a:size(1) ~= b:size(1) then
	 return false
      end
      return torch.Tensor(a:totable()):eq(torch.Tensor(b:totable())):sum() == a:size(1)
   end
   if not strideEq(stride, curStride) then
      contig:zero()
      for i = 1,nonContig:size(2) do
	 contig:add(stride[i], nonContig[{{},i}]-1)
      end
      curStride:resize(stride:size()):copy(stride)
   end
end

function Spaghetti:reset(stdv)
   stdv = stdv or 1
   self.weight:apply(function() return torch.uniform(-stdv, stdv) end)
end

function Spaghetti:updateOutput(input)
   assert(input:nDimension() == self.conSrc:size(2))
   if libspaghetti.spaghetti_blas() == 0 then
      self:recomputeContiguous(self.conSrc, self.conSrcC,
			       input:stride(), self.currentSrcStride)
      self:recomputeContiguous(self.conDst, self.conDstC,
			       self.output:stride(), self.currentDstStride)
   end
   libspaghetti.spaghetti_updateOutput(input, self.conSrc, self.conDst,
				       self.weight, self.output, self.chunks)
   return self.output
end

function Spaghetti:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   if libspaghetti.spaghetti_blas() == 0 then
      self:recomputeContiguous(self.conDst, self.conGOC,
			       gradOutput:stride(), self.currentGOStride)
      self:recomputeContiguous(self.conSrc, self.conGIC,
			       self.gradInput:stride(), self.currentGIStride)
   end
   libspaghetti.spaghetti_updateOutput(gradOutput, self.conGOC, self.conGIC,
				     self.weight, self.gradInput, self.chunksB)
   return self.gradInput
end

function Spaghetti:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if libspaghetti.spaghetti_blas() == 0 then
      self:recomputeContiguous(self.conSrc, self.conSrcC,
			       input:stride(), self.currentSrcStride)
      self:recomputeContiguous(self.conDst, self.conDstC,
			       self.output:stride(), self.currentDstStride)
   end
   libspaghetti.spaghetti_accGradParameters(input, self.conSrcC, self.conDstC,
					    gradOutput, scale, self.gradWeight)
end

function Spaghetti:decayParameters(decay)
   self.weight:add(-decay, self.weight)
end