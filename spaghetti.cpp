extern "C" {
#include<torch/luaT.h>
#include<torch/TH/TH.h>
}
#include<cmath>
#include <cassert>
#include<vector>
#include<algorithm>
#include<iostream>
using namespace std;

#ifdef assert
#undef assert
#endif
#define assert(x) {if (!x) {std::cerr << "Assertion failed file " << __FILE__ << ", line " << __LINE__ << std::endl; exit(0);}}

typedef THDoubleTensor Tensor;
#define ID_TENSOR_STRING "torch.DoubleTensor"
#define Tensor_(a) THDoubleTensor_##a
typedef double Real;
typedef double accreal;

static int Spaghetti_updateOutput(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor* input        = (Tensor*      )luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  Tensor* weights      = (Tensor*      )luaT_checkudata(L, 4, idreal);
  Tensor* output       = (Tensor*      )luaT_checkudata(L, 5, idreal);
  THLongTensor* chunks = (THLongTensor*)luaT_checkudata(L, 6, idlong);

  long  nCon= conSrc->size[0];
  long  css = conSrc->stride[0];
  long  cds = conDst->stride[0];
  long  ws  = weights->stride[0];
  Real* ip  = Tensor_(data)(input);
  long* csp = THLongTensor_data(conSrc);
  long* cdp = THLongTensor_data(conDst);
  Real* wp  = Tensor_(data)(weights);
  Real* op  = Tensor_(data)(output);
  
  long* cp      = THLongTensor_data(chunks);
  long* cs      = chunks->stride;
  long nchunks  = chunks->size[1];
  long nBigChunks = chunks->size[0];

  assert((css == 1) && (cds == 1));
  Tensor_(zero)(output);
  
  int i, j, k;
  if (ws == 1) {
#pragma omp parallel for private(i,j)
    for (k = 0; k < nBigChunks; ++k) {
      for (j = 0; j < nchunks-1; ++j) {
	for (i = cp[k*cs[0]+j*cs[1]]; i < cp[k*cs[0]+(j+1)*cs[1]]; ++i)
	  op[cdp[i]] += wp[i] * ip[csp[i]];
      }
    }
  } else {
#pragma omp parallel for private(i,j)
    for (k = 0; k < nBigChunks; ++k) {
      for (j = 0; j < nchunks-1; ++j) {
	for (i = cp[k*cs[0]+j*cs[1]]; i < cp[k*cs[0]+(j+1)*cs[1]]; ++i)
	  op[cdp[i]] += wp[i*ws] * ip[csp[i]];
      }
    }
  }

  return 0;
}

static int Spaghetti_accGradParameters(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor* input        = (Tensor*      )luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  Tensor* gradOutput   = (Tensor*      )luaT_checkudata(L, 4, idreal);
  const Real    scale  =                lua_tonumber   (L, 5);
  Tensor* gradWeight   = (Tensor*      )luaT_checkudata(L, 6, idreal);

  long  nCon= conSrc->size[0];
  long  css = conSrc->stride[0];
  long  cds = conDst->stride[0];
  long  gws = gradWeight->stride[0];
  Real* ip  = Tensor_(data)(input);
  long* csp = THLongTensor_data(conSrc);
  long* cdp = THLongTensor_data(conDst);
  Real* gop = Tensor_(data)(gradOutput);
  Real* gwp = Tensor_(data)(gradWeight);

  assert((css == 1) && (cds == 1));
  
  int i;
  if ((scale == 1) && (gws == 1)) {
#pragma omp parallel for private(i)
  for (i = 0; i < nCon; ++i)
    gwp[i] += ip[csp[i]] * gop[cdp[i]];
  } else {
#pragma omp parallel for private(i)
  for (i = 0; i < nCon; ++i)
    gwp[i*gws] += scale * ip[csp[i]] * gop[cdp[i]];
  }

  return 0;
}

static const struct luaL_reg libspaghetti[] = {
  {"spaghetti_updateOutput", Spaghetti_updateOutput},
  {"spaghetti_accGradParameters", Spaghetti_accGradParameters},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libspaghetti(lua_State *L) {
  luaL_openlib(L, "libspaghetti", libspaghetti, 0);
  return 1;
}
