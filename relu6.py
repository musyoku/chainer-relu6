import numpy
import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
	cudnn = cuda.cudnn
	libcudnn = cudnn.cudnn
	_cudnn_version = libcudnn.getVersion()
	_mode = libcudnn.CUDNN_ACTIVATION_RELU


class ReLU6(function.Function):
	def check_type_forward(self, in_types):
		type_check.expect(
			in_types.size() == 1,
			in_types[0].dtype.kind == 'f',
		)

	def forward_cpu(self, x):
		self.retain_inputs(())
		self.retain_outputs((0,))
		return utils.force_array(numpy.minimum(numpy.maximum(x[0], 0, dtype=x[0].dtype), 6, dtype=x[0].dtype)),

	def forward_gpu(self, x):
		self.retain_inputs(())
		self._use_cudnn = False
		y = cuda.cupy.minimum(cuda.cupy.maximum(x[0], 0), 6)
		self.retain_outputs((0,))
		return y,

	def backward_cpu(self, x, gy):
		y = self.output_data[0]
		return utils.force_array(gy[0] * (0 < y) * (y < 6)),

	def backward_gpu(self, x, gy):
		y = self.output_data[0]
		gx = cuda.elementwise(
			'T y, T gy', 'T gx',
			'gx = (0 < y && y < 6) ? gy : (T)0',
			'relu_bwd')(y, gy[0])
		return gx,


def relu6(x):
	return ReLU6()(x)