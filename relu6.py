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
		x = x[0]
		return utils.force_array(numpy.minimum(numpy.maximum(0, x), 6), x.dtype),

	def backward_cpu(self, x, gy):
		x = x[0]
		return utils.force_array(gy[0] * (0 < x) * (x < 6), x.dtype),

	def forward_gpu(self, x):
		return cuda.elementwise(
			'T x', 'T y', 'y = min(max(x, (T)0), (T)6)',
			'clipped_relu_fwd')(x[0]),

	def backward_gpu(self, x, gy):
		gx = cuda.elementwise(
			'T x, T gy', 'T gx',
			'gx = ((x > 0) & (x < 6))? gy : (T)0',
			'clipped_relu_bwd')(x[0], gy[0])
		return gx,


def relu6(x):
	return ReLU6()(x)