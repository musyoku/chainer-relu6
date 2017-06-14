import unittest
import mock
import numpy
import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from relu6 import ReLU6, relu6

@testing.parameterize(*testing.product({
	'shape': [(3, 2), ()],
	'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestReLU(unittest.TestCase):

	def setUp(self):
		# Avoid unstability of numerical grad
		self.x = numpy.random.uniform(-10, 10, self.shape).astype(self.dtype)
		for i in numpy.ndindex(self.shape):
			if -0.1 < self.x[i] < 0.1:
				self.x[i] = 0.5
		self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
		self.check_backward_options = {}
		if self.dtype == numpy.float16:
			self.check_backward_options = {'dtype': numpy.float64}

	def check_forward(self, x_data, use_cudnn='always'):
		x = chainer.Variable(x_data)
		with chainer.using_config('use_cudnn', use_cudnn):
			y = relu6(x)
		self.assertEqual(y.data.dtype, self.dtype)

		expected = self.x.copy()
		for i in numpy.ndindex(self.x.shape):
			if self.x[i] < 0:
				expected[i] = 0
			if self.x[i] > 6:
				expected[i] = 6

		testing.assert_allclose(
			expected, y.data)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.x)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.check_forward(cuda.to_gpu(self.x))

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu_no_cudnn(self):
		self.check_forward(cuda.to_gpu(self.x), 'never')

	def check_backward(self, x_data, y_grad, use_cudnn='always'):
		with chainer.using_config('use_cudnn', use_cudnn):
			gradient_check.check_backward(
				ReLU6(), x_data, y_grad,
				**self.check_backward_options)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu_non_contiguous(self):
		self.check_backward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
							cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

	@attr.gpu
	@condition.retry(3)
	def test_backward_cpu_no_cudnn(self):
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')


testing.run_module(__name__, __file__)