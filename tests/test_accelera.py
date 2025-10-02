"""
Unit tests for Accelera framework.
"""

import unittest
import torch
import numpy as np
import accelera as acc
from accelera.memory_manager import MemoryManager
from accelera.chunking import RowChunking, ChunkIterator
from accelera.matrix import Matrix


class TestMemoryManager(unittest.TestCase):
    """Test memory management functionality."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.memory_manager = MemoryManager()
    
    def test_memory_info(self):
        """Test memory information retrieval."""
        info = self.memory_manager.get_memory_usage()
        
        # Check that all required keys are present
        required_keys = [
            'gpu_allocated_gb', 'gpu_cached_gb', 'gpu_available_gb', 
            'gpu_total_gb', 'gpu_utilization', 'cpu_available_gb', 
            'cpu_total_gb', 'cpu_utilization'
        ]
        
        for key in required_keys:
            self.assertIn(key, info)
            self.assertIsInstance(info[key], (int, float))
    
    def test_tensor_size_estimation(self):
        """Test tensor size estimation."""
        shape = (1000, 1000)
        size = self.memory_manager.estimate_tensor_size(shape, torch.float32)
        expected_size = 1000 * 1000 * 4  # float32 = 4 bytes
        self.assertEqual(size, expected_size)
    
    def test_gpu_transfers(self):
        """Test CPU-GPU tensor transfers."""
        # Create a small tensor
        tensor = torch.randn(100, 100)
        
        # Move to GPU
        gpu_tensor = self.memory_manager.move_to_gpu(tensor)
        self.assertEqual(gpu_tensor.device, self.memory_manager.device)
        
        # Move back to CPU
        cpu_tensor = self.memory_manager.move_to_cpu(gpu_tensor)
        self.assertEqual(cpu_tensor.device, torch.device('cpu'))
        
        # Check values are preserved
        self.assertTrue(torch.allclose(tensor, cpu_tensor))


class TestChunking(unittest.TestCase):
    """Test chunking strategies."""
    
    def test_row_chunking(self):
        """Test row chunking strategy."""
        strategy = RowChunking()
        shape = (1000, 500)
        chunk_size = 250
        
        chunks = strategy.get_chunks(shape, chunk_size)
        
        # Should have 4 chunks (1000 / 250)
        self.assertEqual(len(chunks), 4)
        
        # Check chunk slices
        expected_chunks = [
            (slice(0, 250), slice(None)),
            (slice(250, 500), slice(None)),
            (slice(500, 750), slice(None)),
            (slice(750, 1000), slice(None))
        ]
        
        self.assertEqual(chunks, expected_chunks)
    
    def test_chunk_iterator(self):
        """Test chunk iterator functionality."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        tensor = torch.randn(1000, 100)
        memory_manager = MemoryManager()
        strategy = RowChunking()
        chunk_size = 250
        
        iterator = ChunkIterator(tensor, strategy, chunk_size, memory_manager, prefetch=False)
        
        # Check iterator length
        self.assertEqual(len(iterator), 4)
        
        # Iterate through chunks
        total_rows = 0
        for i, chunk in enumerate(iterator):
            self.assertIsInstance(chunk, torch.Tensor)
            self.assertEqual(chunk.shape[1], 100)  # Second dimension should be preserved
            total_rows += chunk.shape[0]
        
        # Should cover all rows
        self.assertEqual(total_rows, 1000)


class TestMatrix(unittest.TestCase):
    """Test Matrix wrapper functionality."""
    
    def test_matrix_creation(self):
        """Test matrix creation from different sources."""
        # From numpy
        np_array = np.random.randn(10, 20)
        matrix1 = Matrix(np_array)
        self.assertEqual(matrix1.shape, (10, 20))
        
        # From tensor
        tensor = torch.randn(15, 25)
        matrix2 = Matrix(tensor)
        self.assertEqual(matrix2.shape, (15, 25))
        
        # From list
        list_data = [[1, 2], [3, 4]]
        matrix3 = Matrix(list_data)
        self.assertEqual(matrix3.shape, (2, 2))
    
    def test_matrix_operations(self):
        """Test basic matrix operations."""
        A = Matrix.randn((5, 5))
        B = Matrix.randn((5, 5))
        
        # Addition
        C = A + B
        self.assertEqual(C.shape, (5, 5))
        
        # Element-wise multiplication
        D = A * B
        self.assertEqual(D.shape, (5, 5))
        
        # Matrix multiplication
        E = A @ B
        self.assertEqual(E.shape, (5, 5))
    
    def test_static_constructors(self):
        """Test static constructor methods."""
        # Zeros
        zeros = Matrix.zeros((3, 4))
        self.assertEqual(zeros.shape, (3, 4))
        self.assertTrue(torch.allclose(zeros.tensor(), torch.zeros(3, 4)))
        
        # Ones
        ones = Matrix.ones((2, 3))
        self.assertEqual(ones.shape, (2, 3))
        self.assertTrue(torch.allclose(ones.tensor(), torch.ones(2, 3)))
        
        # Identity
        eye = Matrix.eye(4)
        self.assertEqual(eye.shape, (4, 4))
        self.assertTrue(torch.allclose(eye.tensor(), torch.eye(4)))


class TestMatrixEngine(unittest.TestCase):
    """Test Matrix Engine functionality."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.engine = acc.MatrixEngine(enable_progress=False)  # Disable progress bars for tests
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication with different sizes."""
        # Small matrices (should fit in memory)
        A_small = Matrix.randn((100, 80))
        B_small = Matrix.randn((80, 120))
        
        C_small = self.engine.matmul(A_small, B_small)
        self.assertEqual(C_small.shape, (100, 120))
        
        # Verify correctness with direct computation
        C_direct = A_small @ B_small
        self.assertTrue(torch.allclose(C_small.tensor(), C_direct.tensor(), atol=1e-5))
    
    def test_element_wise_operations(self):
        """Test element-wise operations."""
        A = Matrix.randn((200, 150))
        B = Matrix.randn((200, 150))
        
        # Addition
        C_add = self.engine.add(A, B)
        self.assertEqual(C_add.shape, (200, 150))
        
        # Element-wise multiplication
        C_mul = self.engine.multiply(A, B)
        self.assertEqual(C_mul.shape, (200, 150))
        
        # Verify correctness
        C_add_direct = A + B
        C_mul_direct = A * B
        
        self.assertTrue(torch.allclose(C_add.tensor(), C_add_direct.tensor(), atol=1e-5))
        self.assertTrue(torch.allclose(C_mul.tensor(), C_mul_direct.tensor(), atol=1e-5))
    
    def test_dimension_validation(self):
        """Test that dimension mismatches are caught."""
        A = Matrix.randn((100, 50))
        B = Matrix.randn((60, 80))  # Wrong dimension
        
        with self.assertRaises(ValueError):
            self.engine.matmul(A, B)
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        # Get initial memory usage
        initial_memory = self.engine.get_memory_info()
        
        # Perform operations
        A = Matrix.randn((500, 400))
        B = Matrix.randn((400, 600))
        C = self.engine.matmul(A, B)
        
        # Force cleanup
        self.engine.cleanup()
        
        # Memory usage should be similar to initial (allowing for some variance)
        final_memory = self.engine.get_memory_info()
        memory_diff = abs(final_memory['gpu_utilization'] - initial_memory['gpu_utilization'])
        
        # Allow up to 5% difference due to fragmentation
        self.assertLess(memory_diff, 5.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.engine = acc.MatrixEngine(enable_progress=False)
    
    def test_large_matrix_chain(self):
        """Test chain of matrix operations."""
        # Create matrices
        A = Matrix.randn((500, 400))
        B = Matrix.randn((400, 300))
        C = Matrix.randn((500, 300))
        
        # Chain operations: (A @ B) + C
        intermediate = self.engine.matmul(A, B)
        result = self.engine.add(intermediate, C)
        
        self.assertEqual(result.shape, (500, 300))
        
        # Verify with direct computation (if memory allows)
        try:
            direct_result = (A @ B) + C
            self.assertTrue(torch.allclose(result.tensor(), direct_result.tensor(), atol=1e-4))
        except RuntimeError:
            # Skip verification if direct computation fails due to memory
            pass
    
    def test_mixed_input_types(self):
        """Test engine with different input types."""
        # NumPy arrays
        A_np = np.random.randn(100, 80).astype(np.float32)
        B_np = np.random.randn(80, 120).astype(np.float32)
        
        C_from_np = self.engine.matmul(A_np, B_np)
        self.assertEqual(C_from_np.shape, (100, 120))
        
        # PyTorch tensors
        A_torch = torch.randn(100, 80)
        B_torch = torch.randn(80, 120)
        
        C_from_torch = self.engine.matmul(A_torch, B_torch)
        self.assertEqual(C_from_torch.shape, (100, 120))
        
        # Matrix objects
        A_matrix = Matrix.randn((100, 80))
        B_matrix = Matrix.randn((80, 120))
        
        C_from_matrix = self.engine.matmul(A_matrix, B_matrix)
        self.assertEqual(C_from_matrix.shape, (100, 120))


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    unittest.main()