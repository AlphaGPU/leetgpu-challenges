import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="BFS Shortest Path",
            atol=0,
            rtol=0,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, grid: torch.Tensor, result: torch.Tensor, rows: int, cols: int, 
                       start_row: int, start_col: int, end_row: int, end_col: int):
        """
        Reference implementation that finds shortest path using BFS.
        
        Args:
            grid: Flattened 2D grid of size rows*cols (0=free, 1=obstacle)
            result: Single element tensor to store the result
            rows, cols: Grid dimensions
            start_row, start_col: Starting position
            end_row, end_col: Target position
        """
        assert grid.dtype == torch.int32
        assert result.dtype == torch.int32
        assert grid.shape == (rows * cols,)
        assert result.shape == (1,)
        assert 0 <= start_row < rows and 0 <= start_col < cols
        assert 0 <= end_row < rows and 0 <= end_col < cols
        
        # If start and end are the same
        if start_row == end_row and start_col == end_col:
            result[0] = 0
            return
            
        # Reshape grid for easier indexing
        grid_2d = grid.view(rows, cols)
        
        # BFS implementation
        from collections import deque
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize visited array
        visited = torch.zeros((rows, cols), dtype=torch.bool, device=grid.device)
        
        # BFS queue: (row, col, distance)
        queue = deque([(start_row, start_col, 0)])
        visited[start_row, start_col] = True
        
        while queue:
            row, col, dist = queue.popleft()
            
            # Check if we reached the target
            if row == end_row and col == end_col:
                result[0] = dist
                return
                
            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    # Check if not visited and not obstacle
                    if not visited[new_row, new_col] and grid_2d[new_row, new_col] == 0:
                        visited[new_row, new_col] = True
                        queue.append((new_row, new_col, dist + 1))
        
        # No path found
        result[0] = -1

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "grid": (ctypes.POINTER(ctypes.c_int), "in"),
            "result": (ctypes.POINTER(ctypes.c_int), "out"),
            "rows": (ctypes.c_int, "in"),
            "cols": (ctypes.c_int, "in"),
            "start_row": (ctypes.c_int, "in"),
            "start_col": (ctypes.c_int, "in"),
            "end_row": (ctypes.c_int, "in"),
            "end_col": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_int = torch.int32
        
        # Example from challenge.html
        # Grid: [[0,0,0,0], [1,1,0,1], [0,0,0,0], [0,1,1,0]]
        grid_data = torch.tensor([
            0, 0, 0, 0,  # row 0
            1, 1, 0, 1,  # row 1
            0, 0, 0, 0,  # row 2
            0, 1, 1, 0   # row 3
        ], device="cuda", dtype=dtype_int)
        
        result_data = torch.tensor([0], device="cuda", dtype=dtype_int)
        
        return {
            "grid": grid_data,
            "result": result_data,
            "rows": 4,
            "cols": 4,
            "start_row": 0,
            "start_col": 0,
            "end_row": 3,
            "end_col": 3
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_int = torch.int32
        test_cases = []
        
        # Test case 1: Simple path exists
        test_cases.append({
            "grid": torch.tensor([0, 0, 1, 0, 0, 0], device="cuda", dtype=dtype_int),
            "result": torch.tensor([0], device="cuda", dtype=dtype_int),
            "rows": 2,
            "cols": 3,
            "start_row": 0,
            "start_col": 0,
            "end_row": 1,
            "end_col": 2
        })
        
        # Test case 2: No path (blocked)
        test_cases.append({
            "grid": torch.tensor([0, 1, 0, 1, 0], device="cuda", dtype=dtype_int),
            "result": torch.tensor([0], device="cuda", dtype=dtype_int),
            "rows": 1,
            "cols": 5,
            "start_row": 0,
            "start_col": 0,
            "end_row": 0,
            "end_col": 4
        })
        
        # Test case 3: Same start and end
        test_cases.append({
            "grid": torch.tensor([0, 1, 0, 0], device="cuda", dtype=dtype_int),
            "result": torch.tensor([0], device="cuda", dtype=dtype_int),
            "rows": 2,
            "cols": 2,
            "start_row": 0,
            "start_col": 0,
            "end_row": 0,
            "end_col": 0
        })
        
        # Test case 4: Single cell
        test_cases.append({
            "grid": torch.tensor([0], device="cuda", dtype=dtype_int),
            "result": torch.tensor([0], device="cuda", dtype=dtype_int),
            "rows": 1,
            "cols": 1,
            "start_row": 0,
            "start_col": 0,
            "end_row": 0,
            "end_col": 0
        })
        
        # Test case 5: Larger grid with path
        large_grid = torch.zeros(25, device="cuda", dtype=dtype_int)  # 5x5 grid
        large_grid[6] = 1  # obstacle at (1,1)
        large_grid[7] = 1  # obstacle at (1,2)
        large_grid[8] = 1  # obstacle at (1,3)
        test_cases.append({
            "grid": large_grid,
            "result": torch.tensor([0], device="cuda", dtype=dtype_int),
            "rows": 5,
            "cols": 5,
            "start_row": 0,
            "start_col": 0,
            "end_row": 4,
            "end_col": 4
        })
        
        # Test case 6: Complex maze
        maze_grid = torch.tensor([
            0, 1, 0, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 0, 1, 0,
            1, 1, 0, 0, 0,
            0, 0, 0, 1, 0
        ], device="cuda", dtype=dtype_int)
        test_cases.append({
            "grid": maze_grid,
            "result": torch.tensor([0], device="cuda", dtype=dtype_int),
            "rows": 5,
            "cols": 5,
            "start_row": 0,
            "start_col": 0,
            "end_row": 4,
            "end_col": 4
        })
        
        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype_int = torch.int32
        rows, cols = 500, 500
        
        # Create a large grid with some random obstacles
        torch.manual_seed(42)
        grid = torch.randint(0, 2, (rows * cols,), device="cuda", dtype=dtype_int)
        
        # Ensure start and end are free
        grid[0] = 0  # start at (0,0)
        grid[-1] = 0  # end at (rows-1, cols-1)
        
        # Create some clear paths to avoid always getting -1
        for i in range(0, rows * cols, cols):
            if i + cols - 1 < rows * cols:
                grid[i:i + min(cols, 10)] = 0  # Clear first 10 cells of each row
                
        result = torch.tensor([0], device="cuda", dtype=dtype_int)
        
        return {
            "grid": grid,
            "result": result,
            "rows": rows,
            "cols": cols,
            "start_row": 0,
            "start_col": 0,
            "end_row": rows - 1,
            "end_col": cols - 1
        }