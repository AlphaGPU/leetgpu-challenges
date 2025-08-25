from memory import UnsafePointer

# grid, result are device pointers
@export
def solve(grid: UnsafePointer[Int32], result: UnsafePointer[Int32], rows: Int32, cols: Int32,
         start_row: Int32, start_col: Int32, end_row: Int32, end_col: Int32):
    pass