from std.memory import UnsafePointer

# grid, result are device pointers
@export
fn solve(grid: UnsafePointer[Int32, MutExternalOrigin], result: UnsafePointer[Int32, MutExternalOrigin], rows: Int32, cols: Int32,
         start_row: Int32, start_col: Int32, end_row: Int32, end_col: Int32) raises:
    pass
