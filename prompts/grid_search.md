我现在的硬件架构(Chip)是这样的:
-  整体上看是一个中心化的架构, 有用一个中心的 IO die, 和 c 个 compute die,  每个 compute die 通过独立的链路与 IO die 相连. 这个架构用来做矩阵运算
-  每个 Compute Die 拥有一个独立的 Memory, 拥有 4 个参数, 分别是 与 IOdie 的输入/输出带宽, 自身 Memory 的带宽, 自身的算力. 
-  ComputeDie 输入, 运算, 输出是能够流水运算的
-  IO Die 每次给 Compute Die 发送输入数据, 多个 ComputeDie 并行运算, 最后 IO Die 收集输出结果, IO Die 上拥有 reduce 模块, 可以对 Compute Die 的输出结果按需要进行累加. 

我现在有一个任务, 我有一个大的矩阵, 矩阵的尺寸是 M x N,  我需要将这个矩阵拆分为多个 tile, 然后将 tile 映射到多个 Compute Die 上, 然后以 执行时间最长的 Compute Die 的执行时间作为最终的延迟. 

我将 Matrix, ComputeDie,Chip, Tile, Mapping 的结构定义放到了 @matrixmachine/core/description.py  文件中.  如果有一个 Mapping, 可以根据 @matrixmachine/core/sim_engine.py 中的函数, 仿真出其运行时间. @matrixmachine/core/utils.py  中则有算力利用率计算函数, 我们追求最高的算力利用率

现在要求你首先自己想一套算法, 来实现一个最优的 mapping, 达成最短的时间, 也就是最高的算力利用率, 你要思考的努力一些, 相关的代码放到 strategy/gpt_mapping.py 下, 

然后要求你实现一个我设计的复杂 DSE 搜索过程, 搜索出一个最优的 Mapping, 相关代码放到 strategy/gpt_grid_mapping.py 下.

这个搜索的过程是这样的, 首先矩阵可以按照 n_splits_row 和 n_splits_col  切几刀, n_splits_col  和 n_splits_row 的范围 从 1 到 16 进行搜索, 然后每次切完会得到多个 tile  (可以上取整), 然后看 Compute die 的数量按照 round robin 的方式进行分配, 可能会分配多轮, 类似 GPU 编程中的 Wave 概念, 最后还会剩下一些尾巴块, 这时对于这些尾巴块, 我们看看其能够构成什么新的矩阵, 例如 4 个尾巴块可以按照 1x4, 4x1,2x2 的形式构成新的矩阵 (要检查一下尾巴组成的形状是否存在, 不能比原来的矩阵大), 如果尾巴不能拼成单个矩阵就拼成 2 个矩阵, 然后将尾巴作为一个子问题, 递归的方式解决. 目前先将递归的层级设置为两层. 你可以遍历也可以用其他的 DSE 来算法来找出这种表达下的最佳映射, 告诉我 tile 有哪些, 都分配给了哪些 ComputeDie. 

上面这两种算法都实现结束后, 写两个文件分别进行测试, 可以参考 main.py 中的软硬件配置. 