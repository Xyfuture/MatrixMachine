我要实现一个Mapping DSE 的算法, 通过搜索的方式找出最佳的 Mapping.

算法是一个递归的过程, 通过不断处理子问题来获得最佳的执行执行时间.

算法的流程是这样的

输入: 矩阵, 硬件配置, num_split_row 的候选项(暂时设置为为 1 到 8) num_split_col 的候选项(暂时设置为 1 到 8)

输出: 针对这个矩阵的一个 mapping, 用 description.py 中的数据结构进行表示. 

- 对于一个输入的矩阵, 我们遍历 num_split_row_candidate 和 num_split_col_candidate, 选择一个具体的 num_split_row 和 num_split_col
  - 用这个指定的 num_split_row 和 num_split_col 对矩阵进行拆分, 将矩阵拆分为 num_split_row x num_split_col 个 tile, 对于边缘上的 tile 进行上取整操作
  - 按照 round-robin 的方式, 将 tile 分配给 ComputeDie, 每个 ComputeDie 可能分到多个 tile, 如果剩下的 Tile 个数不能够分配满所有的 ComputeDie, 则停下, 剩下的几个 tile 称之为尾块 (类似于整除和余数)
  - 对于尾块, 我们选择作为子问题进行处理, 递归调用前面的算法.
    - 尾块能构成一个什么样的矩阵是不确定的, 这里要分情况进行讨论
      - 可以构成单个矩阵, 例如 4*4 的网格中, 分配结束后剩下 4 个 tile, 那么这时候 能构成 1x4, 4 x1, 2x2 这 3 种合法的形式, 要遍历这三种子矩阵.
      - 不能构成单个矩阵, 必须要拆分成 2 个子矩阵, 例如 还是 4x4 的网格, 剩下 5 个 tile, 如果构成单个 1x5 或者 5x1 的矩阵都是非法的, 因为原始的网格边长最大是 4, 这个时候只能拆为 (1x1, 4x1) 或者(1x2,1x3)这两种合法的 2 个子矩阵的形式了, 那么要同时处理这两个子矩阵
- 每次都返回遍历后时间最优的 mapping

你要用力思考一下这个算法里有没有什么问题, 如果有问题直接是修复它, 或者你认为有什么优化点, 也加进去. 将这个算法实现出来, 放到 strategy/codex_grid_search.py 中, 然后写一份你对这个算法的理解放到 prompt/codex_grid_search.md 中. 

最后参考 main.py 写一个样例调用这个算法, 跑一个 4k x 4k 的矩阵, 输出算力利用率的信息 