def A_star_main(cost, heuristic, start, goals, path, visited):
    path.append(start)
    fn = [[heuristic[start], path]]
    while len(fn) > 0:
        curr_cost, curr_path = fn.pop(0)
        curr_pointer = curr_path[-1]
        curr_cost -= heuristic[curr_pointer]
        if curr_pointer in goals:
            return curr_path
        visited.append(curr_pointer)
        neighbors = [i for i in range(len(cost[0]))
                     if cost[curr_pointer][i] not in [0, -1]]
        for i in neighbors:
            new_curr_path = curr_path + [i]
            new_path_cost = curr_cost + cost[curr_pointer][i] + heuristic[i]
            if i not in visited and new_curr_path not in [i[1] for i in fn]:
                fn.append((new_path_cost, new_curr_path))
                fn = sorted(fn, key=lambda x: (x[0], x[1]))
            elif new_curr_path in [i[1] for i in fn]:
                for index in range(len(fn)):
                    if fn[index][1] == path:
                        break
                fn[index][0] = min(fn[index][0], new_path_cost)
                fn = sorted(fn, key=lambda x: (x[0], x[1]))
    return -1


def A_star_Traversal(cost, heuristic, start_point, goals):
    path = []
    visited = []
    path = A_star_main(cost, heuristic, start_point, goals, path, visited)
    return path


def DFS_Traversal(cost, start_point, goals):
    path = []
    l = len(cost[0])
    visited = [0]*l
    dfs_main(cost, start_point, goals, path, visited)
    return path


def dfs_main(cost, start, goals, path, visited):
    visited[start] = 1
    path.append(start)
    if(start not in goals):
        temp = cost[start]
        for i in range(len(temp)):
            if((visited[i] == 0) and (temp[i] > 0)):
                result = dfs_main(cost, i, goals, path, visited)
                if result == -1:
                    return -1
                path.pop()
    else:
        return -1
