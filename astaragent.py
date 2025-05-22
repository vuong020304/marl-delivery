import numpy as np
from collections import deque
import heapq

class OptimalAStarAgent:
    def __init__(self):
        self.map = None
        self.robots = []  # [(x, y, package_id)]
        self.packages = {}  # {id: (pickup_x, pickup_y, delivery_x, delivery_y, deadline)}
        self.path_cache = {}  # {(start, goal): path}
        self.robot_targets = []  # [package_id/'free']
        self.assigned_packages = set()  # Gói hàng đã được phân công
        self.repeat_count = []  # Đếm số lần robot đứng im
        self.last_actions = []  # Lưu hành động trước đó
        
        # Để phát hiện và xử lý xung đột
        self.moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1), 'S': (0, 0)}
        
    def init_agents(self, state):
        self.map = state['map']
        self.map_size = len(self.map)
        self.robots = [(r[0], r[1], 0) for r in state['robots']]
        self.n_robots = len(self.robots)
        self.robot_targets = ['free'] * self.n_robots
        self.repeat_count = [0] * self.n_robots
        self.last_actions = [('S', '0')] * self.n_robots
    
    def is_valid_position(self, pos):
        """Kiểm tra vị trí có hợp lệ không"""
        x, y = pos
        if x < 1 or x > len(self.map) or y < 1 or y > len(self.map[0]):
            return False
        return self.map[x-1][y-1] == 0  # 0 là ô trống
    
    def get_next_position(self, pos, move):
        """Tính toán vị trí mới sau khi di chuyển"""
        r, c = pos
        dx, dy = self.moves.get(move, (0, 0))
        new_r, new_c = r + dx, c + dy
        if not self.is_valid_position((new_r, new_c)):
            return r, c
        return new_r, new_c
    
    def a_star_search(self, start, goal):
        """Thuật toán A* tìm đường đi ngắn nhất"""
        # Kiểm tra cache
        if (start, goal) in self.path_cache:
            return self.path_cache[(start, goal)]
        
        # Nếu start hoặc goal không hợp lệ
        if not self.is_valid_position(start) or not self.is_valid_position(goal):
            return []
        
        # A* search
        open_list = []
        heapq.heappush(open_list, (0, 0, start, []))  # (f, g, pos, path)
        closed_set = set()
        g_scores = {start: 0}
        
        while open_list:
            _, g, current, path = heapq.heappop(open_list)
            
            if current == goal:
                full_path = self.build_path(path + [current])
                self.path_cache[(start, goal)] = full_path
                return full_path
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Xét 4 hướng di chuyển
            for move, (dx, dy) in self.moves.items():
                if move == 'S':  # Bỏ qua đứng yên trong tìm đường
                    continue
                    
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)
                
                if not self.is_valid_position(next_pos):
                    continue
                
                tentative_g = g + 1
                
                if next_pos in g_scores and tentative_g >= g_scores[next_pos]:
                    continue
                    
                g_scores[next_pos] = tentative_g
                h = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])  # Manhattan
                f = tentative_g + h
                
                heapq.heappush(open_list, (f, tentative_g, next_pos, path + [current]))
        
        # Không tìm thấy đường đi
        self.path_cache[(start, goal)] = []
        return []
    
    def build_path(self, positions):
        """Chuyển đổi các vị trí thành chuỗi hành động"""
        if not positions or len(positions) <= 1:
            return []
            
        actions = []
        for i in range(1, len(positions)):
            prev, curr = positions[i-1], positions[i]
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            
            if dx == -1 and dy == 0:
                actions.append('U')
            elif dx == 1 and dy == 0:
                actions.append('D')
            elif dx == 0 and dy == -1:
                actions.append('L')
            elif dx == 0 and dy == 1:
                actions.append('R')
        
        return actions
    
    def find_cycles(self, robots, actions):
        """Phát hiện chu trình trong các hành động dự kiến"""
        # Lưu vị trí hiện tại và tiếp theo của robot
        current_pos = {}
        next_pos = {}
        
        for i in range(len(robots)):
            pos = (robots[i][0], robots[i][1])
            current_pos[i] = pos
            next_pos[i] = self.get_next_position(pos, actions[i][0])
        
        # Tạo đồ thị phụ thuộc (dependency graph)
        dependency_graph = {}
        for i in range(len(robots)):
            dependency_graph[i] = []
            for j in range(len(robots)):
                if i != j and next_pos[i] == current_pos[j]:
                    dependency_graph[i].append(j)
        
        # Phát hiện chu trình
        cycles = []
        visited = set()
        
        def dfs(node, path):
            if node in path:
                cycle = path[path.index(node):]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
                
            visited.add(node)
            path.append(node)
            
            for neighbor in dependency_graph[node]:
                if dfs(neighbor, path):
                    return True
            
            path.pop()
            return False
        
        for i in range(len(robots)):
            if i not in visited:
                dfs(i, [])
                
        return cycles
    
    def assign_packages(self, state):
        """Phân công gói hàng cho robot tự do"""
        free_robots = [i for i, r in enumerate(self.robots) 
                      if r[2] == 0 and self.robot_targets[i] == 'free']
        
        # Lấy danh sách gói hàng chưa được phân công
        free_packages = [p for p in state['packages'] 
                       if p[0] not in self.assigned_packages]
        
        if not free_robots or not free_packages:
            return
        
        # Tính ma trận chi phí
        cost_matrix = np.full((len(free_robots), len(free_packages)), float('inf'))
        
        for i, robot_id in enumerate(free_robots):
            robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
            
            for j, pkg in enumerate(free_packages):
                pickup = (pkg[1], pkg[2])
                delivery = (pkg[3], pkg[4])
                deadline = pkg[5]
                
                # Tính đường đi và chi phí
                pickup_path = self.a_star_search(robot_pos, pickup)
                if not pickup_path:
                    continue
                    
                delivery_path = self.a_star_search(pickup, delivery)
                if not delivery_path:
                    continue
                
                pickup_cost = len(pickup_path)
                delivery_cost = len(delivery_path)
                total_cost = pickup_cost + delivery_cost
                
                # Tính toán độ ưu tiên dựa trên deadline và chi phí
                est_arrival_time = state['time_step'] + total_cost
                time_left = deadline - state['time_step']
                
                # Ưu tiên gói hàng sắp hết hạn
                urgency = 0
                if time_left > 0:
                    if time_left <= total_cost:  # Gần hết hạn
                        urgency = 50
                    elif time_left < total_cost * 2:  # Cần xử lý sớm
                        urgency = 30
                    else:  # Còn nhiều thời gian
                        urgency = max(0, 20 - time_left/10)
                
                # Tính lợi nhuận dự kiến
                expected_reward = 10 if est_arrival_time <= deadline else 1
                profit = expected_reward - total_cost * 0.01
                
                # Chi phí càng thấp càng tốt (nghịch đảo để tìm min)
                final_cost = -profit - urgency/100
                cost_matrix[i, j] = final_cost
        
        # Giải bài toán phân công tối ưu
        row_ind, col_ind = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
        
        # Đánh dấu robot và package đã được phân công
        assigned_robots = set()
        for i, j in zip(row_ind, col_ind):
            if i in assigned_robots or cost_matrix[i, j] == float('inf'):
                continue
                
            robot_id = free_robots[i]
            pkg_id = free_packages[j][0]
            
            self.robot_targets[robot_id] = pkg_id
            self.assigned_packages.add(pkg_id)
            assigned_robots.add(i)
            
            if len(assigned_robots) == len(free_robots):
                break
    
    def resolve_conflicts(self, robots, actions):
        """Giải quyết xung đột giữa các robot"""
        # Phát hiện chu trình
        cycles = self.find_cycles(robots, actions)
        
        for cycle in cycles:
            # Giải quyết chu trình bằng cách thay đổi hành động của robot có ID thấp nhất
            robot_id = min(cycle)
            actions[robot_id] = ('S', actions[robot_id][1])
        
        # Kiểm tra và giải quyết va chạm trực tiếp
        next_positions = {}
        for i in range(len(robots)):
            pos = (robots[i][0], robots[i][1])
            next_pos = self.get_next_position(pos, actions[i][0])
            
            # Kiểm tra nếu có nhiều robot hướng đến cùng vị trí
            if next_pos in next_positions:
                # Robot có ID cao hơn sẽ đứng yên
                if i > next_positions[next_pos]:
                    actions[i] = ('S', actions[i][1])
                else:
                    actions[next_positions[next_pos]] = ('S', actions[next_positions[next_pos]][1])
                    next_positions[next_pos] = i
            else:
                next_positions[next_pos] = i
        
        return actions
    
    def get_actions(self, state):
        """Hàm chính để lấy hành động cho tất cả robot"""
        if not hasattr(self, 'n_robots'):
            self.init_agents(state)
            
        # Cập nhật trạng thái robot
        for i, robot in enumerate(state['robots']):
            prev_pos = (self.robots[i][0], self.robots[i][1])
            prev_package = self.robots[i][2]
            self.robots[i] = (robot[0], robot[1], robot[2])
            
            # Kiểm tra robot đứng yên
            if (robot[0], robot[1]) == prev_pos:
                self.repeat_count[i] += 1
            else:
                self.repeat_count[i] = 0
            
            # Kiểm tra gói hàng đã được giao
            if prev_package != 0 and robot[2] == 0:
                if prev_package in self.assigned_packages:
                    self.assigned_packages.remove(prev_package)
                self.robot_targets[i] = 'free'
                
            # Cập nhật packages vào cache nếu cần
            for pkg in state['packages']:
                pkg_id = pkg[0]
                if pkg_id not in self.packages:
                    self.packages[pkg_id] = (pkg[1], pkg[2], pkg[3], pkg[4], pkg[5])
            
        # Phân công gói hàng cho robot tự do
        self.assign_packages(state)
        
        # Tạo hành động mặc định
        actions = [('S', '0') for _ in range(self.n_robots)]
        
        # Xử lý robot đang mang gói hàng (giao hàng)
        for i in range(self.n_robots):
            robot_pos = (self.robots[i][0], self.robots[i][1])
            
            if self.robots[i][2] != 0:  # Robot đang mang gói hàng
                pkg_id = self.robots[i][2]
                if pkg_id in self.packages:
                    delivery_pos = (self.packages[pkg_id][2], self.packages[pkg_id][3])
                    
                    if robot_pos == delivery_pos:
                        # Đã ở vị trí giao hàng
                        actions[i] = ('S', '2')
                    else:
                        # Tìm đường đi đến điểm giao
                        path = self.a_star_search(robot_pos, delivery_pos)
                        if path:
                            actions[i] = (path[0], '0')
                
            elif self.robot_targets[i] != 'free':  # Robot đi lấy gói hàng
                pkg_id = self.robot_targets[i]
                if pkg_id in self.packages:
                    pickup_pos = (self.packages[pkg_id][0], self.packages[pkg_id][1])
                    
                    if robot_pos == pickup_pos:
                        # Đã ở vị trí lấy hàng
                        actions[i] = ('S', '1')
                    else:
                        # Tìm đường đi đến điểm lấy
                        path = self.a_star_search(robot_pos, pickup_pos)
                        if path:
                            actions[i] = (path[0], '0')
            
            # Giải quyết robot bị kẹt (đứng yên quá lâu)
            if self.repeat_count[i] > 5:
                # Thử di chuyển ngẫu nhiên để thoát khỏi tình trạng kẹt
                available_moves = []
                for move in ['U', 'D', 'L', 'R']:
                    next_pos = self.get_next_position(robot_pos, move)
                    if next_pos != robot_pos:  # Nếu di chuyển được
                        available_moves.append(move)
                
                if available_moves:
                    import random
                    actions[i] = (random.choice(available_moves), '0')
                    # Reset counter
                    self.repeat_count[i] = 0
        
        # Giải quyết xung đột
        actions = self.resolve_conflicts(state['robots'], actions)
        
        self.last_actions = actions

        print("State robot: ", self.robots)

        print("N robots = ", len(self.robots))
        print("Actions = ", actions)
        return actions