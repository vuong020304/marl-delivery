import random
import numpy as np  # Cần cho các tính toán
import collections  # Để phát hiện mẫu lặp lại

def run_bfs(map, start, goal):
    n_rows = len(map)
    n_cols = len(map[0])

    queue = []
    visited = set()
    queue.append((goal, []))
    visited.add(goal)
    d = {}
    d[goal] = 0

    while queue:
        current, path = queue.pop(0)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if next_pos[0] < 0 or next_pos[0] >= n_rows or next_pos[1] < 0 or next_pos[1] >= n_cols:
                continue
            if next_pos not in visited and map[next_pos[0]][next_pos[1]] == 0:
                visited.add(next_pos)
                d[next_pos] = d[current] + 1
                queue.append((next_pos, path + [next_pos]))

    if start not in d:
        return 'S', 100000

    t = 0
    actions = ['U', 'D', 'L', 'R']
    current = start
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_pos = (current[0] + dx, current[1] + dy)
        if next_pos in d:
            if d[next_pos] == d[current] - 1:
                return actions[t], d[next_pos]
        t += 1
    return 'S', d[start]


class GreedyAgentsOptimal:
    def __init__(self):
        self.agents = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.state = None
        self.is_init = False
        self.idle_time = []  # Thời gian robot đứng yên
        self.previous_pos = []  # Vị trí trước đó
        self.target_attempts = {}  # Số lần thử với từng gói hàng
        self.time_step = 0
        
        # Thêm cấu trúc dữ liệu để theo dõi lịch sử di chuyển của mỗi robot
        self.move_history = {}  # {robot_id: deque([(pos, action), ...], maxlen=10)}
        self.detected_patterns = {}  # {robot_id: True/False}
        
        # Thêm cấu trúc để ghi nhớ các vị trí đã đi qua
        self.visited_positions = {}  # {robot_id: {pos: timestamp}}
        self.position_frequency = {}  # {robot_id: {pos: count}}
        self.deadlock_zones = set()  # Vùng thường xảy ra deadlock
        self.global_visited = {}  # {pos: [robot_ids]} - ghi nhớ robot nào đã đi qua vị trí nào

    def init_agents(self, state):
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(robot[0] - 1, robot[1] - 1, 0) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.packages += [(p[0], p[1] - 1, p[2] - 1, p[3] - 1, p[4] - 1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)
        self.idle_time = [0] * self.n_robots
        self.previous_pos = [(r[0], r[1]) for r in self.robots]
        self.already_processed_packages = set()
        self.move_history = {i: collections.deque(maxlen=10) for i in range(self.n_robots)}
        self.detected_patterns = {i: False for i in range(self.n_robots)}
        self.visited_positions = {i: {} for i in range(self.n_robots)}
        self.position_frequency = {i: {} for i in range(self.n_robots)}
        
        # Phân tích bản đồ để tìm khu vực có khả năng deadlock
        self.analyze_map_for_deadlocks()

    def analyze_map_for_deadlocks(self):
        """Phân tích bản đồ để xác định khu vực có khả năng gây deadlock"""
        # Tìm các hành lang hẹp (chỉ có 1 ô đi qua được)
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == 0:  # ô trống
                    # Đếm số ô trống xung quanh
                    empty_neighbors = 0
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < len(self.map) and 0 <= nj < len(self.map[0]) and self.map[ni][nj] == 0:
                            empty_neighbors += 1
                    
                    # Nếu chỉ có 2 lối đi (hành lang), đánh dấu là khu vực tiềm ẩn deadlock
                    if empty_neighbors == 2:
                        self.deadlock_zones.add((i, j))

    def update_move_to_target(self, robot_id, target_package_id, phase='start'):
        if phase == 'start':
            distance = abs(self.packages[target_package_id][1] - self.robots[robot_id][0]) + \
                       abs(self.packages[target_package_id][2] - self.robots[robot_id][1])
        else:
            # Chuyển sang khoảng cách đến đích (3, 4) nếu phase == 'target'
            distance = abs(self.packages[target_package_id][3] - self.robots[robot_id][0]) + \
                       abs(self.packages[target_package_id][4] - self.robots[robot_id][1])
        i = robot_id

        # CẢI TIẾN: Giảm ngưỡng thời gian đứng yên để tăng phản ứng
        if self.idle_time[robot_id] > 3:  # Giảm từ 5 xuống 3
            move = self.get_random_move(robot_id)
            if move != 'S':  # Nếu tìm được hướng ngẫu nhiên hợp lệ
                return move, '0'

        # Thông thường, tìm đường bằng BFS
        pkg_act = 0
        move = 'S'
        if distance >= 1:
            pkg = self.packages[target_package_id]

            target_p = (pkg[1], pkg[2])
            if phase == 'target':
                target_p = (pkg[3], pkg[4])
            move, distance = run_bfs(self.map, (self.robots[i][0], self.robots[i][1]), target_p)

            # CẢI TIẾN: Thêm xử lý khi BFS không tìm được đường đi tốt
            if move == 'S' and self.idle_time[robot_id] > 2:
                # Thử các hướng khác nếu BFS trả về đứng yên
                alternate_move = self.get_break_pattern_move(robot_id)
                if alternate_move != 'S':
                    return alternate_move, '0'

            if distance == 0:
                if phase == 'start':
                    pkg_act = 1  # Nhặt gói
                else:
                    pkg_act = 2  # Thả gói
        else:
            move = 'S'
            pkg_act = 1
            if phase == 'start':
                pkg_act = 1  # Nhặt gói
            else:
                pkg_act = 2  # Thả gói

        return move, str(pkg_act)

    def get_random_move(self, robot_id):
        """Trả về một hướng di chuyển ngẫu nhiên hợp lệ"""
        robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
        possible_moves = []
        
        # Kiểm tra tất cả các hướng có thể
        for move, (dx, dy) in [('U', (-1, 0)), ('D', (1, 0)), ('L', (0, -1)), ('R', (0, 1))]:
            new_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            if self.is_valid_position(new_pos):
                # Ưu tiên các hướng chưa đi qua gần đây
                freq = self.position_frequency[robot_id].get(new_pos, 0)
                if freq < 3:  # Nếu vị trí chưa được đi qua quá nhiều lần
                    possible_moves.append((move, freq))
                    
        if possible_moves:
            # Sắp xếp theo tần suất thăm (ưu tiên vị trí ít đi qua)
            possible_moves.sort(key=lambda x: x[1])
            return possible_moves[0][0]
            
        # Nếu không có vị trí ưu tiên, chọn bất kỳ vị trí hợp lệ nào
        for move, (dx, dy) in [('U', (-1, 0)), ('D', (1, 0)), ('L', (0, -1)), ('R', (0, 1))]:
            new_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            if self.is_valid_position(new_pos):
                possible_moves.append(move)
                
        if possible_moves:
            return random.choice(possible_moves)
        return 'S'  # Nếu không có hướng nào hợp lệ

    def is_valid_position(self, position):
        """Kiểm tra xem vị trí có hợp lệ không"""
        x, y = position
        if x < 0 or x >= len(self.map) or y < 0 or y >= len(self.map[0]):
            return False
        if self.map[x][y] == 1:
            return False
        return True

    def detect_pattern(self, robot_id):
        """Phát hiện mẫu lặp lại trong lịch sử di chuyển của robot"""
        history = self.move_history[robot_id]
        if len(history) < 6:  # Cần ít nhất 6 bước để có mẫu lặp lại có ý nghĩa
            return False
            
        # Kiểm tra mẫu lặp lại đơn giản: LRLR hoặc UDUD hoặc tương tự
        history_list = list(history)  # Chuyển deque thành list
        moves = [h[1] for h in history_list]
        positions = [h[0] for h in history_list]
        
        # Kiểm tra vị trí lặp lại - nếu trở lại vị trí cũ trong vòng lặp ngắn
        position_set = set(positions[-6:])
        if len(position_set) < 3:  # Nếu di chuyển qua ít hơn 3 vị trí trong 6 bước gần đây
            return True
        
        # Kiểm tra mẫu lặp lại 2 bước
        if len(moves) >= 4:
            if moves[-1] == moves[-3] and moves[-2] == moves[-4]:
                # Phát hiện mẫu lặp lại 2 bước, ví dụ: LRLR
                return True
                
        # Kiểm tra mẫu lặp lại đơn giản: LR-LR hoặc tương tự
        for pattern_len in [2, 3, 4]:
            if len(moves) < pattern_len * 2:
                continue
                
            pattern1 = moves[-pattern_len:]
            pattern2 = moves[-(pattern_len*2):-pattern_len]
            
            if pattern1 == pattern2:
                return True
                
        return False

    def get_break_pattern_move(self, robot_id):
        """Tạo một di chuyển để phá vỡ mẫu lặp lại, tránh các vị trí đã đi nhiều lần"""
        pos = (self.robots[robot_id][0], self.robots[robot_id][1])
        
        # Lấy 4 di chuyển gần đây nhất
        recent_moves = [h[1] for h in list(self.move_history[robot_id])[-4:]] if len(self.move_history[robot_id]) >= 4 else []
        
        # Tạo danh sách các di chuyển có thể với điểm ưu tiên
        scored_moves = []
        for move, (dx, dy) in [('U', (-1, 0)), ('D', (1, 0)), ('L', (0, -1)), ('R', (0, 1))]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if not self.is_valid_position(new_pos):
                continue
                
            # Tính điểm ưu tiên cho di chuyển này (càng thấp càng tốt)
            score = 0
            
            # Nếu đã đi qua vị trí này nhiều lần, tăng điểm (không ưu tiên)
            freq = self.position_frequency[robot_id].get(new_pos, 0)
            score += freq * 3
            
            # Nếu di chuyển này nằm trong mẫu gần đây, tăng điểm
            if move in recent_moves:
                score += 5
                
            # Nếu vị trí mới nằm trong vùng có khả năng deadlock, tăng điểm
            if new_pos in self.deadlock_zones:
                score += 4
                
            # Thêm vào danh sách với điểm số
            scored_moves.append((move, score, new_pos))
            
        if scored_moves:
            # Sắp xếp theo điểm (ưu tiên điểm thấp)
            scored_moves.sort(key=lambda x: x[1])
            return scored_moves[0][0]
        
        return 'S'  # Không tìm được di chuyển hợp lệ

    def get_smart_path_move(self, robot_id, target_pos):
        """Tìm đường đi thông minh đến đích, tránh các vùng deadlock"""
        curr_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
        
        # Nếu robot đang ở vị trí deadlock và bị kẹt
        if curr_pos in self.deadlock_zones and self.idle_time[robot_id] > 3:
            # Thử di chuyển ra khỏi vùng deadlock
            for move, (dx, dy) in [('U', (-1, 0)), ('D', (1, 0)), ('L', (0, -1)), ('R', (0, 1))]:
                new_pos = (curr_pos[0] + dx, curr_pos[1] + dy)
                if self.is_valid_position(new_pos) and new_pos not in self.deadlock_zones:
                    return move, '0'
        
        # Sử dụng BFS thông thường
        move, _ = run_bfs(self.map, curr_pos, target_pos)
        return move, '0'
    
    def update_inner_state(self, state):
        self.time_step += 1
        # Cập nhật vị trí và trạng thái robot
        for i in range(len(state['robots'])):
            prev_pos = (self.robots[i][0], self.robots[i][1])
            prev = (prev_pos[0], prev_pos[1], self.robots[i][2])
            robot = state['robots'][i]
            self.robots[i] = (robot[0] - 1, robot[1] - 1, robot[2])
            current_pos = (self.robots[i][0], self.robots[i][1])
            
            # Cập nhật bản đồ tần suất vị trí
            if current_pos not in self.position_frequency[i]:
                self.position_frequency[i][current_pos] = 1
            else:
                self.position_frequency[i][current_pos] += 1
                
            # Cập nhật vị trí toàn cục
            if current_pos not in self.global_visited:
                self.global_visited[current_pos] = []
            if i not in self.global_visited[current_pos]:
                self.global_visited[current_pos].append(i)
                
            # Cập nhật lịch sử di chuyển
            if prev_pos != current_pos:
                # Xác định hướng di chuyển
                if prev_pos[0] < current_pos[0]:
                    move = 'D'
                elif prev_pos[0] > current_pos[0]:
                    move = 'U'
                elif prev_pos[1] < current_pos[1]:
                    move = 'R'
                elif prev_pos[1] > current_pos[1]:
                    move = 'L'
                else:
                    move = 'S'
                    
                self.move_history[i].append((current_pos, move))
                
                # Phát hiện mẫu lặp lại
                if self.detect_pattern(i):
                    if not self.detected_patterns[i]:
                        print(f"Phát hiện robot {i} đang lặp lại mẫu di chuyển. Sẽ cố gắng phá vỡ mẫu.")
                        self.detected_patterns[i] = True
                else:
                    self.detected_patterns[i] = False
            
            # Kiểm tra robot có di chuyển không
            if prev_pos == current_pos:
                self.idle_time[i] += 1
                
                # Phát hiện kẹt sớm hơn
                if self.idle_time[i] > 5:
                    # Đánh dấu vị trí kẹt
                    self.deadlock_zones.add(current_pos)
                
                # CẢI TIẾN: Reset nhanh nếu bị kẹt để không mất thời gian
                if self.idle_time[i] > 7 and self.robots_target[i] != 'free' and self.robots[i][2] == 0:
                    pkg_id = self.robots_target[i]
                    key = (i, pkg_id)
                    if key not in self.target_attempts:
                        self.target_attempts[key] = 1
                    else:
                        self.target_attempts[key] += 1
                    
                    # Reset luôn nếu bị kẹt quá 7 bước
                    print(f"Robot {i} từ bỏ gói {pkg_id} vì bị kẹt")
                    self.robots_target[i] = 'free'
                    self.packages_free[pkg_id - 1] = True
            else:
                self.idle_time[i] = 0
                
            self.previous_pos[i] = prev_pos
            
            if prev[2] != 0:
                if self.robots[i][2] == 0:
                    # Robot đã thả gói hàng
                    self.robots_target[i] = 'free'
                else:
                    self.robots_target[i] = self.robots[i][2]

        # Cập nhật vị trí và trạng thái gói hàng
        new_packages = []
        for p in state['packages']:
            pkg_id = p[0]
            if pkg_id not in self.already_processed_packages:
                new_packages.append((p[0], p[1] - 1, p[2] - 1, p[3] - 1, p[4] - 1, p[5]))
                self.already_processed_packages.add(pkg_id)
                
        if new_packages:
            self.packages += new_packages
            self.packages_free += [True] * len(new_packages)

    def compute_valid_position(self, map, position, move):
        """
        Tính toán vị trí mới dự kiến cho robot dựa trên vị trí hiện tại và lệnh di chuyển.
        """
        r, c = position
        if move == 'S':
            i, j = r, c
        elif move == 'L':
            i, j = r, c - 1
        elif move == 'R':
            i, j = r, c + 1
        elif move == 'U':
            i, j = r - 1, c
        elif move == 'D':
            i, j = r + 1, c
        else:
            i, j = r, c
        if i < 0 or i >= len(self.map) or j < 0 or j >= len(self.map[0]):
            return r, c
        if map[i][j] == 1:
            return r, c
        return i, j


    def valid_position(self, map, position):
        i, j = position
        if i < 0 or i >= len(self.map) or j < 0 or j >= len(self.map[0]):
            return False
        if map[i][j] == 1:
            return False
        return True
        
    def calculate_package_priority(self, robot_id, package_id):
        """Tính điểm ưu tiên để chọn gói hàng tốt nhất"""
        pkg = self.packages[package_id]
        robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
        
        # Khoảng cách đến điểm lấy
        pickup_distance = abs(pkg[1] - robot_pos[0]) + abs(pkg[2] - robot_pos[1])
        
        # Khoảng cách từ điểm lấy đến điểm giao
        delivery_distance = abs(pkg[3] - pkg[1]) + abs(pkg[4] - pkg[2])
        
        # Tổng khoảng cách - yếu tố chính
        total_distance = pickup_distance + delivery_distance
        
        # CẢI TIẾN: Thêm yếu tố thời gian và deadline
        time_remaining = pkg[5] + 100 - self.time_step
        
        # CẢI TIẾN QUAN TRỌNG: Tính điểm với công thức mới
        
        # 1. THAY ĐỔI ĐỘT PHÁ: Ưu tiên cực kỳ cao cho gói gần
        if pickup_distance <= 3:
            score = -30  # Siêu ưu tiên cho gói rất gần
        else:
            # Với các gói xa hơn, ưu tiên dựa trên khoảng cách và thời gian
            score = pickup_distance * 1.5 + delivery_distance * 0.5
        
        # 2. Cơ chế thông minh hơn cho độ khẩn cấp
        time_urgency = max(0, 80 - time_remaining)  # Giá trị lớn = gấp
        
        # 3. Điều chỉnh điểm theo thời gian còn lại
        # Nếu có thể giao hàng kịp
        if total_distance + 10 < time_remaining:  # Thêm buffer 10 bước
            # Gói có thể giao kịp - ưu tiên cao cho gói gấp
            score -= time_urgency * 0.6
            
            # Ưu tiên thêm cho gói rất gấp
            if time_remaining < 30:
                score -= 15
        else:
            # Gói không thể giao kịp - giảm ưu tiên mạnh
            score += 40
        
        # 4. Giảm ưu tiên gói đã thử nhiều lần
        key = (robot_id, pkg[0])
        if key in self.target_attempts:
            score += self.target_attempts[key] * 25
        
        return score

    def get_actions(self, state):
        if self.is_init == False:
            self.is_init = True
            self.update_inner_state(state)
        else:
            self.update_inner_state(state)

        actions = []
        map = state['map']
        
        # CẢI TIẾN: Giảm ngưỡng phát hiện và tăng tần suất reset
        stuck_robots = sum(1 for t in self.idle_time if t > 5)
        if stuck_robots >= self.n_robots * 0.3 and self.time_step > 10:
            print(f"Phát hiện tình trạng kẹt: {stuck_robots}/{self.n_robots} robot bị kẹt!")
            
            # Reset nhiều robot không mang hàng để thoát khỏi kẹt
            for i in range(self.n_robots):
                if self.idle_time[i] > 5 and self.robots[i][2] == 0 and self.robots_target[i] != 'free':
                    pkg_id = self.robots_target[i]
                    self.robots_target[i] = 'free'
                    self.packages_free[pkg_id - 1] = True
        
        # CẢI TIẾN QUAN TRỌNG: Ưu tiên cao cho các gói gấp
        urgent_packages = []
        for j in range(len(self.packages)):
            if not self.packages_free[j]:
                continue
                
            pkg = self.packages[j]
            # Chỉ xét gói đã xuất hiện
            if pkg[5] <= self.time_step:
                time_remaining = pkg[5] + 100 - self.time_step
                # Gói đang gấp
                if 0 < time_remaining < 50:
                    robot_pos = None
                    min_dist = float('inf')
                    
                    # Tìm robot rảnh gần nhất để xử lý gói gấp này
                    for r_id in range(self.n_robots):
                        if self.robots_target[r_id] == 'free':
                            r_pos = (self.robots[r_id][0], self.robots[r_id][1])
                            dist = abs(pkg[1] - r_pos[0]) + abs(pkg[2] - r_pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                robot_pos = r_id
                    
                    # Nếu có robot rảnh và khoảng cách hợp lý
                    if robot_pos is not None and min_dist < 20:
                        # Gán gói ngay cho robot gần nhất
                        self.packages_free[j] = False
                        self.robots_target[robot_pos] = pkg[0]
                        print(f"Gói khẩn cấp {pkg[0]} được gán cho robot {robot_pos}")
        
        # Phân công các robot còn lại như thông thường
        for i in range(self.n_robots):
            if self.robots_target[i] != 'free':
                # Xử lý robot đã có mục tiêu
                closest_package_id = self.robots_target[i]
                
                # Bước 1b: Kiểm tra xem robot đã đến gói hàng chưa
                if self.robots[i][2] != 0:
                    # Di chuyển đến điểm đích
                    pkg = self.packages[closest_package_id - 1]
                    target_pos = (pkg[3], pkg[4])  # Vị trí đích
                    
                    # Phát hiện mẫu lặp lại hoặc bị kẹt lâu
                    if (self.detected_patterns[i] and self.idle_time[i] > 2) or self.idle_time[i] > 4:  # Giảm ngưỡng
                        # Thử sử dụng đường đi thông minh
                        move, _ = self.get_smart_path_move(i, target_pos)
                        
                        # Nếu vẫn không di chuyển được, thử di chuyển phá vỡ mẫu
                        if move == 'S' or ((self.robots[i][0], self.robots[i][1]) in self.deadlock_zones and self.idle_time[i] > 4):
                            alternate_move = self.get_break_pattern_move(i)
                            if alternate_move != 'S':
                                print(f"Robot {i} đang phá vỡ mẫu lặp lại, thay thế {move} bằng {alternate_move}")
                                move = alternate_move
                        
                        actions.append((move, '0'))
                    else:
                        # Sử dụng thuật toán thông thường
                        move, action = self.update_move_to_target(i, closest_package_id - 1, 'target')
                        actions.append((move, action))
                else:
                    # Bước 1c: Tiếp tục di chuyển đến gói hàng
                    pkg = self.packages[closest_package_id - 1]
                    target_pos = (pkg[1], pkg[2])  # Vị trí lấy hàng
                    
                    # Phát hiện mẫu lặp lại hoặc bị kẹt lâu
                    if (self.detected_patterns[i] and self.idle_time[i] > 2) or self.idle_time[i] > 4:  # Giảm ngưỡng
                        # Thử sử dụng đường đi thông minh
                        move, _ = self.get_smart_path_move(i, target_pos)
                        
                        # Nếu vẫn không di chuyển được, thử di chuyển phá vỡ mẫu
                        if move == 'S' or ((self.robots[i][0], self.robots[i][1]) in self.deadlock_zones and self.idle_time[i] > 4):
                            alternate_move = self.get_break_pattern_move(i)
                            if alternate_move != 'S':
                                print(f"Robot {i} đang phá vỡ mẫu lặp lại, thay thế {move} bằng {alternate_move}")
                                move = alternate_move
                        
                        actions.append((move, '0'))
                    else:
                        # Sử dụng thuật toán thông thường
                        move, action = self.update_move_to_target(i, closest_package_id - 1)
                        actions.append((move, action))
            else:
                # Tìm gói hàng tốt nhất cho robot rảnh
                best_package_id = None
                best_score = float('inf')
                
                for j in range(len(self.packages)):
                    if not self.packages_free[j]:
                        continue
                        
                    if self.packages[j][5] <= self.time_step:  # Gói đã xuất hiện
                        score = self.calculate_package_priority(i, j)
                        if score < best_score:
                            best_score = score
                            best_package_id = j
                
                if best_package_id is not None:
                    self.packages_free[best_package_id] = False
                    self.robots_target[i] = self.packages[best_package_id][0]
                    move, action = self.update_move_to_target(i, best_package_id)
                    actions.append((move, action))
                else:
                    # Không có gói, di chuyển ngẫu nhiên
                    move = self.get_random_move(i)
                    actions.append((move, '0'))
                    
        return actions