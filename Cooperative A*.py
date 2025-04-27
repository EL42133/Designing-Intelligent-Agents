import gymnasium as gym
import rware
import time
import datetime
from enum import Enum
import heapq
import numpy as np

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
GOAL_BUFFER_TIME = 2
_MAX_EXPANSION = 5000

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4

class TaskPhase(Enum):
    GO_TO_PICKUP = 0
    DELIVER = 1
    PICKUP_AGAIN = 2
    RETURN_TO_PICKUP = 3
    DONE = 4

class STANode:
    def __init__(self, x, y, t, parent=None):
        self.x = x
        self.y = y
        self.t = t
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Central Reservation Table for Space-Time Positions
class ReservationTable:
    def __init__(self):
        self.position_reservations = {}  # (x, y, t) -> agent_id
        self.edge_reservations = {}      # ((x1, y1), (x2, y2), t) -> agent_id
    
    def is_position_free(self, x, y, t, agent_id=None):
        key = (x, y, t)
        if key in self.position_reservations and self.position_reservations[key] != agent_id:
            return False
        return True
    
    def is_edge_free(self, pos1, pos2, t, agent_id=None):
        key = (pos1, pos2, t)
        reverse_key = (pos2, pos1, t)
        if (key in self.edge_reservations and self.edge_reservations[key] != agent_id) or \
           (reverse_key in self.edge_reservations and self.edge_reservations[reverse_key] != agent_id):
            return False
        return True
    
    def add_path_reservation(self, path, agent_id, carrying=False, buffer_time=GOAL_BUFFER_TIME):
        """Reserve all positions and edges along a path"""
        success = True
        reservations = []
        
        # First check if the entire path can be reserved
        for t, pos in enumerate(path):
            if not self.is_position_free(pos[0], pos[1], t, agent_id):
                success = False
                break
            
            if t > 0:
                prev_pos = path[t-1]
                if not self.is_edge_free(prev_pos, pos, t-1, agent_id):
                    success = False
                    break
            
            # Check for turn delays
            if t > 0 and t < len(path) - 1:
                prev_pos = path[t-1]
                next_pos = path[t+1]
                
                # Calculate direction vectors
                prev_dir = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                next_dir = (next_pos[0] - pos[0], next_pos[1] - pos[1])
                
                # If direction changed, check if turn positions are free
                if prev_dir != next_dir:
                    dir_to_compass = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
                    prev_compass = dir_to_compass.get(prev_dir, 0)
                    next_compass = dir_to_compass.get(next_dir, 0)
                    turns_needed = min((next_compass - prev_compass) % 4, (prev_compass - next_compass) % 4)
                    
                    for delay in range(1, turns_needed + 1):
                        if t + delay < len(path) and not self.is_position_free(pos[0], pos[1], t + delay, agent_id):
                            success = False
                            break
        
        # Add padding at the end for goal positions
        final_pos = path[-1] if path else None
        if final_pos:
            for pad in range(1, buffer_time + 1):
                t_pad = len(path) + pad - 1
                if not self.is_position_free(final_pos[0], final_pos[1], t_pad, agent_id):
                    success = False
                    break
        
        # If the entire path can be reserved, make the reservations
        if success:
            for t, pos in enumerate(path):
                self.position_reservations[(pos[0], pos[1], t)] = agent_id
                reservations.append((pos[0], pos[1], t))
                
                if t > 0:
                    prev_pos = path[t-1]
                    self.edge_reservations[(prev_pos, pos, t-1)] = agent_id
                    self.edge_reservations[(pos, prev_pos, t-1)] = agent_id
                
                # Add turn delays
                if t > 0 and t < len(path) - 1:
                    prev_pos = path[t-1]
                    next_pos = path[t+1]
                    
                    prev_dir = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                    next_dir = (next_pos[0] - pos[0], next_pos[1] - pos[1])
                    
                    if prev_dir != next_dir:
                        dir_to_compass = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
                        prev_compass = dir_to_compass.get(prev_dir, 0)
                        next_compass = dir_to_compass.get(next_dir, 0)
                        turns_needed = min((next_compass - prev_compass) % 4, (prev_compass - next_compass) % 4)
                        
                        for delay in range(1, turns_needed + 1):
                            if t + delay < len(path):
                                self.position_reservations[(pos[0], pos[1], t + delay)] = agent_id
                                reservations.append((pos[0], pos[1], t + delay))
            
            # Add padding at the end
            if final_pos:
                for pad in range(1, buffer_time + 1):
                    t_pad = len(path) + pad - 1
                    self.position_reservations[(final_pos[0], final_pos[1], t_pad)] = agent_id
                    reservations.append((final_pos[0], final_pos[1], t_pad))
        
        return success, reservations
    
    def release_reservations(self, agent_id):
        """Release all reservations for an agent"""
        to_remove_pos = [key for key, aid in self.position_reservations.items() if aid == agent_id]
        to_remove_edge = [key for key, aid in self.edge_reservations.items() if aid == agent_id]
        
        for key in to_remove_pos:
            del self.position_reservations[key]
        
        for key in to_remove_edge:
            del self.edge_reservations[key]

# Track task delivery times
class TaskTimeTracker:
    def __init__(self):
        self.task_start_times = {}  # task_id -> start time
        self.task_deadlines = {}    # task_id -> deadline (start + priority)
        self.task_completion_times = {}  # task_id -> completion time
        self.current_time = 0       # global time counter for shelf movements
        self.task_priorities = {}   # task_id -> priority level
        
    def assign_task(self, task_id, priority):
        """Record when a task is assigned with its deadline based on priority"""
        # Only track tasks with priority > 0
        if priority <= 0:
            return
            
        self.task_start_times[task_id] = self.current_time
        self.task_deadlines[task_id] = self.current_time + priority
        self.task_priorities[task_id] = priority
        print(f"Task {task_id} assigned at time {self.current_time}, priority: {priority}, deadline: {self.task_deadlines[task_id]}")
        
    def tick_time(self, shelf_ids_moving):
        """Increment time only for shelves that are moving"""
        # Only increment global time if any shelf is moving
        if shelf_ids_moving:
            self.current_time += 1
            
    def complete_task(self, task_id):
        """Record task completion time and check if it met deadline"""
        # Only process tracked tasks (priority > 0)
        if task_id in self.task_start_times:
            self.task_completion_times[task_id] = self.current_time
            on_time = self.current_time <= self.task_deadlines[task_id]
            delay = max(0, self.current_time - self.task_deadlines[task_id])
            priority = self.task_priorities.get(task_id, 0)
            
            print(f"Task {task_id} (priority {priority}) completed at time {self.current_time}")
            print(f"Delivery {'on time' if on_time else f'LATE by {delay} steps'}")
            
            return on_time, delay
        return False, None
    
    def get_task_stats(self):
        """Get statistics about on-time delivery performance"""
        total_tasks = len(self.task_completion_times)
        if total_tasks == 0:
            return {"total": 0, "on_time": 0, "late": 0, "on_time_percentage": 0}
            
        on_time_count = sum(1 for task_id in self.task_completion_times 
                           if self.task_completion_times[task_id] <= self.task_deadlines[task_id])
        
        return {
            "total": total_tasks,
            "on_time": on_time_count,
            "late": total_tasks - on_time_count,
            "on_time_percentage": (on_time_count / total_tasks) * 100 if total_tasks > 0 else 0
        }

# Cooperative A*
def coop_astar(grid, start, goal, carrying, reservation_table, agent_id, max_time=200):
    
    open_list = []
    closed_set = set()
    g_score = {}
    layers, height, width = grid.shape
    
    # Define collision layers based on whether agent is carrying something
    shelf_layers = [_LAYER_SHELFS] if carrying else []
    collision_layers = [_LAYER_AGENTS] + shelf_layers
    
    # Create the starting node and set its scores
    start_node = STANode(start[0], start[1], 0)
    start_node.h = heuristic(start, goal)
    start_node.f = start_node.h
    heapq.heappush(open_list, start_node)
    g_score[(start_node.x, start_node.y, start_node.t)] = 0
    
    expansions = 0
    
    while open_list:
        expansions += 1
        if expansions > _MAX_EXPANSION:  # Increased expansion limit for thoroughness
            print(f"[STA*] Gave up on {start} -> {goal} after {_MAX_EXPANSION} expansions")
            return []
        
        current = heapq.heappop(open_list)
        
        # Goal check with temporal padding
        if (current.x, current.y) == goal:
            # Check if goal is free for GOAL_BUFFER_TIME
            goal_free = True
            for t in range(current.t + 1, current.t + GOAL_BUFFER_TIME + 1):
                if not reservation_table.is_position_free(goal[0], goal[1], t, agent_id):
                    goal_free = False
                    break
            
            if goal_free:
                path = []
                temp = current
                while temp:
                    path.append((temp.x, temp.y))
                    temp = temp.parent
                return path[::-1]
        
        # Time limit check to prevent infinite waits
        if current.t >= max_time:
            print(f"[STA*] Time limit reached for agent {agent_id}")
            return []
        
        closed_set.add((current.x, current.y, current.t))
        
        # Explore neighbors (including waiting)
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = current.x + dx, current.y + dy
            nt = current.t + 1
            
            # Check boundaries
            if not (0 <= nx < width and 0 <= ny < height):
                continue
                
            # Check if already explored
            if (nx, ny, nt) in closed_set:
                continue
                
            # Check reservation table
            if not reservation_table.is_position_free(nx, ny, nt, agent_id):
                continue
                
            # Check edge conflicts
            if dx != 0 or dy != 0:  # If not waiting
                if not reservation_table.is_edge_free((current.x, current.y), (nx, ny), current.t, agent_id):
                    continue
            
            # Check static obstacles (except at goal)
            if (nx, ny) != goal and any(grid[layer, ny, nx] > 0 for layer in collision_layers):
                continue
            
            # Cost calculation - moving diagonally costs more
            move_cost = 1 if (dx == 0 or dy == 0) else 1.4
            # Waiting has a slight penalty to encourage movement
            if dx == 0 and dy == 0:
                move_cost = 1.05
                
            tentative_g = current.g + move_cost
            key = (nx, ny, nt)
            
            if key in g_score and tentative_g >= g_score[key]:
                continue
            
            g_score[key] = tentative_g
            new_node = STANode(nx, ny, nt, current)
            new_node.g = tentative_g
            new_node.h = heuristic((nx, ny), goal)
            new_node.f = new_node.g + new_node.h
            heapq.heappush(open_list, new_node)
    
    print(f"[STA*] Failed to find a path from {start} to {goal} for agent {agent_id}")
    return []

# Translates agent path into actions
def get_actions_from_path(current_pos, current_dir, path):
    if not path or len(path) < 2:
        return []
    
    # Find the next position for agent
    next_pos = path[1]
    dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
    
    # Find the required direction for the agent to face
    if dx == 1: 
        required_dir = Direction.RIGHT
    elif dx == -1: 
        required_dir = Direction.LEFT
    elif dy == 1: 
        required_dir = Direction.DOWN
    elif dy == -1: 
        required_dir = Direction.UP
    else: 
        return []

    # Calculate the shortest turning time to face the right direction 
    direction_index = [Direction.UP.value, Direction.RIGHT.value, Direction.DOWN.value, Direction.LEFT.value]
    current_idx = direction_index.index(current_dir.value)
    target_idx = direction_index.index(required_dir.value)
    
    clockwise_steps = (target_idx - current_idx) % 4
    counterclockwise_steps = (current_idx - target_idx) % 4
    
    actions = []
    if clockwise_steps <= counterclockwise_steps:
        actions.extend([Action.RIGHT.value] * clockwise_steps)
    else:
        actions.extend([Action.LEFT.value] * counterclockwise_steps)
        
    actions.append(Action.FORWARD.value)
    
    return actions

# Task assignment function that respects agent priorities
def cbta_assign_tasks_with_priority(env, agent_positions, carrying_shelf, engaged_shelves, agent_paths, agent_priorities):
    tasks = env.env.request_queue
    goals = env.env.goals
    grid = env.env.grid
    
    # Sort tasks by priority (highest first)
    sorted_tasks = sorted(enumerate(tasks), key=lambda x: x[1].priority, reverse=True)
    
    # Get unassigned agents
    unassigned_agents = [idx for idx, (pos, carrying, engaged) in enumerate(zip(agent_positions, carrying_shelf, engaged_shelves)) 
                         if not carrying and engaged is None]
    
    if not unassigned_agents or not sorted_tasks:
        return [None for _ in agent_positions]
    
    # Track engaged shelves
    engaged_ids = {sid for sid in engaged_shelves if sid is not None}
    
    assignments = [None for _ in agent_positions]
    assigned_ids = set()
    
    # Assign high priority tasks first to any available agent
    for task_idx, task in sorted_tasks:
        if task.id in engaged_ids or task.id in assigned_ids:
            continue
            
        pickup = (task.x, task.y)
        best_agent = None
        best_cost = float('inf')
        best_paths = None
            
        # Find best agent for this task
        for agent_idx in unassigned_agents:
            if assignments[agent_idx] is not None:
                continue
                
            # Get agent position
            agent_pos = agent_positions[agent_idx]
            
            # Find path to pickup
            path_to_pickup = coop_astar(grid, agent_pos, pickup, False, ReservationTable(), agent_idx)
            if not path_to_pickup:
                continue
                
            # Find best goal and path to it
            best_goal_cost = float('inf')
            best_goal_path = None
            best_goal_pos = None
            
            for goal in goals:
                path_to_goal = coop_astar(grid, pickup, goal, True, ReservationTable(), agent_idx) 
                if path_to_goal:
                    cost = len(path_to_pickup) + len(path_to_goal)
                    if cost < best_goal_cost:
                        best_goal_cost = cost
                        best_goal_path = path_to_goal
                        best_goal_pos = goal
            
            if best_goal_path:
                total_cost = len(path_to_pickup) + len(best_goal_path)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_agent = agent_idx
                    best_paths = (pickup, best_goal_pos, path_to_pickup, best_goal_path)
                    
        # Assign the task if we found a suitable agent
        if best_agent is not None:
            assignments[best_agent] = best_paths
            engaged_shelves[best_agent] = task.id
            assigned_ids.add(task.id)
            
            # Update agent priority based on task priority
            agent_priorities[best_agent] = task.priority
            print(f"Assigned task {task.id} (priority {task.priority}) to agent {best_agent}")
            
    return assignments

# Handle task phase transitions with time tracking
def handle_task_phase(agent_idx, agent, current_pos, carrying, phase, env, original_pickup, assigned_goals, agent_paths, carrying_shelf, 
                      task_states, completed_tasks, engaged_shelves, n_iterations, agent_priorities, reservation_table, time_tracker):
    
    if phase == TaskPhase.GO_TO_PICKUP:
        if original_pickup[agent_idx] and current_pos == original_pickup[agent_idx] and not carrying:
            carrying_shelf[agent_idx] = True
            task_states[agent_idx] = TaskPhase.DELIVER
            
            # Release previous reservations if using reservation table
            if reservation_table:
                reservation_table.release_reservations(agent_idx)
            
            if reservation_table is None:
                path = coop_astar(env.env.grid, current_pos, assigned_goals[agent_idx], True, ReservationTable(), agent_idx)
                agent_paths[agent_idx] = path if path else []
            
            return Action.TOGGLE_LOAD.value

    elif phase == TaskPhase.DELIVER:
        if current_pos == assigned_goals[agent_idx] and carrying:
            # Mark task completion in time tracker first
            task_id = engaged_shelves[agent_idx]
            if task_id is not None:
                time_tracker.complete_task(task_id)
                
            task_states[agent_idx] = TaskPhase.RETURN_TO_PICKUP
            
            # Release previous reservations if using reservation table
            if reservation_table:
                reservation_table.release_reservations(agent_idx)
              
            if reservation_table is None:
                path = copp_astar(env.env.grid, current_pos, original_pickup[agent_idx], True, ReservationTable(), agent_idx)
                agent_paths[agent_idx] = path if path else []
            
            return Action.NOOP.value

    elif phase == TaskPhase.RETURN_TO_PICKUP:
        if current_pos == original_pickup[agent_idx] and carrying:
            carrying_shelf[agent_idx] = False
            task_states[agent_idx] = TaskPhase.DONE
            agent_paths[agent_idx] = None
            
            # Release reservations when done
            if reservation_table:
                reservation_table.release_reservations(agent_idx)
                
            return Action.TOGGLE_LOAD.value

    elif phase == TaskPhase.DONE:
        if sum(completed_tasks) < n_iterations:
            if env.env.request_queue:
                env.env.request_queue.pop(0)
                task_states[agent_idx] = TaskPhase.GO_TO_PICKUP
                original_pickup[agent_idx] = None
                assigned_goals[agent_idx] = None
                agent_paths[agent_idx] = None
                engaged_shelves[agent_idx] = None
                agent_priorities[agent_idx] = 0  # Reset priority when task is complete
            completed_tasks[agent_idx] += 1
        return Action.NOOP.value

    return None

# Global path coordination function
def coordinate_all_paths(env, agent_positions, carrying_shelf, task_states, original_pickup, assigned_goals, agent_priorities):
    num_agents = len(agent_positions)
    grid = env.env.grid
    reservation_table = ReservationTable()
    
    # Sort agents by priority (highest first)
    agent_indices = list(range(num_agents))
    sorted_agents = sorted(agent_indices, key=lambda i: (agent_priorities[i], -i), reverse=True)
    
    new_paths = [None for _ in range(num_agents)]
    
    for agent_idx in sorted_agents:
        current_pos = agent_positions[agent_idx]
        carrying = carrying_shelf[agent_idx]
        phase = task_states[agent_idx]
        
        # Skip agents without assigned tasks
        if phase == TaskPhase.DONE or (phase == TaskPhase.GO_TO_PICKUP and original_pickup[agent_idx] is None):
            continue
        
        # Determine goal based on task phase
        if phase == TaskPhase.GO_TO_PICKUP:
            goal = original_pickup[agent_idx]
            is_carrying = False
        elif phase == TaskPhase.DELIVER:
            goal = assigned_goals[agent_idx]
            is_carrying = True
        elif phase == TaskPhase.PICKUP_AGAIN:
            goal = assigned_goals[agent_idx]
            is_carrying = False
        elif phase == TaskPhase.RETURN_TO_PICKUP:
            goal = original_pickup[agent_idx]
            is_carrying = True
        else:
            continue
        
        if not goal:
            continue
            
        # Plan path using reservation table
        max_time = 10000
        path = coop_astar(grid, current_pos, goal, is_carrying, reservation_table, agent_idx, max_time)
        
        if path:
            # Try to reserve the path
            success, _ = reservation_table.add_path_reservation(path, agent_idx, is_carrying)
            if success:
                new_paths[agent_idx] = path
                print(f"[Agent {agent_idx}] Priority {agent_priorities[agent_idx]} path planned and reserved: {len(path)} steps")
            else:
                print(f"[Agent {agent_idx}] Failed to reserve path - conflict detected")
        else:
            print(f"[Agent {agent_idx}] Failed to find valid path")
    
    return new_paths

# Main Loop
def main_loop_with_priority(layout = None):
    # Start the overall timing
    overall_start_time = time.time()
    
    n_iterations = 20
    env = gym.make("rware-medium-2ag-easy-v2", request_queue_size=n_iterations, n_agents=20, num_high_priority=10, priority_level=55, layout = layout)
    obs, info = env.reset()
    completed_tasks = [0 for _ in range(env.env.n_agents)]
    carrying_shelf = [False for _ in range(env.env.n_agents)]
    task_states = [TaskPhase.GO_TO_PICKUP for _ in range(env.env.n_agents)]
    original_pickup = [None for _ in range(env.env.n_agents)]
    assigned_goals = [None for _ in range(env.env.n_agents)]
    agent_paths = [None for _ in env.env.agents]
    engaged_shelves = [None for _ in env.env.agents]
    agent_priorities = [0 for _ in env.env.agents]
    
    total_makespan = 0
    agent_travel_distances = [0 for _ in range(env.env.n_agents)]
    
    # Task time tracker
    time_tracker = TaskTimeTracker()
    
    # Add performance metrics
    performance_metrics = {
        'task_assignment_time': 0,
        'path_planning_time': 0,
        'execution_time': 0}
    
    # Global reservation table
    reservation_table = ReservationTable()
    
    # Path validation tracker
    path_execution_trackers = [0 for _ in env.env.agents]  # Tracks position in path
    
    # Task ID to priority mapping
    task_priority_map = {}  # Track task priorities
    
    for step in range(300):
        # Get agent positions
        positions = [(a.x, a.y) for a in env.env.agents]
        
        if sum(completed_tasks) >= n_iterations:
            total_makespan = time_tracker.current_time
            print(f"All tasks completed at time: {total_makespan}")
            break
        
        # Task assignment - only reassign if some agents are free
        free_agents = any(state == TaskPhase.GO_TO_PICKUP and orig is None 
                         for state, orig in zip(task_states, original_pickup))

        if free_agents:
            # Store current task priorities before assignment
            for task in env.env.request_queue:
                task_priority_map[task.id] = task.priority
                
            ta_start_time = time.time()
            assignments = cbta_assign_tasks_with_priority(env, positions, carrying_shelf, engaged_shelves, agent_paths, agent_priorities)
            
            ta_time = time.time() - ta_start_time
            performance_metrics['task_assignment_time'] += ta_time
            
            # Process new assignments
            for idx, assignment in enumerate(assignments):
                if assignment:
                    pickup, goal, _, _ = assignment
                    if task_states[idx] == TaskPhase.GO_TO_PICKUP and original_pickup[idx] is None:
                        original_pickup[idx] = pickup
                        assigned_goals[idx] = goal
                        
                        # Register task assignment with time tracker
                        if engaged_shelves[idx] is not None and engaged_shelves[idx] in task_priority_map:
                            task_id = engaged_shelves[idx]
                            priority = task_priority_map[task_id]
                            time_tracker.assign_task(task_id, priority)
        
        # Check if replanning is needed
        replan_needed = False
        
        # Verify path validity
        for agent_idx in range(env.env.n_agents):
            if agent_paths[agent_idx] and path_execution_trackers[agent_idx] < len(agent_paths[agent_idx]):
                expected_pos = agent_paths[agent_idx][path_execution_trackers[agent_idx]]
                actual_pos = positions[agent_idx]
                
                if expected_pos != actual_pos:
                    print(f"[Agent {agent_idx}] Path deviation detected. Expected {expected_pos}, actual {actual_pos}")
                    replan_needed = True
                    break
        
        # Major replanning done only when needed
        if step == 0 or replan_needed or step % 20 == 0:  # Initial plan + periodic safety replans
            print(f"\n[Step {step}] Global path coordination")
            # Release all reservations before replanning
            reservation_table = ReservationTable()
            
            pp_start_time = time.time()
            
            # Plan paths for all agents at once
            new_paths = coordinate_all_paths(env, positions, carrying_shelf, task_states, 
                                           original_pickup, assigned_goals, agent_priorities)
            
            pp_time = time.time() - pp_start_time
            performance_metrics['path_planning_time'] += pp_time
            
            # Update paths and reset execution trackers
            for idx, path in enumerate(new_paths):
                if path:
                    agent_paths[idx] = path
                    path_execution_trackers[idx] = 0
        
        # Execute actions
        actions = [None] * env.env.n_agents
        shelves_moving = set()  # Track which shelves are moving in this step
        
        for agent_idx in range(env.env.n_agents):
            agent = env.env.agents[agent_idx]
            current_pos = positions[agent_idx]
            carrying = carrying_shelf[agent_idx]
            phase = task_states[agent_idx]
            
            # Track task completion events
            prev_phase = phase
            
            # Handle task phase transitions first
            action = handle_task_phase(
                agent_idx, agent, current_pos, carrying, phase, env,
                original_pickup, assigned_goals, agent_paths, carrying_shelf,
                task_states, completed_tasks, engaged_shelves, n_iterations,
                agent_priorities, reservation_table, time_tracker
            )
            
            if action is not None:
                actions[agent_idx] = action
                continue

            # Follow current path if we have one
            path = agent_paths[agent_idx]
            if path and path_execution_trackers[agent_idx] < len(path) - 1:
                current_path_pos = path_execution_trackers[agent_idx]
                next_path_pos = min(current_path_pos + 1, len(path) - 1)
                
                expected_cur_pos = path[current_path_pos]
                if expected_cur_pos == current_pos:  # Only proceed if at expected position
                    action_list = get_actions_from_path(current_pos, agent.dir, path[current_path_pos:])
                    if action_list:
                        actions[agent_idx] = action_list[0]
                        if action_list[0] == Action.FORWARD.value:
                            agent_travel_distances[agent_idx] += 1
                            path_execution_trackers[agent_idx] += 1  # Update path tracker
                            # If carrying a shelf, track that it's moving
                            if carrying and engaged_shelves[agent_idx] is not None:
                                shelves_moving.add(engaged_shelves[agent_idx])
                        continue
            
            # Fallback: No path or can't follow path
            actions[agent_idx] = Action.NOOP.value
        
        # Update time tracker if any shelves are moving
        time_tracker.tick_time(shelves_moving)
        
        # Execute actions
        ex_start_time = time.time()
        obs, rewards, done, _, _ = env.step(actions)
        
        ex_time = time.time() - ex_start_time
        performance_metrics['execution_time'] += ex_time
        
        env.render()
        time.sleep(0.05)
        
        if done:
            break

    # Calculate total runtime
    total_runtime = time.time() - overall_start_time
    
    # Print final statistics
    stats = time_tracker.get_task_stats()
    total_distance = sum(agent_travel_distances)
    avg_distance = total_distance / env.env.n_agents
    total_tasks_completed = sum(completed_tasks)
    overall_throughput = total_tasks_completed / time_tracker.current_time if time_tracker.current_time > 0 else 0
    high_priority_throughput = stats['total'] / time_tracker.current_time if time_tracker.current_time > 0 else 0
    
    print(f"\n============ Runtime Performance Summary ============")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Task assignment time: {performance_metrics['task_assignment_time']:.2f} seconds ({performance_metrics['task_assignment_time'] / total_runtime * 100:.1f}%)")
    print(f"Path planning time: {performance_metrics['path_planning_time']:.2f} seconds ({performance_metrics['path_planning_time'] / total_runtime * 100:.1f}%)")
    print(f"Execution time: {performance_metrics['execution_time']:.2f} seconds ({performance_metrics['execution_time'] / total_runtime * 100:.1f}%)")
    
    print("\n============ Task Delivery Performance ============")
    print(f"Total Tasks Completed: {total_tasks_completed}")
    print(f"Overall Throughput: {overall_throughput:.3f} tasks/time unit")
    print(f"Total High-priority Tasks Completed: {stats['total']}")
    print(f"High-priority Task Throughput: {high_priority_throughput:.3f} tasks/time unit")
    print(f"Tasks Delivered on Time: {stats['on_time']} ({stats['on_time_percentage']:.1f}%)")
    print(f"Tasks Delivered Late: {stats['late']}")
    print(f"Total Makespan: {total_makespan}")
    print(f"Total Travel Distance: {total_distance}")
    print(f"Average Travel Distance per Agent: {avg_distance:.2f}")
    
    print("===================================================")
            
    env.close()

main_loop_with_priority()