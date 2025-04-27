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
    
class TaskTimeTracker:
    def __init__(self):
        self.task_start_times = {}  # task_id -> start time
        self.task_deadlines = {}    # task_id -> deadline (start + priority)
        self.task_completion_times = {}  # task_id -> completion time
        self.current_time = 0  # global time counter for shelf movements
        self.task_priorities = {}  # task_id -> priority level
        
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

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Space Time A*
def space_time_astar(grid, start, goal, carrying, dynamic_obstacles_pack):
    
    # Unpack dynamic obstacles
    dynamic_obstacles, edge_obstacles = dynamic_obstacles_pack
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
            goal_freed = None
            for wait_t in range(current.t, current.t + GOAL_BUFFER_TIME + 1):
                if (goal[0], goal[1]) not in dynamic_obstacles.get(wait_t, set()):
                    goal_freed = wait_t
                    break

            if goal_freed is not None:
                temp = current
                while temp.t < goal_freed:
                    temp = STANode(current.x, current.y, temp.t + 1, parent=temp)
                path = []
                while temp:
                    path.append((temp.x, temp.y))
                    temp = temp.parent
                return path[::-1]
            else:
                print(f"[STA*] Goal {goal} remains blocked from t={current.t} to t={current.t + GOAL_BUFFER_TIME}")

        closed_set.add((current.x, current.y, current.t))

        # Explore neighbors (including waiting)
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = current.x + dx, current.y + dy
            nt = current.t + 1
            
            # Check if new position is within grid bounds
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            
            # Skip if already explored
            if (nx, ny, nt) in closed_set:
                continue
            
            # Skip if there's a dynamic obstacle at the new position and time
            if (nx, ny) in dynamic_obstacles.get(nt, set()):
                continue
            
             # Skip if there's a collision with static obstacles (unless it's the goal)
            if (nx, ny) != goal and any(grid[layer, ny, nx] > 0 for layer in collision_layers):
                continue
            
            # Check for edge constraints
            edge = ((current.x, current.y), (nx, ny))
            reverse_edge = ((nx, ny), (current.x, current.y))
            if edge in edge_obstacles.get(current.t, set()) or reverse_edge in edge_obstacles.get(current.t, set()):
                continue

            key = (nx, ny, nt)
            tentative_g = current.g + 1

            # Skip if we already have a better path to this node
            if key in g_score and tentative_g >= g_score[key]:
                continue

            g_score[key] = tentative_g
            new_node = STANode(nx, ny, nt, current)
            new_node.g = tentative_g
            new_node.h = heuristic((nx, ny), goal)
            new_node.f = new_node.g + new_node.h
            heapq.heappush(open_list, new_node)

        # Consider waiting in place as a special case
        wait_key = (current.x, current.y, current.t + 1)
        if wait_key not in closed_set and (current.x, current.y) not in dynamic_obstacles.get(current.t + 1, set()):
            tentative_g = current.g + 1.05
            if wait_key not in g_score or tentative_g < g_score[wait_key]:
                g_score[wait_key] = tentative_g
                wait_node = STANode(current.x, current.y, current.t + 1, current)
                wait_node.g = tentative_g
                wait_node.h = heuristic((current.x, current.y), goal)
                wait_node.f = wait_node.g + wait_node.h
                heapq.heappush(open_list, wait_node)

    print(f"[STA*] Failed to find a path from {start} to {goal}")
    return []

# Building the dynamic obstacles
def build_dynamic_obstacles_with_priority(agent_idx, planned_paths, agent_priorities):
    obstacles = {}
    edge_obstacles = {}

    # Only consider agents with higher priority
    for idx, path in enumerate(planned_paths):
        if path is None or len(path) == 0 or idx == agent_idx:
            continue
        if agent_priorities[idx] < agent_priorities[agent_idx]:
            continue
        if agent_priorities[idx] == agent_priorities[agent_idx] and idx >= agent_idx:
            continue

        # Process each position in the path
        for t, pos in enumerate(path):
            # Always block the current position
            if t not in obstacles:
                obstacles[t] = set()
            obstacles[t].add(pos)
                        
            if t > 0:
                prev = path[t - 1]
                if t - 1 not in edge_obstacles:
                    edge_obstacles[t - 1] = set()
                edge_obstacles[t - 1].add((prev, pos))
                edge_obstacles[t - 1].add((pos, prev))

        # Pad final position
        final_pos = path[-1]
        for pad in range(GOAL_BUFFER_TIME):
            t_pad = len(path) + pad
            if t_pad not in obstacles:
                obstacles[t_pad] = set()
            obstacles[t_pad].add(final_pos)

    return obstacles, edge_obstacles

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
def cbta_assign_tasks_with_priority(env, agent_positions, carrying_shelf, engaged_shelves, 
                                    agent_paths, agent_priorities, time_tracker, step):
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
                
            # Get agent position and build obstacles
            agent_pos = agent_positions[agent_idx]
            obstacles_pack = build_dynamic_obstacles_with_priority(agent_idx, agent_paths, agent_priorities)
            
            # Find path to pickup
            path_to_pickup = space_time_astar(grid, agent_pos, pickup, False, obstacles_pack)
            if not path_to_pickup:
                continue
                
            # Find best goal and path to it
            best_goal_cost = float('inf')
            best_goal_path = None
            best_goal_pos = None
            
            for goal in goals:
                path_to_goal = space_time_astar(grid, pickup, goal, True, obstacles_pack) 
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
            
            # Record task assignment in time tracker
            if task.priority > 0:
                time_tracker.assign_task(task.id, task.priority)
            
            # Update agent priority based on task priority
            agent_priorities[best_agent] = task.priority
            print(f"Assigned task {task.id} (priority {task.priority}) to agent {best_agent}")
            
    return assignments


# Handle task phase transitions with time tracking
def handle_task_phase(agent_idx, agent, current_pos, carrying, phase, env, original_pickup, assigned_goals, agent_paths, carrying_shelf, 
                      task_states, completed_tasks, engaged_shelves, n_iterations, agent_priorities, time_tracker, step):
    """Handle task phase transitions with respect to agent priorities"""
    if phase == TaskPhase.GO_TO_PICKUP:
        if original_pickup[agent_idx] and current_pos == original_pickup[agent_idx] and not carrying:
            carrying_shelf[agent_idx] = True
            task_states[agent_idx] = TaskPhase.DELIVER
            
            # High priority planning happens first
            obstacles_pack = build_dynamic_obstacles_with_priority(agent_idx, agent_paths, agent_priorities)
            path = space_time_astar(env.env.grid, current_pos, assigned_goals[agent_idx], True, obstacles_pack)
            agent_paths[agent_idx] = path if path else []
            return Action.TOGGLE_LOAD.value

    elif phase == TaskPhase.DELIVER:
        if current_pos == assigned_goals[agent_idx] and carrying:
            # Check if task was delivered on time using the time tracker
            task_id = engaged_shelves[agent_idx]
            on_time, delay = time_tracker.complete_task(task_id)
            
            task_states[agent_idx] = TaskPhase.RETURN_TO_PICKUP
            
            obstacles_pack = build_dynamic_obstacles_with_priority(agent_idx, agent_paths, agent_priorities)
            path = space_time_astar(env.env.grid, current_pos, original_pickup[agent_idx], True, obstacles_pack)
            agent_paths[agent_idx] = path if path else []
            
            return Action.NOOP.value

    elif phase == TaskPhase.RETURN_TO_PICKUP:
        if current_pos == original_pickup[agent_idx] and carrying:
            carrying_shelf[agent_idx] = False
            task_states[agent_idx] = TaskPhase.DONE
            agent_paths[agent_idx] = None
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

# Replan a single agent's path considering the agent's priority
def replan_path_for_agent(agent_idx, positions, agent_paths, task_states, original_pickup,
                         assigned_goals, carrying_shelf, grid, agent_priorities):
    
    phase = task_states[agent_idx]
    current_pos = positions[agent_idx]
    obstacles_pack = build_dynamic_obstacles_with_priority(agent_idx, agent_paths, agent_priorities)
    
    # Map phase to goal and carrying state
    goal = None
    is_carrying = carrying_shelf[agent_idx]
    
    if phase in (TaskPhase.GO_TO_PICKUP, TaskPhase.RETURN_TO_PICKUP):
        goal = original_pickup[agent_idx]
    elif phase in (TaskPhase.DELIVER, TaskPhase.PICKUP_AGAIN):
        goal = assigned_goals[agent_idx]
    
    # Skip if no goal or phase not handled
    if not goal:
        return None
        
    # Plan new path
    new_path = space_time_astar(grid, current_pos, goal, is_carrying, obstacles_pack)
    
    if new_path:
        print(f"[Agent {agent_idx}] Priority {agent_priorities[agent_idx]} path planned: {len(new_path)} steps")
        agent_paths[agent_idx] = new_path
        
    return new_path

def main_loop_with_priority(layout = None):
    # Start the overall timing
    overall_start_time = time.time()
    
    n_iterations = 10
    env = gym.make("rware-medium-2ag-easy-v2", request_queue_size=n_iterations, n_agents=10, num_high_priority=2, priority_level=55, layout = layout)
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
    
    time_tracker = TaskTimeTracker()
    
    performance_metrics = {
        'task_assignment_time': 0,
        'path_planning_time': 0,
        'execution_time': 0}
    
    for step in range(250):
        # Get agent positions
        positions = [(a.x, a.y) for a in env.env.agents]
        
        if sum(completed_tasks) >= n_iterations:
            total_makespan = time_tracker.current_time
            print(f"All tasks completed at time: {total_makespan}")
            break
            
        # Task assignment - give high priority tasks to agents that are free
        if step % 2 == 0:  # Periodic reassignment
            ta_start_time = time.time()
            assignments = cbta_assign_tasks_with_priority(env, positions, carrying_shelf, engaged_shelves, 
                                                          agent_paths, agent_priorities, time_tracker, step)
            
            ta_time = time.time() - ta_start_time
            performance_metrics['task_assignment_time'] += ta_time
            
            for idx, assignment in enumerate(assignments):
                if assignment:
                    pickup, goal, path_pickup, path_goal = assignment
                    if task_states[idx] == TaskPhase.GO_TO_PICKUP and original_pickup[idx] is None:
                        original_pickup[idx] = pickup
                        assigned_goals[idx] = goal
                        agent_paths[idx] = path_pickup
                        
        # Track which shelves are moving for time tracker
        moving_shelves = [engaged_shelves[idx] for idx in range(env.env.n_agents) 
                         if carrying_shelf[idx] and engaged_shelves[idx] is not None]
        time_tracker.tick_time(moving_shelves)
        
        # Plan paths in priority order
        agent_indices = list(range(env.env.n_agents))
        priority_sorted_indices = sorted(agent_indices, key=lambda i: (agent_priorities[i], -i), reverse=True)
        
        pp_start_time = time.time()
        
        # First planning pass: High priority agents plan freely
        for agent_idx in priority_sorted_indices:
            if agent_priorities[agent_idx] > 0:  # High priority
                replan_path_for_agent(agent_idx, positions, agent_paths, task_states, 
                                      original_pickup, assigned_goals, carrying_shelf, 
                                      env.env.grid, agent_priorities)
        
        # Second planning pass: Low priority agents plan around high priority ones
        for agent_idx in priority_sorted_indices:
            if agent_priorities[agent_idx] == 0:  # Low priority
                replan_path_for_agent(agent_idx, positions, agent_paths, task_states, 
                                     original_pickup, assigned_goals, carrying_shelf, 
                                     env.env.grid, agent_priorities)
        
        
        pp_time = time.time() - pp_start_time
        performance_metrics['path_planning_time'] += pp_time
        
        # Execute actions in priority order
        exec_start_time = time.time()
        actions = [None] * env.env.n_agents
        for agent_idx in priority_sorted_indices:
            agent = env.env.agents[agent_idx]
            current_pos = positions[agent_idx]
            carrying = carrying_shelf[agent_idx]
            phase = task_states[agent_idx]
            
            # Handle task phase transitions first
            action = handle_task_phase(
                agent_idx, agent, current_pos, carrying, phase, env,
                original_pickup, assigned_goals, agent_paths, carrying_shelf,
                task_states, completed_tasks, engaged_shelves, n_iterations,
                agent_priorities, time_tracker, step
            )
            
            if action is not None:
                actions[agent_idx] = action
                continue
            
            # Follow current path if we have one
            path = agent_paths[agent_idx]
            if path and len(path) > 1:
                action_list = get_actions_from_path(current_pos, agent.dir, path)
                if action_list:
                    actions[agent_idx] = action_list[0]
                    if action_list[0] == Action.FORWARD.value:
                        agent_travel_distances[agent_idx] += 1
                        agent_paths[agent_idx] = path[1:]  # Trim path
                    continue
            
            # Fallback: No path or can't follow path
            actions[agent_idx] = Action.NOOP.value
            
            # Try to replan if blocked
            if path and len(path) > 1:
                print(f"[Agent {agent_idx}] Priority {agent_priorities[agent_idx]} blocked at {current_pos}. Replanning...")
                replan_path_for_agent(agent_idx, positions, agent_paths, task_states,
                                     original_pickup, assigned_goals, carrying_shelf,
                                     env.env.grid, agent_priorities)
                        
        # Execute actions
        obs, rewards, done, _, _ = env.step(actions)
        exec_time = time.time() - exec_start_time
        performance_metrics['execution_time'] += exec_time
        
        env.render()
        time.sleep(0.05)
        if done:
            break
        
    # Calculate total runtime
    total_runtime = time.time() - overall_start_time
    
    # Print delivery statistics at the end
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