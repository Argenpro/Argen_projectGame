import random
import numpy as np
import pickle
import os
import time

# --- Q-learning параметры ---
gamma = 0.9
alpha = 0.1
initial_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.999
train_episodes = 20000
q_table_file = "q_table.pkl"

q_table = dict()


def save_q_table():
    """Сохранение Q-таблицы в файл"""
    with open(q_table_file, 'wb') as f:
        pickle.dump(q_table, f)

def load_q_table():
    """Загрузка Q-таблицы из файла"""
    global q_table
    if os.path.exists(q_table_file):
        with open(q_table_file, 'rb') as f:
            q_table = pickle.load(f)
    else:
        print("Q-table not found! Starting training...")
        train_agents(board_size=3)  # если файла нет — обучаем заново

def board_to_tuple(board):
    return tuple(tuple(row) for row in board)

def available_moves(board):
    return [(r, c) for r in range(len(board)) for c in range(len(board)) if board[r][c] == '']

def check_winner(board, player):
    size = len(board)
    lines = board + list(zip(*board))
    diagonals = [[board[i][i] for i in range(size)], [board[i][size - 1 - i] for i in range(size)]]
    lines.extend(diagonals)
    for line in lines:
        if all(cell == player for cell in line):
            return True
    return False

def is_full(board):
    return all(all(cell != '' for cell in row) for row in board)

def choose_action(state, moves, epsilon):
    """Выбор хода на основе Q-таблицы или случайным образом"""
    if random.random() < epsilon:
        return random.choice(moves)
    q_vals = [q_table.get((state, a), 0) for a in moves]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(moves, q_vals) if q == max_q]
    return random.choice(best_actions)

def update_q_table(state, action, reward, next_state, next_moves):
    """Обновление Q-таблицы на основе формулы Q-learning"""
    old_q = q_table.get((state, action), 0)
    future_q = max([q_table.get((next_state, a), 0) for a in next_moves], default=0)
    q_table[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

def train_agents(board_size):
    """Обучение агентов на заданной доске с увеличением эпизодов и стратегии epsilon-greedy"""
    epsilon = initial_epsilon
    for episode in range(train_episodes):
        board = [['' for _ in range(board_size)] for _ in range(board_size)]
        players = ['X', 'O']
        random.shuffle(players)
        current = 0
        game_over = False
        history = []

        while not game_over:
            state = board_to_tuple(board)
            moves = available_moves(board)
            action = choose_action(state, moves, epsilon)

            board[action[0]][action[1]] = players[current]
            next_state = board_to_tuple(board)
            next_moves = available_moves(board)

            history.append((state, action, players[current]))

            if check_winner(board, players[current]):
                for s, a, p in history:
                    reward = 1 if p == players[current] else -1
                    update_q_table(s, a, reward, next_state, next_moves)
                game_over = True
                break

            if is_full(board):
                for s, a, _ in history:
                    update_q_table(s, a, 0.3, next_state, next_moves)
                game_over = True
                break

            current = 1 - current

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}/{train_episodes}, Epsilon: {epsilon:.4f}")

    save_q_table()  # Сохраняем Q-таблицу после обучения

def agent_vs_agent(board_size, rounds=10):
    """Игра между двумя агентами"""
    for episode in range(rounds):
        board = [['' for _ in range(board_size)] for _ in range(board_size)]
        players = ['X', 'O']
        random.shuffle(players)
        current = 0

        while True:
            state = board_to_tuple(board)
            moves = available_moves(board)
            if not moves:
                result = "Draw"
                break

            action = choose_action(state, moves, epsilon=0.0)
            board[action[0]][action[1]] = players[current]

            if check_winner(board, players[current]):
                result = f"Agent {players[current]} wins"
                break

            if is_full(board):
                result = "Draw"
                break

            current = 1 - current

        print(f"Game {episode+1}/{rounds} result: {result}")
        time.sleep(0.05)
