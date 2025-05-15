import random
import numpy as np
import pickle
import os
import time
import pygame
import sys
import matplotlib.pyplot as plt
import imageio


def draw_board_image(board, filename):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='black', linewidth=2)

    for r in range(3):
        for c in range(3):
            symbol = board[r][c]
            if symbol:
                ax.text(c + 0.5, 2.5 - r, symbol, size=40,
                        ha='center', va='center',
                        color='blue' if symbol == 'X' else 'red')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def gif_agent_vs_agent(board_size=3, save_path="game.gif"):
    frames = []
    board = [['' for _ in range(board_size)] for _ in range(board_size)]
    players = ['X', 'O']
    current = 0
    game_over = False
    step = 0

    while not game_over:
        state = board_to_tuple(board)
        moves = available_moves(board)
        if not moves:
            break

        action = choose_action(state, moves, epsilon=0.0)
        board[action[0]][action[1]] = players[current]

        filename = f"frame_{step}.png"
        draw_board_image(board, filename)
        frames.append(filename)
        step += 1

        if check_winner(board, players[current]) or is_full(board):
            filename = f"frame_{step}.png"
            draw_board_image(board, filename)
            frames.append(filename)
            break

        current = 1 - current

    # Сделать gif
    images = [imageio.v2.imread(f) for f in frames]
    imageio.mimsave(save_path, images, duration=0.8)

    # Очистка временных кадров
    for f in frames:
        os.remove(f)

    print(f"🎉 Игра сохранена в {save_path}")

# --- Pygame визуализация ---
CELL_SIZE = 100
LINE_WIDTH = 5
X_COLOR = (66, 66, 255)
O_COLOR = (255, 66, 66)
LINE_COLOR = (0, 0, 0)
BG_COLOR = (255, 255, 255)

import matplotlib.pyplot as plt
import imageio

def draw_board(board, game_num, move_num):
    fig, ax = plt.subplots()
    ax.set_title(f"Game {game_num+1} - Move {move_num+1}")
    size = len(board)

    for r in range(size):
        for c in range(size):
            cell = board[r][c]
            ax.text(c, size - r - 1, cell, ha='center', va='center', fontsize=24)
            ax.plot([c-0.5, c+0.5], [size - r - 0.5, size - r - 0.5], color='black')  # top
            ax.plot([c-0.5, c+0.5], [size - r + 0.5, size - r + 0.5], color='black')  # bottom
            ax.plot([c - 0.5, c - 0.5], [size - r - 0.5, size - r + 0.5], color='black')  # left
            ax.plot([c + 0.5, c + 0.5], [size - r - 0.5, size - r + 0.5], color='black')  # right

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.axis('off')

    filename = f"frame_{game_num}_{move_num}.png"
    plt.savefig(filename)
    plt.close()
    return filename


def visualize_agent_vs_agent(board_size=3, rounds=10):
    pygame.init()
    screen = pygame.display.set_mode((CELL_SIZE * board_size, CELL_SIZE * board_size))
    pygame.display.set_caption("🤖 Agent vs Agent: Tic Tac Toe")

    for episode in range(rounds):
        board = [['' for _ in range(board_size)] for _ in range(board_size)]
        players = ['X', 'O']
        current = 0
        game_over = False

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            state = board_to_tuple(board)
            moves = available_moves(board)

            if not moves:
                game_over = True
                break

            action = choose_action(state, moves, epsilon=0.0)
            board[action[0]][action[1]] = players[current]

            draw_board(screen, board)
            pygame.time.delay(400)

            if check_winner(board, players[current]) or is_full(board):
                draw_board(screen, board)
                pygame.time.delay(800)
                game_over = True

            current = 1 - current

    pygame.quit()


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
    with open(q_table_file, 'wb') as f:
        pickle.dump(q_table, f)

def load_q_table():
    global q_table
    if os.path.exists(q_table_file):
        with open(q_table_file, 'rb') as f:
            q_table = pickle.load(f)

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
    if random.random() < epsilon:
        return random.choice(moves)
    q_vals = [q_table.get((state, a), 0) for a in moves]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(moves, q_vals) if q == max_q]
    return random.choice(best_actions)

def update_q_table(state, action, reward, next_state, next_moves):
    old_q = q_table.get((state, action), 0)
    future_q = max([q_table.get((next_state, a), 0) for a in next_moves], default=0)
    q_table[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

def train_agents(board_size):
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

    save_q_table()

def agent_vs_agent(board_size, rounds=10):
    all_frames = []

    for episode in range(rounds):
        board = [['' for _ in range(board_size)] for _ in range(board_size)]
        players = ['X', 'O']
        random.shuffle(players)
        current = 0
        move_num = 0

        while True:
            state = board_to_tuple(board)
            moves = available_moves(board)
            if not moves:
                result = "Draw"
                break

            action = choose_action(state, moves, epsilon=0.0)
            board[action[0]][action[1]] = players[current]

            # Добавляем кадр
            frame = draw_board(board, episode, move_num)
            all_frames.append(frame)
            move_num += 1

            if check_winner(board, players[current]):
                result = f"Agent {players[current]} wins"
                break

            if is_full(board):
                result = "Draw"
                break

            current = 1 - current

        print(f"Game {episode+1}/{rounds} result: {result}")

    # Сохраняем одну гифку после всех игр
    images = [imageio.v2.imread(f) for f in all_frames]
    imageio.mimsave("full_10_games.gif", images, duration=0.8)

    print("✅ Сохранено в full_10_games.gif")

    # Очистим PNG
    for f in all_frames:
        os.remove(f)

if __name__ == "__main__":
    board_size = 3
    if not os.path.exists(q_table_file):
        train_agents(board_size)
    load_q_table()
    gif_agent_vs_agent(board_size=3, save_path="agent_game.gif")
