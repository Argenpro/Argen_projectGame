from q_logic import load_q_table, agent_vs_agent, train_agents
import os


def main():
    board_size = 3  # Или 4 на 4, если хочешь большую доску
    if not os.path.exists("q_table.pkl"):
        train_agents(board_size)  # Обучаем агентов, если модели нет
    load_q_table()  # Загружаем сохраненную модель (Q-таблицу)
    agent_vs_agent(board_size, rounds=500)  # Играем агенты против агентов


if __name__ == "__main__":
    main()
